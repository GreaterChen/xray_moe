import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import MistralForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
import gc
from utils import analyze_gpu_memory

class MistralFinetuner(nn.Module):
    def __init__(
        self,
        config,
        base_model="mistralai/Mistral-7B-v0.1",
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        visual_dim=768,
        text_dim=768,
        load_in_4bit=True,
    ):
        super(MistralFinetuner, self).__init__()
        
        self.config = config
        
        # 设置BitsAndBytes配置
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            bnb_config = None
        
        # 加载基础Mistral模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = MistralForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        
        # 添加特殊标记
        special_tokens = {
            "additional_special_tokens": [
                "<image>", "</image>", "<history>", "</history>", "<findings>"
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # 为LoRA准备模型
        if load_in_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # 配置LoRA
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        
        # 应用LoRA配置
        self.model = get_peft_model(self.model, peft_config)
        
        # 视觉和文本特征投影层
        self.visual_projection = nn.Linear(visual_dim, self.model.config.hidden_size)
        self.text_projection = nn.Linear(text_dim, self.model.config.hidden_size)
        
    def forward(
        self,
        visual_features,
        history_encoding=None,
        findings=None,
        attention_mask=None,
        labels=None,
    ):
        """
        前向传播
        
        Args:
            visual_features: 视觉特征 [batch_size, num_tokens, visual_dim]
            history_encoding: 历史文本的编码 {input_ids, attention_mask}
            findings: 报告文本
            attention_mask: 注意力掩码 [batch_size, seq_len]
            labels: 标签 [batch_size, seq_len]
        """
        batch_size = visual_features.shape[0]
        device = visual_features.device
        
        # 将视觉特征映射到模型维度
        projected_visual = self.visual_projection(visual_features)
        
        # 构建标准顺序的提示文本：<image> [IMAGE] </image> <history> [HISTORY] </history> <findings>
        prompts = ["<image> </image> <history> </history> <findings>" for _ in range(batch_size)]
        
        prompt_encoding = self.tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        prompt_input_ids = prompt_encoding.input_ids
        prompt_attention_mask = prompt_encoding.attention_mask
        
        # 并行处理整个批次的序列构建
        # 找到所有样本中<image>和</history>标记的位置（由于prompt相同，所有样本的位置都相同）
        image_start_idx = torch.where(prompt_input_ids[0] == self.tokenizer.convert_tokens_to_ids("<image>"))[0][0].item()
        image_end_idx = torch.where(prompt_input_ids[0] == self.tokenizer.convert_tokens_to_ids("</image>"))[0][0].item()
        
        # 提取前缀和后缀token IDs并获取嵌入
        prefix_ids = prompt_input_ids[0, :image_start_idx+1]  # 包含<image>标记
        suffix_ids = prompt_input_ids[0, image_end_idx:]      # 包含</image>标记
        
        prefix_embeds = self.model.get_input_embeddings()(prefix_ids)
        suffix_embeds = self.model.get_input_embeddings()(suffix_ids)
        
        # 并行构建输入嵌入和注意力掩码
        # 批量扩展前缀和后缀嵌入
        batch_prefix_embeds = prefix_embeds.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, prefix_len, hidden_size]
        batch_suffix_embeds = suffix_embeds.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, suffix_len, hidden_size]
        
        # 计算前缀、视觉特征和后缀的长度
        prefix_len = batch_prefix_embeds.size(1)
        visual_len = projected_visual.size(1)
        suffix_len = batch_suffix_embeds.size(1)
        total_len = prefix_len + visual_len + suffix_len
        
        # 创建输出张量
        inputs_embeds = torch.zeros(batch_size, total_len, projected_visual.size(2), device=device)
        attention_mask = torch.ones(batch_size, total_len, dtype=torch.long, device=device)
        
        # 填充各部分
        inputs_embeds[:, :prefix_len, :] = batch_prefix_embeds
        inputs_embeds[:, prefix_len:prefix_len+visual_len, :] = projected_visual
        inputs_embeds[:, prefix_len+visual_len:, :] = batch_suffix_embeds
        
        # 如果有历史文本，处理历史文本部分
        if history_encoding is not None:
            # 获取历史标记的位置（由于prompt相同，所有样本的位置都相同）
            history_start_idx = torch.where(prompt_input_ids[0] == self.tokenizer.convert_tokens_to_ids("<history>"))[0][0].item()
            history_end_idx = torch.where(prompt_input_ids[0] == self.tokenizer.convert_tokens_to_ids("</history>"))[0][0].item()
            
            # 找到嵌入中的位置索引
            prefix_len = prefix_embeds.size(0)
            visual_len = projected_visual.size(1)
            
            # 计算history在嵌入序列中的起始和结束位置
            history_start_embed = prefix_len + visual_len + 1  # +1 是为了包含</image>标记后的<history>标记
            
            # 提取每个样本的前缀和后缀
            prefix_slice = slice(0, history_start_embed)
            suffix_slice = slice(history_start_embed + 1, None)  # +1 是为了跳过<history>标记
            
            batch_prefix = inputs_embeds[:, prefix_slice]
            batch_suffix = inputs_embeds[:, suffix_slice]
            
            # 获取历史文本嵌入
            history_ids = history_encoding["input_ids"]
            
            # 获取最大历史长度用于创建输出张量
            max_history_len = history_ids.size(1)
            
            # 为历史文本创建批量嵌入
            batch_history_embeds = self.model.get_input_embeddings()(history_ids)
            
            # 计算新序列的总长度
            prefix_len = batch_prefix.size(1)
            suffix_len = batch_suffix.size(1)
            total_len = prefix_len + max_history_len + suffix_len
            
            # 创建新的输出张量
            new_inputs_embeds = torch.zeros(batch_size, total_len, inputs_embeds.size(2), device=device)
            new_attention_mask = torch.ones(batch_size, total_len, dtype=torch.long, device=device)
            
            # 填充各部分
            new_inputs_embeds[:, :prefix_len] = batch_prefix
            new_inputs_embeds[:, prefix_len:prefix_len+max_history_len] = batch_history_embeds
            new_inputs_embeds[:, prefix_len+max_history_len:] = batch_suffix
            
            # 更新
            inputs_embeds = new_inputs_embeds
            attention_mask = new_attention_mask
        
        # 处理findings部分（如果在训练模式下）
        if findings is not None:
            # findings已经是编码好的input_ids和attention_mask
            findings_input_ids = findings["input_ids"]
            findings_attention_mask = findings["attention_mask"]
            
            # 获取findings的嵌入部分
            findings_embeds = self.model.get_input_embeddings()(findings_input_ids[:, 1:])  # 跳过BOS
            
            # 当前序列长度作为findings的起始位置
            prefix_len = inputs_embeds.size(1)
            
            # 获取批量的findings部分长度
            findings_len = findings_embeds.size(1)
            
            # 计算新序列的总长度
            total_len = prefix_len + findings_len
            
            # 创建新的输出张量
            new_inputs_embeds = torch.zeros(batch_size, total_len, inputs_embeds.size(2), device=device)
            new_attention_mask = torch.zeros(batch_size, total_len, dtype=torch.long, device=device)
            
            # 创建标签张量（只对findings部分计算损失）
            labels = torch.full((batch_size, total_len), -100, device=device)
            
            # 填充各部分
            new_inputs_embeds[:, :prefix_len] = inputs_embeds
            new_inputs_embeds[:, prefix_len:] = findings_embeds
            
            # 设置注意力掩码
            new_attention_mask[:, :prefix_len] = attention_mask
            new_attention_mask[:, prefix_len:] = findings_attention_mask[:, 1:]  # 跳过BOS
            
            # 设置标签（只对findings部分计算损失）
            labels[:, prefix_len:] = findings_input_ids[:, 1:]  # 从第二个token开始（跳过BOS）
            
            # 更新
            inputs_embeds = new_inputs_embeds
            attention_mask = new_attention_mask
        
        # 运行模型
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels if findings is not None else None,
            return_dict=True,
        )
        
        return outputs
    
    def generate(
        self,
        visual_features,
        history_encoding=None,
        max_length=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    ):
        """
        生成报告文本
        
        Args:
            visual_features: 视觉特征 [batch_size, num_tokens, visual_dim]
            history_encoding: 历史文本的编码 {input_ids, attention_mask}
            max_length: 生成文本的最大长度
            do_sample: 是否采样生成
            temperature: 温度参数
            top_p: 概率截断阈值
        """
        batch_size = visual_features.shape[0]
        device = visual_features.device
        
        # 将视觉特征映射到模型维度
        projected_visual = self.visual_projection(visual_features)
        
        # 构建标准顺序的提示文本：<image> [IMAGE] </image> <history> [HISTORY] </history> <findings>
        prompts = ["<image> </image> <history> </history> <findings>" for _ in range(batch_size)]
        
        prompt_encoding = self.tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        prompt_input_ids = prompt_encoding.input_ids
        prompt_attention_mask = prompt_encoding.attention_mask
        
        # 并行处理整个批次的序列构建
        # 找到所有样本中<image>和</history>标记的位置（由于prompt相同，所有样本的位置都相同）
        image_start_idx = torch.where(prompt_input_ids[0] == self.tokenizer.convert_tokens_to_ids("<image>"))[0][0].item()
        image_end_idx = torch.where(prompt_input_ids[0] == self.tokenizer.convert_tokens_to_ids("</image>"))[0][0].item()
        
        # 提取前缀和后缀token IDs并获取嵌入
        prefix_ids = prompt_input_ids[0, :image_start_idx+1]  # 包含<image>标记
        suffix_ids = prompt_input_ids[0, image_end_idx:]      # 包含</image>标记
        
        prefix_embeds = self.model.get_input_embeddings()(prefix_ids)
        suffix_embeds = self.model.get_input_embeddings()(suffix_ids)
        
        # 并行构建输入嵌入和注意力掩码
        # 批量扩展前缀和后缀嵌入
        batch_prefix_embeds = prefix_embeds.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, prefix_len, hidden_size]
        batch_suffix_embeds = suffix_embeds.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, suffix_len, hidden_size]
        
        # 计算前缀、视觉特征和后缀的长度
        prefix_len = batch_prefix_embeds.size(1)
        visual_len = projected_visual.size(1)
        suffix_len = batch_suffix_embeds.size(1)
        total_len = prefix_len + visual_len + suffix_len
        
        # 创建输出张量
        inputs_embeds = torch.zeros(batch_size, total_len, projected_visual.size(2), device=device)
        attention_mask = torch.ones(batch_size, total_len, dtype=torch.long, device=device)
        
        # 填充各部分
        inputs_embeds[:, :prefix_len, :] = batch_prefix_embeds
        inputs_embeds[:, prefix_len:prefix_len+visual_len, :] = projected_visual
        inputs_embeds[:, prefix_len+visual_len:, :] = batch_suffix_embeds
        
        # 如果有历史文本，处理历史文本部分
        if history_encoding is not None:
            # 获取历史标记的位置（由于prompt相同，所有样本的位置都相同）
            history_start_idx = torch.where(prompt_input_ids[0] == self.tokenizer.convert_tokens_to_ids("<history>"))[0][0].item()
            history_end_idx = torch.where(prompt_input_ids[0] == self.tokenizer.convert_tokens_to_ids("</history>"))[0][0].item()
            
            # 找到嵌入中的位置索引
            prefix_len = prefix_embeds.size(0)
            visual_len = projected_visual.size(1)
            
            # 计算history在嵌入序列中的起始和结束位置
            history_start_embed = prefix_len + visual_len + 1  # +1 是为了包含</image>标记后的<history>标记
            
            # 提取每个样本的前缀和后缀
            prefix_slice = slice(0, history_start_embed)
            suffix_slice = slice(history_start_embed + 1, None)  # +1 是为了跳过<history>标记
            
            batch_prefix = inputs_embeds[:, prefix_slice]
            batch_suffix = inputs_embeds[:, suffix_slice]
            
            # 获取历史文本嵌入
            history_ids = history_encoding["input_ids"]
            
            # 获取最大历史长度用于创建输出张量
            max_history_len = history_ids.size(1)
            
            # 为历史文本创建批量嵌入
            batch_history_embeds = self.model.get_input_embeddings()(history_ids)
            
            # 计算新序列的总长度
            prefix_len = batch_prefix.size(1)
            suffix_len = batch_suffix.size(1)
            total_len = prefix_len + max_history_len + suffix_len
            
            # 创建新的输出张量
            new_inputs_embeds = torch.zeros(batch_size, total_len, inputs_embeds.size(2), device=device)
            new_attention_mask = torch.ones(batch_size, total_len, dtype=torch.long, device=device)
            
            # 填充各部分
            new_inputs_embeds[:, :prefix_len] = batch_prefix
            new_inputs_embeds[:, prefix_len:prefix_len+max_history_len] = batch_history_embeds
            new_inputs_embeds[:, prefix_len+max_history_len:] = batch_suffix
            
            # 更新
            inputs_embeds = new_inputs_embeds
            attention_mask = new_attention_mask
        
        # 使用Hugging Face的生成功能
        generation_config = transformers.GenerationConfig(
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # 使用model_kwargs传递inputs_embeds和attention_mask
        with torch.no_grad():
            generation_output = self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        # 获取生成的token序列
        generated_ids = generation_output.sequences
        
        # 解码生成的文本
        decoded_outputs = []
        findings_token_id = self.tokenizer.convert_tokens_to_ids("<findings>")
        
        for i, output in enumerate(generated_ids):
            # 找到<findings>标记的位置
            findings_start_pos = (output == findings_token_id).nonzero()
            if findings_start_pos.numel() > 0:
                findings_start = findings_start_pos[-1].item() + 1
                # 只保留<findings>之后的部分
                findings_text = self.tokenizer.decode(output[findings_start:], skip_special_tokens=True)
                decoded_outputs.append(findings_text)
            else:
                # 如果没有<findings>标记，返回原始的生成结果（跳过prompt的部分）
                original_prompt_len = prompt_input_ids[i].size(0)
                decoded_text = self.tokenizer.decode(output[original_prompt_len:], skip_special_tokens=True)
                decoded_outputs.append(decoded_text)
        
        return decoded_outputs