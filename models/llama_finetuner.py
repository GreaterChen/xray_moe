import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import LlamaForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
import gc
from utils import analyze_gpu_memory

class LlamaFinetuner(nn.Module):
    def __init__(
        self,
        config,
        base_model="meta-llama/Meta-Llama-3.2-3B",
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        visual_dim=768,
        text_dim=768,
        load_in_4bit=True,
    ):
        super(LlamaFinetuner, self).__init__()
        
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
        
        # 加载基础Llama模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = LlamaForCausalLM.from_pretrained(
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
        history_encoding,
        findings,
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
        
        # 构建完整的提示文本结构
        prompts = ["<image></image><history></history><findings>" for _ in range(batch_size)]
        
        prompt_encoding = self.tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        prompt_input_ids = prompt_encoding.input_ids
        
        # 获取所有特殊标记的ID
        image_start_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
        image_end_token_id = self.tokenizer.convert_tokens_to_ids("</image>")
        history_start_token_id = self.tokenizer.convert_tokens_to_ids("<history>")
        history_end_token_id = self.tokenizer.convert_tokens_to_ids("</history>")
        findings_token_id = self.tokenizer.convert_tokens_to_ids("<findings>")
        
        # 确保所有标记都能被正确识别
        for name, token_id in {
            "<image>": image_start_token_id,
            "</image>": image_end_token_id,
            "<history>": history_start_token_id,
            "</history>": history_end_token_id,
            "<findings>": findings_token_id
        }.items():
            if token_id == self.tokenizer.unk_token_id:
                print(f"警告: 特殊标记 {name} 被识别为未知token")
        
        # 找到所有特殊标记的位置
        image_start_idx = torch.where(prompt_input_ids[0] == image_start_token_id)[0][0].item()
        image_end_idx = torch.where(prompt_input_ids[0] == image_end_token_id)[0][0].item()
        history_start_idx = torch.where(prompt_input_ids[0] == history_start_token_id)[0][0].item()
        history_end_idx = torch.where(prompt_input_ids[0] == history_end_token_id)[0][0].item()
        findings_idx = torch.where(prompt_input_ids[0] == findings_token_id)[0][0].item()
        
        # 一次性构建完整的嵌入序列
        # 1. <s>到<image>部分
        prefix_ids = prompt_input_ids[0, :image_start_idx+1]
        prefix_embeds = self.model.get_input_embeddings()(prefix_ids)
        batch_prefix_embeds = prefix_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 2. 视觉特征部分 (已经映射到模型维度)
        
        # 3. </image>到<history>部分
        middle1_ids = prompt_input_ids[0, image_end_idx:history_start_idx+1]
        middle1_embeds = self.model.get_input_embeddings()(middle1_ids)
        batch_middle1_embeds = middle1_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 4. 历史文本部分
        history_ids = history_encoding["input_ids"]
        history_embeds = self.model.get_input_embeddings()(history_ids)
        history_len = history_embeds.size(1)
        
        # 5. </history>到<findings>部分
        middle2_ids = prompt_input_ids[0, history_end_idx:findings_idx+1]
        middle2_embeds = self.model.get_input_embeddings()(middle2_ids)
        batch_middle2_embeds = middle2_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        middle2_len = batch_middle2_embeds.size(1)
        
        # 计算各部分长度
        prefix_len = batch_prefix_embeds.size(1)
        visual_len = projected_visual.size(1)
        middle1_len = batch_middle1_embeds.size(1)
        
        # 计算总长度并创建输出张量（不包含findings部分）
        prompt_len = prefix_len + visual_len + middle1_len + history_len + middle2_len
        
        # 初始化当前位置计数器
        current_pos = 0
        
        # findings部分处理
        findings_input_ids = findings["input_ids"]
        findings_attention_mask = findings["attention_mask"]
        
        # 获取findings的嵌入部分
        findings_embeds = self.model.get_input_embeddings()(findings_input_ids[:, 1:])  # 跳过BOS
        findings_len = findings_embeds.size(1)
        
        # 计算完整序列的总长度
        total_len = prompt_len + findings_len
        
        # 创建输出张量
        inputs_embeds = torch.zeros(batch_size, total_len, projected_visual.size(2), device=device)
        attention_mask = torch.zeros(batch_size, total_len, dtype=torch.long, device=device)
        
        # 创建标签张量（只对findings部分计算损失）
        labels = torch.full((batch_size, total_len), -100, device=device)
        
        # 填充各部分
        # 1. 前缀部分：<s>到<image>
        inputs_embeds[:, current_pos:current_pos+prefix_len] = batch_prefix_embeds
        attention_mask[:, current_pos:current_pos+prefix_len] = 1
        current_pos += prefix_len
        
        # 2. 视觉特征部分
        inputs_embeds[:, current_pos:current_pos+visual_len] = projected_visual
        attention_mask[:, current_pos:current_pos+visual_len] = 1
        current_pos += visual_len
        
        # 3. 中间部分1：</image>到<history>
        inputs_embeds[:, current_pos:current_pos+middle1_len] = batch_middle1_embeds
        attention_mask[:, current_pos:current_pos+middle1_len] = 1
        current_pos += middle1_len
        
        # 4. 历史文本部分
        inputs_embeds[:, current_pos:current_pos+history_len] = history_embeds
        attention_mask[:, current_pos:current_pos+history_len] = 1
        current_pos += history_len
        
        # 5. 中间部分2：</history>到<findings>
        inputs_embeds[:, current_pos:current_pos+middle2_len] = batch_middle2_embeds
        attention_mask[:, current_pos:current_pos+middle2_len] = 1
        current_pos += middle2_len
        
        # 6. findings部分
        inputs_embeds[:, current_pos:current_pos+findings_len] = findings_embeds
        attention_mask[:, current_pos:current_pos+findings_len] = findings_attention_mask[:, 1:]  # 跳过BOS
        
        # 设置标签（只对findings部分计算损失）
        labels[:, current_pos:current_pos+findings_len] = findings_input_ids[:, 1:]  # 从第二个token开始（跳过BOS）
        
        # 运行模型
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        
        return outputs
    
    def generate(
        self,
        visual_features,
        history_encoding,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    ):
        """
        生成报告文本
        
        Args:
            visual_features: 视觉特征 [batch_size, num_tokens, visual_dim]
            history_encoding: 历史文本的编码 {input_ids, attention_mask}
            max_new_tokens: 生成文本的最大新token数量
            do_sample: 是否采样生成
            temperature: 温度参数
            top_p: 概率截断阈值
        """
        batch_size = visual_features.shape[0]
        device = visual_features.device
        
        # 将视觉特征映射到模型维度
        projected_visual = self.visual_projection(visual_features)
        
        # 构建完整的提示文本结构
        prompts = ["<image></image><history></history><findings>" for _ in range(batch_size)]
        
        prompt_encoding = self.tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        prompt_input_ids = prompt_encoding.input_ids
        
        # 获取所有特殊标记的ID
        image_start_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
        image_end_token_id = self.tokenizer.convert_tokens_to_ids("</image>")
        history_start_token_id = self.tokenizer.convert_tokens_to_ids("<history>")
        history_end_token_id = self.tokenizer.convert_tokens_to_ids("</history>")
        findings_token_id = self.tokenizer.convert_tokens_to_ids("<findings>")
        
        # 确保所有标记都能被正确识别
        for name, token_id in {
            "<image>": image_start_token_id,
            "</image>": image_end_token_id,
            "<history>": history_start_token_id,
            "</history>": history_end_token_id,
            "<findings>": findings_token_id
        }.items():
            if token_id == self.tokenizer.unk_token_id:
                print(f"警告: 特殊标记 {name} 被识别为未知token")
        
        # 找到所有特殊标记的位置
        image_start_idx = torch.where(prompt_input_ids[0] == image_start_token_id)[0][0].item()
        image_end_idx = torch.where(prompt_input_ids[0] == image_end_token_id)[0][0].item()
        history_start_idx = torch.where(prompt_input_ids[0] == history_start_token_id)[0][0].item()
        history_end_idx = torch.where(prompt_input_ids[0] == history_end_token_id)[0][0].item()
        findings_idx = torch.where(prompt_input_ids[0] == findings_token_id)[0][0].item()
        
        # 一次性构建完整的嵌入序列
        # 1. <s>到<image>部分
        prefix_ids = prompt_input_ids[0, :image_start_idx+1]
        prefix_embeds = self.model.get_input_embeddings()(prefix_ids)
        batch_prefix_embeds = prefix_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 2. 视觉特征部分
        
        # 3. </image>到<history>部分
        middle1_ids = prompt_input_ids[0, image_end_idx:history_start_idx+1]
        middle1_embeds = self.model.get_input_embeddings()(middle1_ids)
        batch_middle1_embeds = middle1_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 4. 历史文本部分
        history_ids = history_encoding["input_ids"]
        history_embeds = self.model.get_input_embeddings()(history_ids)
        
        # 5. </history>到<findings>部分
        middle2_ids = prompt_input_ids[0, history_end_idx:findings_idx+1]
        middle2_embeds = self.model.get_input_embeddings()(middle2_ids)
        batch_middle2_embeds = middle2_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 计算各部分长度
        prefix_len = batch_prefix_embeds.size(1)
        visual_len = projected_visual.size(1)
        middle1_len = batch_middle1_embeds.size(1)
        history_len = history_embeds.size(1)
        middle2_len = batch_middle2_embeds.size(1)
        
        # 计算总长度并创建输出张量
        total_len = prefix_len + visual_len + middle1_len + history_len + middle2_len
        inputs_embeds = torch.zeros(batch_size, total_len, projected_visual.size(2), device=device)
        attention_mask = torch.ones(batch_size, total_len, dtype=torch.long, device=device)
        
        # 一次性填充所有部分
        current_pos = 0
        
        # 1. 前缀部分：<s>到<image>
        inputs_embeds[:, current_pos:current_pos+prefix_len] = batch_prefix_embeds
        current_pos += prefix_len
        
        # 2. 视觉特征部分
        inputs_embeds[:, current_pos:current_pos+visual_len] = projected_visual
        current_pos += visual_len
        
        # 3. 中间部分1：</image>到<history>
        inputs_embeds[:, current_pos:current_pos+middle1_len] = batch_middle1_embeds
        current_pos += middle1_len
        
        # 4. 历史文本部分
        inputs_embeds[:, current_pos:current_pos+history_len] = history_embeds
        current_pos += history_len
        
        # 5. 中间部分2：</history>到<findings>
        inputs_embeds[:, current_pos:current_pos+middle2_len] = batch_middle2_embeds
        
        # 记录输入长度，用于后续提取生成部分
        input_length = total_len
        
        # 使用Hugging Face的生成功能
        generation_config = transformers.GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # 生成文本
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
        
        # 直接使用batch_decode解码生成部分，跳过输入部分
        # 注意：input_length是输入序列的长度，对应模型内部转换的token IDs长度
        # 我们只解码input_length之后的部分，这就是新生成的内容
        decoded_outputs = self.tokenizer.batch_decode(
            generated_ids[:, input_length:], 
            skip_special_tokens=True  # 保留特殊标记以保持一致性
        )
        
        return decoded_outputs 