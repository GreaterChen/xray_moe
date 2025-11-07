import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertTokenizer, BertLMHeadModel


class BertLMHeadModelWithCrossAttention(BertLMHeadModel):
    """
    继承自BertLMHeadModel，添加对encoder_hidden_states和encoder_attention_mask的正确处理
    """
    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        """
        重写以正确处理encoder_hidden_states和encoder_attention_mask在beam search中的扩展
        """
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        # 添加dummy token
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")

        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        # 处理encoder_hidden_states和encoder_attention_mask
        # 这些在第一次调用generate时会被传入，之后会保存在model_kwargs中
        inputs = {
            "input_ids": input_ids, 
            "attention_mask": attention_mask,
        }
        
        # 保留encoder相关的参数
        if "encoder_hidden_states" in model_kwargs:
            inputs["encoder_hidden_states"] = model_kwargs["encoder_hidden_states"]
        if "encoder_attention_mask" in model_kwargs:
            inputs["encoder_attention_mask"] = model_kwargs["encoder_attention_mask"]
            
        return inputs
    
    @staticmethod
    def _expand_inputs_for_generation(
        expand_size=1,
        is_encoder_decoder=False,
        input_ids=None,
        **model_kwargs,
    ):
        """
        重写以正确扩展encoder_hidden_states和encoder_attention_mask用于beam search
        使用repeat_interleave而不是index_select，遵循transformers标准实现
        """
        # 如果expand_size为1，不需要扩展
        if expand_size == 1:
            return input_ids, model_kwargs

        def _expand_dict_for_generation(dict_to_expand):
            """扩展字典中的所有张量"""
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        # 扩展input_ids
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        # 扩展model_kwargs中的所有张量（包括encoder_hidden_states和encoder_attention_mask）
        model_kwargs = _expand_dict_for_generation(model_kwargs)

        return input_ids, model_kwargs


class BertCrossDecoder(nn.Module):
    """
    BERT交叉注意力解码器模型，使用视觉特征作为KV源，历史文本作为Q的开头
    """
    def __init__(
        self,
        config,
        tokenizer=None,
        hidden_dim=768,
        max_length=196,
    ):
        super().__init__()

        self.config = config
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        
        # 从配置中获取是否启用RAGT
        self.enable_ragt = getattr(config, 'ENABLE_RGAT', True)
        
        # 使用传入的tokenizer或创建一个新的
        if tokenizer:
            self.tokenizer = tokenizer
            # 确保padding_side为right，与BERT预训练一致
            self.tokenizer.padding_side = 'right'
        else:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
            self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})
            # 设置padding_side为right，与BERT预训练一致
            self.tokenizer.padding_side = 'right'
        
        # 加载BERT配置，启用交叉注意力
        bert_config_path = os.path.join(config.ROOT_DIR if hasattr(config, 'ROOT_DIR') else '.', "configs/bert_config.json")
        if os.path.exists(bert_config_path):
            decoder_config = BertConfig.from_json_file(bert_config_path)
        else:
            # 创建默认配置
            decoder_config = BertConfig.from_pretrained("bert-base-uncased")
            
        # 配置交叉注意力参数
        decoder_config.encoder_width = hidden_dim
        decoder_config.add_cross_attention = True
        decoder_config.is_decoder = True

        # 初始化解码器，使用自定义的BertLMHeadModelWithCrossAttention类
        # 这个类正确处理了encoder_hidden_states和encoder_attention_mask在beam search中的扩展
        self.text_decoder = BertLMHeadModelWithCrossAttention.from_pretrained(
            "bert-base-uncased", config=decoder_config, local_files_only=True
        )

        # 调整词表大小
        self.text_decoder.resize_token_embeddings(len(self.tokenizer))
        
        # 添加特征映射层，确保视觉特征和疾病特征维度匹配
        # visual_features: [B, 30, 768] (1个CLS + 29个区域)
        # disease_features: [B, 14, 768] (仅在ENABLE_RGAT=True时使用)
        self.visual_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # 只有在启用RAGT时才创建疾病特征映射层
        if self.enable_ragt:
            self.disease_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # 设置视觉和疾病特征的token数量
        self.num_visual_tokens = 30
        self.num_disease_tokens = 14

    def forward(self, visual_features, history, target_text=None, mode="train", generation_params=None, use_history=False):
        """
        前向传播
        
        Args:
            visual_features: 视觉特征 [batch_size, num_tokens, hidden_dim]
                           如果ENABLE_RGAT=True: [B, 44, 768] 包含视觉特征(前30个token)和疾病特征(后14个token)
                           如果ENABLE_RGAT=False: [B, 30, 768] 仅包含视觉特征
            history: 历史文本编码 {input_ids, attention_mask} 或原始文本列表
            target_text: 目标生成文本编码 {input_ids, attention_mask} 或原始文本列表
            mode: 训练模式 "train" 或 "generate"
            generation_params: 生成参数字典，用于mode="generate"
            use_history: 是否使用历史文本作为prompt，如为False则仅使用视觉特征
            
        Returns:
            如果mode="train"：返回 logits, hidden_states, decoded_texts, loss_lm
            如果mode="generate"：返回生成的文本列表
        """
        batch_size = visual_features.shape[0]
        device = visual_features.device
        
        # 根据是否启用RAGT来处理输入特征
        if self.enable_ragt:
            # RAGT模式：输入包含视觉特征和疾病特征
            # visual_features: [B, 44, 768]，前30个token是视觉特征，后14个token是疾病特征
            visual_part = visual_features[:, :self.num_visual_tokens, :]  # [B, 30, 768]
            disease_part = visual_features[:, self.num_visual_tokens:self.num_visual_tokens+self.num_disease_tokens, :]  # [B, 14, 768]
            
            # 分别对视觉特征和疾病特征进行映射
            projected_visual = self.visual_projection(visual_part)  # [B, 30, 768]
            projected_disease = self.disease_projection(disease_part)  # [B, 14, 768]
            
            # 拼接映射后的特征作为encoder_hidden_states
            projected_features = torch.cat([projected_visual, projected_disease], dim=1)  # [B, 44, 768]
        else:
            # 非RAGT模式：输入仅包含视觉特征
            # visual_features: [B, 30, 768]
            projected_features = self.visual_projection(visual_features)  # [B, 30, 768]
        
        # 创建特征的attention mask
        visual_attention_mask = torch.ones(
            projected_features.size()[:-1], dtype=torch.long, device=device
        )
        
        # 处理历史文本
        if not use_history or history is None:
            # 如果不使用历史文本或历史文本为空，创建一个只包含起始token的序列
            history_input_ids = torch.full(
                (batch_size, 1),
                self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.cls_token_id,
                dtype=torch.long,
                device=device
            )
            history_attention_mask = torch.ones_like(history_input_ids)
        else:
            history_input_ids = history.input_ids.to(device)
            # 如果history_attention_mask全为0，则设置为全1
            history_attention_mask = history.attention_mask.to(device)
            if history_attention_mask.sum() == 0:
                history_attention_mask = torch.ones_like(history_input_ids)

        if mode == "train" and target_text is not None:
            # 处理目标文本
            target_input_ids = target_text.input_ids.to(device)
            target_attention_mask = target_text.attention_mask.to(device)
            
            if use_history:
                # ============================================
                # 使用历史文本作为prompt的自回归训练
                # ============================================
                # 目标：实现"预测下一个token"的自回归范式
                # 
                # 训练格式（history保持完整句子，target做自回归shift）：
                #   输入 (input_ids): [CLS] h1 ... h_n [SEP] | [CLS] t1 ... t_m
                #   标签 (labels):    [-100]........[-100] | t1 ... t_m [SEP]
                # 
                # 其中：
                #   - history: [CLS] h1 ... h_n [SEP] [PAD]... (原始tokenization)
                #   - target:  [CLS] t1 ... t_m [SEP] [PAD]... (原始tokenization)
                #   - 拼接时保留history的[SEP]，仅对target做去CLS+右移
                #   - 自回归shift：输入的第i个位置预测第i+1个位置的token
                # ============================================
                
                # 获取每个样本的实际history长度（去除padding）
                actual_history_lengths = history_attention_mask.sum(dim=1)  # [batch_size]
                
                # 获取每个样本的实际target长度（去除padding）
                # target格式: [CLS] t1 t2 ... tm [SEP] [PAD] ...
                actual_target_lengths = target_attention_mask.sum(dim=1)  # [batch_size]
                
                batch_size = history_input_ids.shape[0]
                # 计算最大总长度：history（保留末尾SEP）+ target（去掉CLS与最后一个token用于shift）
                max_history_len = actual_history_lengths.max().item()
                max_target_input_len = torch.clamp(actual_target_lengths - 1, min=0).max().item()
                max_total_len = max_history_len + max_target_input_len
                
                # 初始化拼接后的张量
                full_input_ids = torch.full(
                    (batch_size, max_total_len), 
                    self.tokenizer.pad_token_id, 
                    dtype=torch.long, 
                    device=device
                )
                full_attention_mask = torch.zeros(
                    (batch_size, max_total_len), 
                    dtype=torch.long, 
                    device=device
                )
                labels = torch.full(
                    (batch_size, max_total_len), 
                    -100, 
                    dtype=torch.long, 
                    device=device
                )
                
                # 逐样本处理以正确对齐
                for i in range(batch_size):
                    h_len = actual_history_lengths[i].item()
                    t_len = actual_target_lengths[i].item()
                    
                    # History部分：保留完整history（含末尾[SEP]）
                    full_input_ids[i, :h_len] = history_input_ids[i, :h_len]
                    full_attention_mask[i, :h_len] = 1
                    # labels的history部分全部设为-100（不计算损失）
                    
                    # Target部分：自回归shift（去掉target结尾token，保留开头[CLS]以驱动首个预测）
                    target_input_len = max(t_len - 1, 0)
                    if target_input_len <= 0:
                        continue
                    
                    # 输入部分：[CLS] t1 ... t_m（不含最终[SEP]）
                    start = h_len
                    end = h_len + target_input_len
                    full_input_ids[i, start:end] = target_input_ids[i, :target_input_len]
                    full_attention_mask[i, start:end] = 1
                    
                    # 标签部分：t1 ... t_m [SEP]
                    labels[i, start:end] = target_input_ids[i, 1:1+target_input_len]
            
            else:
                # ============================================
                # 不使用历史文本，仅使用视觉特征的自回归训练
                # ============================================
                # 训练格式：
                #   输入 (input_ids): [CLS] t1 t2 ... t_m
                #   标签 (labels):    t1 t2 ... t_m [SEP]
                # 
                # 自回归shift：输入的第i个位置预测第i+1个位置的token
                # ============================================
                
                # 获取每个样本的实际长度
                actual_lengths = target_attention_mask.sum(dim=1)  # [batch_size]
                max_input_len = torch.clamp(actual_lengths - 1, min=0).max().item()
                
                # 初始化输入和标签（去掉最后一个token用于shift）
                full_input_ids = torch.full(
                    (batch_size, max_input_len), 
                    self.tokenizer.pad_token_id, 
                    dtype=torch.long, 
                    device=device
                )
                full_attention_mask = torch.zeros(
                    (batch_size, max_input_len), 
                    dtype=torch.long, 
                    device=device
                )
                labels = torch.full(
                    (batch_size, max_input_len), 
                    -100, 
                    dtype=torch.long, 
                    device=device
                )
                
                # 逐样本处理自回归shift
                for i in range(batch_size):
                    actual_len = actual_lengths[i].item()
                    if actual_len <= 1:
                        continue
                    
                    input_len = actual_len - 1
                    # 输入：[CLS] t1 t2 ... t_m（去掉最后一个token，通常是[SEP]）
                    full_input_ids[i, :input_len] = target_input_ids[i, :input_len]
                    full_attention_mask[i, :input_len] = 1
                    
                    # 标签：t1 t2 ... t_m [SEP]（向前shift一位）
                    labels[i, :input_len] = target_input_ids[i, 1:1+input_len]
            
            # 模型前向传播
            outputs = self.text_decoder(
                input_ids=full_input_ids,
                attention_mask=full_attention_mask,
                encoder_hidden_states=projected_features,  # 视觉特征作为cross-attention的KV源
                encoder_attention_mask=visual_attention_mask,
                labels=labels,  # 根据use_history设置不同的标签
                output_hidden_states=True,
                return_dict=True,
            )
            
            # 获取logits和隐藏状态
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]
            hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_dim]
            
            # 训练过程中不再解码文本，减少CPU内存与字符串开销
            decoded_texts = None
                
            # 获取损失
            loss_lm = outputs.loss  # 这个损失已经只计算了标签不为-100的位置
            
            return logits, hidden_states, decoded_texts, loss_lm
            
        else:  # mode == "generate"
            # 准备生成参数
            # history_input_ids和history_attention_mask已经根据use_history处理好了
            params = {
                "history_input_ids": history_input_ids,
                "history_attention_mask": history_attention_mask,
                "visual_features": projected_features,
                "visual_attention_mask": visual_attention_mask,
            }
            
            # 如果提供了生成参数，将它们添加到参数字典中
            if generation_params:
                params.update(generation_params)
                
            return self.generate(**params)
            
    def generate(
        self,
        history_input_ids,
        history_attention_mask,
        visual_features,
        visual_attention_mask,
        num_beams=3,
        max_new_tokens=100,
        min_length=None,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        repetition_penalty=1.0,
    ):
        """
        根据历史编码和视觉特征生成文本

        Args:
            history_input_ids: 历史文本的input_ids
            history_attention_mask: 历史文本的attention_mask
            visual_features: 视觉特征
            visual_attention_mask: 视觉特征的attention_mask
            num_beams: beam search的宽度
            max_new_tokens: 生成的最大新token数量
            min_length: 生成的最小长度
            do_sample: 是否采样生成
            top_p: 采样的概率阈值
            temperature: 采样的温度
            repetition_penalty: 重复惩罚系数

        Returns:
            generated_texts: 生成的文本列表
        """
        # 移除history的padding，与训练时保持一致
        # 获取每个样本的实际history长度
        actual_history_lengths = history_attention_mask.sum(dim=1)  # [batch_size]
        max_history_len = actual_history_lengths.max().item()
        
        # 截断到实际最大长度，移除无用的padding
        input_ids = history_input_ids[:, :max_history_len]
        attention_mask = history_attention_mask[:, :max_history_len]
        
        # 准备交叉注意力参数
        model_kwargs = {
            "encoder_hidden_states": visual_features,
            "encoder_attention_mask": visual_attention_mask,
        }
        
        # 配置生成参数
        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "eos_token_id": self.tokenizer.sep_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "repetition_penalty": repetition_penalty,
        }
        
        # 当使用beam search (num_beams > 1)时，不能同时使用sampling
        if num_beams > 1:
            generation_kwargs["do_sample"] = False
        else:
            generation_kwargs["do_sample"] = do_sample
            if do_sample:
                generation_kwargs["top_p"] = top_p
                generation_kwargs["temperature"] = temperature
        
        # 添加最小长度约束
        if min_length is not None:
            generation_kwargs["min_length"] = min_length
        
        # 添加交叉注意力参数
        generation_kwargs.update(model_kwargs)
        
        # 生成文本
        outputs = self.text_decoder.generate(**generation_kwargs)

        # 解码生成的文本，去除统一的输入前缀（与传入generate的input_ids长度一致），只保留新生成的内容
        generated_texts = []
        input_len = input_ids.shape[1]
        for _, tokens in enumerate(outputs):
            # 只解码输入前缀之后生成的内容
            generated_part = tokens[input_len:]
            # 解码生成的部分
            text = self.tokenizer.decode(generated_part, skip_special_tokens=True)
            generated_texts.append(text)
            
        return generated_texts 
