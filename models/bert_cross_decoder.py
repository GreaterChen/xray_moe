import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from models.med import BertConfig, BertModel, BertLMHeadModel


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
        
        # 从配置中获取是否启用RGAT
        self.enable_rgat = getattr(config, 'ENABLE_RGAT', True)
        
        # 从配置中获取是否使用历史文本
        self.use_history = getattr(config, 'USE_HISTORY', False)
        
        # 使用传入的tokenizer或创建一个新的
        self.tokenizer = tokenizer
        
        # 加载BERT配置，启用交叉注意力
        bert_config_path = os.path.join(config.ROOT_DIR if hasattr(config, 'ROOT_DIR') else '.', "configs/bert_config.json")
        if os.path.exists(bert_config_path):
            decoder_config = BertConfig.from_json_file(bert_config_path)
        else:
            decoder_config = BertConfig.from_pretrained("bert-base-uncased")
            
        # 配置交叉注意力参数
        decoder_config.encoder_width = hidden_dim
        decoder_config.add_cross_attention = True
        decoder_config.is_decoder = True

        # 初始化解码器（BertLMHeadModel原生支持交叉注意力）
        self.text_decoder = BertLMHeadModel.from_pretrained(
            "bert-base-uncased", config=decoder_config, local_files_only=True
        )

        # 调整词表大小
        self.text_decoder.resize_token_embeddings(len(self.tokenizer))
        
        # 视觉特征映射层（如果需要额外的模态适配）
        # 可以考虑添加一个开关来决定是否使用
        self.use_visual_projection = getattr(config, 'USE_VISUAL_PROJECTION', False)
        if self.use_visual_projection:
            self.visual_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # 只有在启用RGAT时才创建疾病特征映射层
        if self.enable_rgat:
            self.disease_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # 设置视觉和疾病特征的token数量
        self.num_visual_tokens = 30
        self.num_disease_tokens = 14

    def forward(self, visual_features, history, target_text=None, mode="train", generation_params=None):
        """
        前向传播
        
        Args:
            visual_features: 视觉特征 [batch_size, num_tokens, hidden_dim]
                           如果ENABLE_RGAT=True: [B, 44, 768] 包含视觉特征和疾病特征
                           如果ENABLE_RGAT=False: [B, 30, 768] 仅包含视觉特征
            history: 历史文本编码
            target_text: 目标生成文本编码
            mode: "train" 或 "generate"
            generation_params: 生成参数字典
        """
        batch_size = visual_features.shape[0]
        device = visual_features.device
        
        # 处理视觉特征
        if self.enable_rgat:
            # RGAT模式：分离视觉特征和疾病特征
            visual_part = visual_features[:, :self.num_visual_tokens, :]
            disease_part = visual_features[:, self.num_visual_tokens:self.num_visual_tokens+self.num_disease_tokens, :]
            
            # 映射特征
            if self.use_visual_projection:
                visual_part = self.visual_projection(visual_part)
            projected_disease = self.disease_projection(disease_part)
            
            # 拼接特征
            projected_features = torch.cat([visual_part, projected_disease], dim=1)
        else:
            # 非RGAT模式
            if self.use_visual_projection:
                projected_features = self.visual_projection(visual_features)
            else:
                projected_features = visual_features
        
        # 创建特征的attention mask
        visual_attention_mask = torch.ones(
            projected_features.size()[:-1], dtype=torch.long, device=device
        )
        
        # 处理历史文本
        if not self.use_history or history is None:
            # 不使用历史文本
            history_input_ids = torch.empty((batch_size, 0), dtype=torch.long, device=device)
            history_attention_mask = torch.empty((batch_size, 0), dtype=torch.long, device=device)
        else:
            history_input_ids = history.input_ids.to(device)
            history_attention_mask = history.attention_mask.to(device)
            
            # 移除padding，获取实际的history序列
            unpadded_inputs = []
            unpadded_masks = []
            actual_lengths = history_attention_mask.sum(dim=1)
            for i in range(batch_size):
                length = actual_lengths[i].item()
                if length > 0:
                    unpadded_inputs.append(history_input_ids[i, :length])
                    unpadded_masks.append(history_attention_mask[i, :length])
                else:
                    unpadded_inputs.append(torch.empty((0,), dtype=torch.long, device=device))
                    unpadded_masks.append(torch.empty((0,), dtype=torch.long, device=device))
            
            # 重新对齐
            history_input_ids = pad_sequence(unpadded_inputs, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            history_attention_mask = pad_sequence(unpadded_masks, batch_first=True, padding_value=0)

        if mode == "train" and target_text is not None:
            # 处理目标文本
            target_input_ids = target_text.input_ids.to(device)
            target_attention_mask = target_text.attention_mask.to(device)
            
            if self.use_history:
                actual_history_lengths = history_attention_mask.sum(dim=1)
                actual_target_lengths = target_attention_mask.sum(dim=1)
                
                batch_size = history_input_ids.shape[0]
                max_history_len = actual_history_lengths.max().item()
                # target去掉[CLS]的长度（保留完整序列用于模型自动shift）
                max_target_len = torch.clamp(actual_target_lengths - 1, min=0).max().item()  # 只去掉[CLS]
                max_total_len = max_history_len + max_target_len
                
                # 初始化张量
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
                
                # 逐样本处理
                for i in range(batch_size):
                    h_len = actual_history_lengths[i].item()
                    t_len = actual_target_lengths[i].item()
                    
                    # History部分：保留完整history（已包含[CLS] ... [SEP]）
                    full_input_ids[i, :h_len] = history_input_ids[i, :h_len]
                    full_attention_mask[i, :h_len] = 1
                    
                    # Target部分：去掉[CLS]，保留完整序列（t1 ... tm [EOS]）
                    if t_len <= 1:  # 只有[CLS]或空
                        continue
                    
                    # 跳过target的[CLS]（索引0），从t1开始到[EOS]
                    target_tokens = t_len - 1  # 去掉[CLS]
                    if target_tokens <= 0:
                        continue
                    
                    start = h_len
                    end = h_len + target_tokens
                    
                    full_input_ids[i, start:end] = target_input_ids[i, 1:1+target_tokens]
                    full_attention_mask[i, start:end] = 1
                    labels[i, start:end] = target_input_ids[i, 1:1+target_tokens]
                    
            else:
                full_input_ids = target_input_ids
                full_attention_mask = target_attention_mask
                
                # 标签也使用完整序列，但将padding位置设为-100
                labels = target_input_ids.clone()
                labels[target_attention_mask == 0] = -100
            
            # 模型前向传播
            outputs = self.text_decoder(
                input_ids=full_input_ids,
                attention_mask=full_attention_mask,
                encoder_hidden_states=projected_features,
                encoder_attention_mask=visual_attention_mask,
                labels=labels,
                return_dict=True,
            )

            return outputs.loss
            
        else:  # mode == "generate"
            params = {
                "history_input_ids": history_input_ids,
                "history_attention_mask": history_attention_mask,
                "visual_features": projected_features,
                "visual_attention_mask": visual_attention_mask,
            }
            
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
        max_new_tokens=150,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        repetition_penalty=1.0,
    ):
        """
        根据历史编码和视觉特征生成文本
        
        Args:
            history_input_ids: 历史文本的input_ids (使用history时需要)
            history_attention_mask: 历史文本的attention_mask (使用history时需要)
            visual_features: 视觉特征 [batch_size, num_tokens, hidden_dim]
            visual_attention_mask: 视觉特征的attention_mask
            num_beams: beam search的宽度
            max_new_tokens: 生成的最大新token数
            do_sample: 是否采样生成
            top_p: nucleus sampling的概率阈值
            temperature: 温度参数
            repetition_penalty: 重复惩罚系数
            
        Returns:
            generated_texts: 生成的文本列表
        """
        batch_size = visual_features.shape[0]
        device = visual_features.device
        
        # ============================================
        # 1. 准备生成的输入序列
        # ============================================
        if self.use_history and history_input_ids is not None and history_attention_mask is not None:
            # 使用history模式：history作为prompt
            # 注意：由于使用left padding，实际的history内容在右侧
            input_ids = history_input_ids
            attention_mask = history_attention_mask
            
            # 计算每个样本的实际长度（用于后续去除prompt）
            # left padding意味着padding在左侧，实际内容在右侧
            input_lengths = attention_mask.sum(dim=1)  # [batch_size]
            
        else:
            # 不使用history模式：只用[CLS]作为起始
            # 批量创建[CLS] token，shape: [batch_size, 1]
            input_ids = torch.full(
                (batch_size, 1), 
                self.tokenizer.cls_token_id, 
                dtype=torch.long, 
                device=device
            )
            attention_mask = torch.ones_like(input_ids)
            input_lengths = torch.ones(batch_size, dtype=torch.long, device=device)  # 都是1
        
        # ============================================
        # 2. 配置生成参数
        # ============================================
        generation_config = {
            # 基础参数
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            
            # 交叉注意力参数（用于关注视觉特征）
            "encoder_hidden_states": visual_features,
            "encoder_attention_mask": visual_attention_mask,
            
            # 生成策略参数
            "num_beams": num_beams,
            "repetition_penalty": repetition_penalty,
        }
        
        # 配置采样策略：beam search和sampling互斥
        if num_beams > 1:
            # Beam search模式：确定性生成
            generation_config["do_sample"] = False
            generation_config["early_stopping"] = True
        else:
            # Greedy或sampling模式
            generation_config["do_sample"] = do_sample
            if do_sample:
                generation_config["top_p"] = top_p
                generation_config["temperature"] = temperature
        
        # ============================================
        # 3. 执行生成
        # ============================================
        with torch.no_grad():
            generated_ids = self.text_decoder.generate(**generation_config)
        
        # ============================================
        # 4. 解码生成的文本（去除prompt部分）
        # ============================================
        generated_texts = []
        
        if self.use_history and history_input_ids is not None:
            # 使用history模式：需要去除history prompt部分
            for i in range(batch_size):
                # 获取该样本的实际输入长度
                prompt_len = input_lengths[i].item()
                # 只保留生成的新内容
                new_tokens = generated_ids[i, prompt_len:]
                # 解码为文本
                text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                generated_texts.append(text)
        else:
            # 不使用history模式：只需去除起始的[CLS]
            for i, output in enumerate(batch_size):
                text = self.tokenizer.decode(output, skip_special_tokens=True)
                generated_texts.append(text)
        
        return generated_texts