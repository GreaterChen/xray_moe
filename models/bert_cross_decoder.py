import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
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
        """
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

        # 扩展model_kwargs中的所有张量
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
        
        # 从配置中获取是否启用RGAT
        self.enable_rgat = getattr(config, 'ENABLE_RGAT', True)
        
        # 从配置中获取是否使用历史文本
        self.use_history = getattr(config, 'USE_HISTORY', False)
        
        # 使用传入的tokenizer或创建一个新的
        if tokenizer:
            self.tokenizer = tokenizer
            self.tokenizer.padding_side = 'right'
        else:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
            self.tokenizer.padding_side = 'right'
        
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

        # 初始化解码器
        self.text_decoder = BertLMHeadModelWithCrossAttention.from_pretrained(
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
                # ============================================
                # 使用历史文本作为prompt的自回归训练（改进版）
                # ============================================
                # 新格式（去掉target的[CLS]，让拼接更自然）：
                #   输入: [CLS] h1 ... h_n [SEP] t1 t2 ... t_{m-1}
                #   标签: [-100]........[-100] t1 t2 ... t_m [SEP]
                # ============================================
                
                actual_history_lengths = history_attention_mask.sum(dim=1)
                actual_target_lengths = target_attention_mask.sum(dim=1)
                
                batch_size = history_input_ids.shape[0]
                max_history_len = actual_history_lengths.max().item()
                # target去掉[CLS]后的长度
                max_target_input_len = torch.clamp(actual_target_lengths - 2, min=0).max().item()  # -1 for CLS, -1 for last token shift
                max_total_len = max_history_len + max_target_input_len
                
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
                    
                    # History部分：保留完整history
                    full_input_ids[i, :h_len] = history_input_ids[i, :h_len]
                    full_attention_mask[i, :h_len] = 1
                    
                    # Target部分：去掉[CLS]，进行自回归shift
                    if t_len <= 1:  # 只有[CLS]或空
                        continue
                    
                    # 跳过target的[CLS]（索引0），从t1开始
                    # 输入: t1 t2 ... t_{m-1}（不包括最后的[SEP]）
                    target_tokens_to_use = t_len - 2  # 去掉[CLS]和最后一个token
                    if target_tokens_to_use <= 0:
                        continue
                    
                    start = h_len
                    end = h_len + target_tokens_to_use
                    
                    # 输入部分：从target的第2个token开始（跳过[CLS]）
                    full_input_ids[i, start:end] = target_input_ids[i, 1:1+target_tokens_to_use]
                    full_attention_mask[i, start:end] = 1
                    
                    # 标签部分：预测t1到t_m（包括[SEP]）
                    labels[i, start:end] = target_input_ids[i, 2:2+target_tokens_to_use]
            
            else:
                # ============================================
                # 不使用历史文本的自回归训练
                # ============================================
                # 保持原有逻辑
                actual_lengths = target_attention_mask.sum(dim=1)
                max_input_len = torch.clamp(actual_lengths - 1, min=0).max().item()
                
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
                
                for i in range(batch_size):
                    actual_len = actual_lengths[i].item()
                    if actual_len <= 1:
                        continue
                    
                    input_len = actual_len - 1
                    full_input_ids[i, :input_len] = target_input_ids[i, :input_len]
                    full_attention_mask[i, :input_len] = 1
                    labels[i, :input_len] = target_input_ids[i, 1:actual_len]
            
            # 模型前向传播
            outputs = self.text_decoder(
                input_ids=full_input_ids,
                attention_mask=full_attention_mask,
                encoder_hidden_states=projected_features,
                encoder_attention_mask=visual_attention_mask,
                labels=labels,
                output_hidden_states=True,
                return_dict=True,
            )
            
            logits = outputs.logits
            hidden_states = outputs.hidden_states[-1]
            decoded_texts = None
            loss_lm = outputs.loss
            
            return logits, hidden_states, decoded_texts, loss_lm
            
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
        max_new_tokens=100,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        repetition_penalty=1.0,
    ):
        """
        根据历史编码和视觉特征生成文本（改进版）
        """
        batch_size = history_input_ids.shape[0]
        device = history_input_ids.device
        
        # 准备解码器的输入（与训练时保持一致）
        per_sample_inputs = []
        prefix_lengths = []

        if self.use_history:
            history_lengths = history_attention_mask.sum(dim=1)
            for i in range(batch_size):
                length = history_lengths[i].item()
                if length > 0:
                    # 使用完整的history作为prompt: [CLS] h1 ... hn [SEP]
                    seq = history_input_ids[i, :length]
                    per_sample_inputs.append(seq)
                    prefix_lengths.append(seq.size(0))
                else:
                    # 如果history为空，使用[CLS]作为起始
                    prompt = history_input_ids.new_full((1,), self.tokenizer.cls_token_id)
                    per_sample_inputs.append(prompt)
                    prefix_lengths.append(1)
        else:
            # 不使用history时，只用[CLS]
            for i in range(batch_size):
                prompt = history_input_ids.new_full((1,), self.tokenizer.cls_token_id)
                per_sample_inputs.append(prompt)
                prefix_lengths.append(1)

        input_ids = pad_sequence(
            per_sample_inputs, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        prefix_lengths = torch.tensor(prefix_lengths, device=device)
        
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
        
        # beam search和sampling的互斥处理
        if num_beams > 1:
            generation_kwargs["do_sample"] = False
            generation_kwargs["early_stopping"] = True  # 添加early stopping
        else:
            generation_kwargs["do_sample"] = do_sample
            if do_sample:
                generation_kwargs["top_p"] = top_p
                generation_kwargs["temperature"] = temperature
        
        # 添加交叉注意力参数
        generation_kwargs.update(model_kwargs)
        
        # 生成文本
        with torch.no_grad():  # 确保生成时不计算梯度
            outputs = self.text_decoder.generate(**generation_kwargs)

        # 按实际前缀长度裁剪并解码
        generated_texts = []
        for idx, (tokens, prefix_len) in enumerate(zip(outputs, prefix_lengths.tolist())):
            # 只保留生成的新内容
            generated_part = tokens[prefix_len:]
            # 解码（跳过特殊token）
            text = self.tokenizer.decode(generated_part, skip_special_tokens=True)
            generated_texts.append(text)
            
        return generated_texts