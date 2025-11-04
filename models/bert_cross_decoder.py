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
        
        # 添加特征映射层，确保视觉特征和文本特征维度匹配
        self.visual_projection = nn.Linear(hidden_dim, hidden_dim)
        self.text_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, visual_features, history, target_text=None, mode="train", generation_params=None, use_history=False):
        """
        前向传播
        
        Args:
            visual_features: 视觉特征 [batch_size, num_visual_tokens, hidden_dim]
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
        
        # 将视觉特征映射到decoder隐藏维度
        # projected_visual = self.visual_projection(visual_features)
        projected_visual = visual_features
        
        # 创建视觉特征的attention mask
        visual_attention_mask = torch.ones(
            visual_features.size()[:-1], dtype=torch.long, device=device
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
        elif hasattr(history, 'input_ids'):
            # 处理BatchEncoding或类字典类型
            history_input_ids = history.input_ids.to(device)
            history_attention_mask = history.attention_mask.to(device)
        elif isinstance(history, dict) and 'input_ids' in history:
            history_input_ids = history['input_ids'].to(device)
            history_attention_mask = history['attention_mask'].to(device)
        elif isinstance(history, list):
            # 编码文本列表
            history_encoding = self.tokenizer(
                history,
                max_length=100,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            ).to(device)
            history_input_ids = history_encoding.input_ids
            history_attention_mask = history_encoding.attention_mask
        else:
            raise ValueError(f"历史文本必须是BatchEncoding、编码字典或文本列表，当前类型: {type(history)}")
            
        if mode == "train" and target_text is not None:
            # 处理目标文本
            if hasattr(target_text, 'input_ids'):
                # 处理BatchEncoding或类字典类型
                target_input_ids = target_text.input_ids.to(device)
                target_attention_mask = target_text.attention_mask.to(device)
            elif isinstance(target_text, dict) and 'input_ids' in target_text:
                target_input_ids = target_text['input_ids'].to(device)
                target_attention_mask = target_text['attention_mask'].to(device)
            elif isinstance(target_text, list):
                # 编码目标文本
                target_encoding = self.tokenizer(
                    target_text,
                    max_length=196,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt',
                ).to(device)
                target_input_ids = target_encoding.input_ids
                target_attention_mask = target_encoding.attention_mask
            else:
                raise ValueError(f"目标文本必须是BatchEncoding、编码字典或文本列表，当前类型: {type(target_text)}")
            
            if use_history:
                # 获取每个样本的实际history长度（去除padding）
                # history_attention_mask中1的位置表示实际内容
                actual_history_lengths = history_attention_mask.sum(dim=1)  # [batch_size]
                
                # 获取每个样本的实际target长度（去除padding和CLS）
                # target格式: [CLS] t1 t2 ... tm [SEP] [PAD] ...
                # 我们需要: t1 t2 ... tm [SEP]（跳过CLS）
                actual_target_lengths = target_attention_mask.sum(dim=1) - 1  # -1是因为要跳过CLS
                
                # 为了批处理，我们需要统一序列长度
                # 方案：移除history的padding，保留SEP，然后拼接target（跳过CLS）
                
                batch_size = history_input_ids.shape[0]
                max_total_len = actual_history_lengths.max() + actual_target_lengths.max()
                
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
                    
                    # 拼接: [CLS] h1 h2 ... hn [SEP] | t1 t2 ... tm [SEP]
                    # History部分（保留SEP）
                    full_input_ids[i, :h_len] = history_input_ids[i, :h_len]
                    full_attention_mask[i, :h_len] = 1
                    # labels的history部分全部设为-100（不计算损失）
                    
                    # Target部分（跳过CLS，从第2个token开始）
                    full_input_ids[i, h_len:h_len+t_len] = target_input_ids[i, 1:1+t_len]
                    full_attention_mask[i, h_len:h_len+t_len] = 1
                    # labels的target部分设为对应的token id
                    labels[i, h_len:h_len+t_len] = target_input_ids[i, 1:1+t_len]
            
            else:
                # 不使用历史作为prompt，仅使用视觉特征
                # 直接使用目标文本作为输入和标签
                full_input_ids = target_input_ids
                full_attention_mask = target_attention_mask
                
                # 创建标签：目标文本的所有token都计算损失
                labels = target_input_ids.clone()
                # 设置[PAD]位置的标签为-100，使模型不计算这些位置的损失
                pad_positions = (full_input_ids == self.tokenizer.pad_token_id)
                labels[pad_positions] = -100
                
                # 将[CLS]标记的标签设为-100（不计算损失）
                labels[:, 0] = -100
            
            # 模型前向传播
            outputs = self.text_decoder(
                input_ids=full_input_ids,
                attention_mask=full_attention_mask,
                encoder_hidden_states=projected_visual,  # 视觉特征作为cross-attention的KV源
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
                "visual_features": projected_visual,
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

        # 解码生成的文本，去除历史文本部分，只保留新生成的内容
        generated_texts = []
        for i, tokens in enumerate(outputs):
            # 获取当前批次样本的实际历史长度
            # 由于我们已经移除了padding，直接使用实际长度
            actual_history_len = actual_history_lengths[i].item()
            
            # 只解码历史之后生成的内容
            generated_part = tokens[actual_history_len:]
            
            # 解码生成的部分
            text = self.tokenizer.decode(generated_part, skip_special_tokens=True)
            generated_texts.append(text)
            
        return generated_texts 