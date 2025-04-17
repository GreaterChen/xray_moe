import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertTokenizer
from models.med import BertLMHeadModel


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

        # 初始化解码器
        self.text_decoder = BertLMHeadModel.from_pretrained(
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
        if hasattr(history, 'input_ids'):
            # 处理BatchEncoding或类字典类型
            history_input_ids = history.input_ids
            history_attention_mask = history.attention_mask
        elif isinstance(history, dict) and 'input_ids' in history:
            history_input_ids = history['input_ids']
            history_attention_mask = history['attention_mask']
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
        
        # 如果不使用历史文本，创建一个只包含起始token的序列
        if not use_history:
            history_input_ids = torch.full(
                (batch_size, 1),
                self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.cls_token_id,
                dtype=torch.long,
                device=device
            )
            history_attention_mask = torch.ones_like(history_input_ids)
            
        if mode == "train" and target_text is not None:
            # 处理目标文本
            if hasattr(target_text, 'input_ids'):
                # 处理BatchEncoding或类字典类型
                target_input_ids = target_text.input_ids
                target_attention_mask = target_text.attention_mask
            elif isinstance(target_text, dict) and 'input_ids' in target_text:
                target_input_ids = target_text['input_ids']
                target_attention_mask = target_text['attention_mask']
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
                # 直接将历史最后一个token(SEP)替换为PAD
                history_input_ids[:, -1] = self.tokenizer.pad_token_id
                history_attention_mask[:, -1] = 0
                
                # 使用历史作为prompt：将历史文本和目标文本拼接
                # 跳过目标的第一个token(CLS)，确保拼接后只有一个有效的SEP标记
                full_input_ids = torch.cat([history_input_ids, target_input_ids[:, 1:]], dim=1)
                full_attention_mask = torch.cat([history_attention_mask, target_attention_mask[:, 1:]], dim=1)
                
                # 创建标签：历史部分设为-100(不计算损失)，目标部分保持原样
                labels = torch.full_like(full_input_ids, -100)
                # 目标文本位置的标签设置为目标token ID(从历史长度位置开始)
                history_len = history_input_ids.shape[1]
                labels[:, history_len:] = target_input_ids[:, 1:]  # 跳过目标文本的第一个token(CLS)
                
                # 设置[PAD]位置的标签为-100，使模型不计算这些位置的损失
                pad_positions = (full_input_ids == self.tokenizer.pad_token_id)
                labels[pad_positions] = -100
            
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
            
            # 解码预测的文本
            if use_history:
                # 只取历史文本之后的部分进行解码
                history_len = history_input_ids.shape[1]
                pred_tokens = torch.argmax(logits[:, history_len:, :], dim=-1)
            else:
                # 取全部内容解码
                pred_tokens = torch.argmax(logits, dim=-1)
                
            decoded_texts = []
            for tokens in pred_tokens:
                text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                decoded_texts.append(text)
                
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
            top_p: 采样的概率阈值
            temperature: 采样的温度
            repetition_penalty: 重复惩罚系数

        Returns:
            generated_texts: 生成的文本列表
        """
        # 准备生成所需的输入
        input_ids = history_input_ids
        attention_mask = history_attention_mask
        
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
            "do_sample": do_sample,
            "top_p": top_p
        }
        
        # 添加交叉注意力参数
        generation_kwargs.update(model_kwargs)
        
        # 生成文本
        outputs = self.text_decoder.generate(**generation_kwargs)
        
        # 解码生成的文本，去除历史文本部分，只保留新生成的内容
        generated_texts = []
        for i, tokens in enumerate(outputs):
            # 获取当前批次样本的实际历史长度
            actual_history_len = history_input_ids.size(1)
            if actual_history_len > 1:  # 如果使用了实际的历史文本
                # 由于现在使用右侧填充，history_attention_mask中的1表示实际内容
                actual_history_len = torch.sum(history_attention_mask[i]).item()
                # 只解码历史之后生成的内容
                generated_part = tokens[actual_history_len:]
            else:
                # 如果只有起始token，直接跳过第一个token
                generated_part = tokens[1:]
            
            # 解码生成的部分
            text = self.tokenizer.decode(generated_part, skip_special_tokens=True)
            generated_texts.append(text)
            
        return generated_texts 