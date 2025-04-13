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
            # 确保padding_side为left，这对解码器很重要
            self.tokenizer.padding_side = 'left'
        else:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
            self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})
            # 设置padding_side为left，这对解码器架构很重要
            self.tokenizer.padding_side = 'left'
        
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

    def forward(self, visual_features, history, target_text=None, mode="train", generation_params=None):
        """
        前向传播
        
        Args:
            visual_features: 视觉特征 [batch_size, num_visual_tokens, hidden_dim]
            history: 历史文本编码 {input_ids, attention_mask} 或原始文本列表
            target_text: 目标生成文本编码 {input_ids, attention_mask} 或原始文本列表
            mode: 训练模式 "train" 或 "generate"
            generation_params: 生成参数字典，用于mode="generate"
            
        Returns:
            如果mode="train"：返回 logits, hidden_states, decoded_texts, loss_lm
            如果mode="generate"：返回生成的文本列表
        """
        batch_size = visual_features.shape[0]
        device = visual_features.device
        
        # 将视觉特征映射到decoder隐藏维度
        projected_visual = self.visual_projection(visual_features)
        
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
                max_length=196,
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
            
            # 创建用于语言建模的完整序列：将历史文本和目标文本拼接
            full_input_ids = torch.cat([history_input_ids, target_input_ids[:, 1:]], dim=1)  # 跳过目标的第一个token(CLS)
            
            # 创建注意力掩码
            full_attention_mask = torch.cat([history_attention_mask, target_attention_mask[:, 1:]], dim=1)
            
            # 创建标签：历史部分设为-100(不计算损失)，目标部分保持原样
            labels = torch.full_like(full_input_ids, -100)
            # 目标文本位置的标签设置为目标token ID(从历史长度位置开始)
            history_len = history_input_ids.shape[1]
            labels[:, history_len:] = target_input_ids[:, 1:]  # 跳过目标文本的第一个token(CLS)
            
            # 模型前向传播
            outputs = self.text_decoder(
                input_ids=full_input_ids,
                attention_mask=full_attention_mask,
                encoder_hidden_states=projected_visual,  # 视觉特征作为cross-attention的KV源
                encoder_attention_mask=visual_attention_mask,
                labels=labels,  # 历史部分不计算损失，只对目标部分计算
                output_hidden_states=True,
                return_dict=True,
            )
            
            # 获取logits和隐藏状态
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]
            hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_dim]
            
            # 解码预测的文本
            # 只取历史文本之后的部分进行解码，这才是模型实际生成的部分
            pred_tokens = torch.argmax(logits[:, history_len:, :], dim=-1)
            decoded_texts = []
            for tokens in pred_tokens:
                text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                decoded_texts.append(text)
                
            # 获取损失
            loss_lm = outputs.loss  # 这个损失已经只计算了标签不为-100的位置
            
            return logits, hidden_states, decoded_texts, loss_lm
            
        else:  # mode == "generate"
            # 使用历史文本作为起始点生成内容
            params = {
                "history_input_ids": history_input_ids,
                "history_attention_mask": history_attention_mask,
                "visual_features": projected_visual,
                "visual_attention_mask": visual_attention_mask,
            }
            
            # 如果提供了生成参数，将它们添加到参数字典中
            if generation_params:
                # 确保num_beams和early_stopping参数一致
                if "num_beams" in generation_params:
                    num_beams = generation_params["num_beams"]
                    # 如果未指定early_stopping，根据num_beams设置
                    if "early_stopping" not in generation_params:
                        generation_params["early_stopping"] = num_beams > 1
                
                params.update(generation_params)
                
            return self.generate(**params)
            
    def generate(
        self,
        history_input_ids,
        history_attention_mask,
        visual_features,
        visual_attention_mask,
        num_beams=3,
        max_length=100,
        do_sample=True,  # 默认开启采样
        top_p=0.9,
        temperature=0.7,  # 添加温度参数
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
            max_length: 生成的最大长度
            do_sample: 是否使用采样
            top_p: 采样的概率阈值
            temperature: 采样的温度
            repetition_penalty: 重复惩罚系数

        Returns:
            generated_texts: 生成的文本列表
        """
        # 准备生成所需的输入
        input_ids = history_input_ids
        
        # 处理扩展后的视觉特征
        extended_vis_attention = visual_attention_mask.view(
            visual_attention_mask.size(0), 1, 1, visual_attention_mask.size(1)
        ).expand(-1, self.text_decoder.config.num_attention_heads, -1, -1)
        
        # 准备交叉注意力参数
        model_kwargs = {
            "encoder_hidden_states": visual_features,
            "encoder_attention_mask": extended_vis_attention,
        }
        
        # 处理Mask
        attention_mask = history_attention_mask
        
        # 计算实际需要生成的最大长度（考虑已有的历史长度）
        history_length = input_ids.size(1)
        actual_max_length = min(max_length, history_length + 150)  # 确保不超过模型最大长度
        
        # 配置生成参数
        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_length": actual_max_length,
            "num_beams": num_beams,  # 无论采样与否，都使用设定的beam数量
            "early_stopping": num_beams > 1,  # 只有当num_beams>1时设置early_stopping为True
            "eos_token_id": self.tokenizer.sep_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "repetition_penalty": repetition_penalty,
            "do_sample": do_sample,  # 控制是否采样
        }
        
        # 只有在do_sample=True时才设置top_p和temperature
        if do_sample:
            generation_kwargs["top_p"] = top_p
            generation_kwargs["temperature"] = temperature
            
        # 添加交叉注意力参数
        generation_kwargs.update(model_kwargs)
        
        # 生成文本
        outputs = self.text_decoder.generate(**generation_kwargs)
        
        # 解码生成的文本，去除历史文本部分，只保留新生成的内容
        generated_texts = []
        for i, tokens in enumerate(outputs):
            # 获取当前批次样本的历史长度
            curr_history_len = history_length
            
            # 只解码历史之后生成的内容
            generated_part = tokens[curr_history_len:]
            
            # 解码生成的部分
            text = self.tokenizer.decode(generated_part, skip_special_tokens=True)
            generated_texts.append(text)
            
        return generated_texts 