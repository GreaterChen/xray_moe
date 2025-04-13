import torch
import torch.nn as nn
import torch.nn.functional as F

from models.bert_cross_decoder import BertCrossDecoder

class MoEBertAdapter(nn.Module):
    """
    MOE模型的BERT解码器适配器，用于替代原来的LLM解码器
    """
    def __init__(
        self,
        config,
        tokenizer=None,
        hidden_dim=768,
        max_length=196
    ):
        super(MoEBertAdapter, self).__init__()
        
        # 创建BERT交叉解码器
        self.decoder = BertCrossDecoder(
            config=config,
            tokenizer=tokenizer,
            hidden_dim=hidden_dim,
            max_length=max_length
        )
        
        self.tokenizer = self.decoder.tokenizer
    
    def forward(
        self,
        visual_features,
        history_encoding,
        findings,
        attention_mask=None,
        labels=None,
    ):
        """
        适配MOE模型的前向传播接口
        
        Args:
            visual_features: 视觉特征 [batch_size, num_tokens, visual_dim]
            history_encoding: 历史文本的编码 {input_ids, attention_mask} 或 BatchEncoding
            findings: 报告文本编码 {input_ids, attention_mask} 或 BatchEncoding
            attention_mask: 注意力掩码 [batch_size, seq_len]
            labels: 标签 [batch_size, seq_len]
        """
        # 调用BERT交叉解码器
        logits, hidden_states, decoded_texts, loss = self.decoder(
            visual_features=visual_features,
            history=history_encoding,
            target_text=findings,
            mode="train"
        )
        
        # 构造类似于Llama/Mistral模型输出的格式
        outputs = type('BertOutputs', (), {})()
        outputs.loss = loss
        outputs.logits = logits
        outputs.hidden_states = hidden_states
        outputs.decoded_texts = decoded_texts
        
        return outputs
    
    def generate(
        self,
        visual_features,
        history_encoding,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.0,
        num_beams=3,
    ):
        """
        适配MOE模型的生成接口
        
        Args:
            visual_features: 视觉特征 [batch_size, num_tokens, visual_dim]
            history_encoding: 历史文本的编码 {input_ids, attention_mask} 或 BatchEncoding
            max_new_tokens: 生成文本的最大新token数量
            do_sample: 是否采样生成
            temperature: 温度参数
            top_p: 概率截断阈值
            repetition_penalty: 重复惩罚系数
            num_beams: beam search的宽度
        """
        # 确保history_encoding是有效的格式
        if not hasattr(history_encoding, 'input_ids') and not (isinstance(history_encoding, dict) and 'input_ids' in history_encoding):
            raise ValueError(f"历史文本编码必须包含input_ids，当前类型: {type(history_encoding)}")
            
        # 调用BERT交叉解码器生成文本，传递所有生成参数
        return self.decoder(
            visual_features=visual_features,
            history=history_encoding,
            mode="generate", 
            generation_params={
                "max_length": max_new_tokens + history_encoding.input_ids.size(1),  # 添加历史长度
                "do_sample": do_sample,
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "num_beams": num_beams
                # early_stopping参数由decoder的generate方法根据num_beams自动设置
            }
        ) 