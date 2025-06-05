import torch
import torch.nn as nn
import torch.nn.functional as F

from models.bert_cross_decoder import BertCrossDecoder

class MoEBertAdapter(nn.Module):
    """
    MOE模型的BERT解码器适配器，用于替代原来的LLM解码器
    支持增强的文本输入
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
        self.config = config
    
    def _prepare_history_input(self, history_encoding):
        """
        准备历史文本输入，处理各种格式
        
        Args:
            history_encoding: 历史文本编码，可能是字符串列表、BatchEncoding或dict
            
        Returns:
            准备好的历史文本编码
        """
        if history_encoding is None:
            return None
            
        # 如果是字符串列表，重新编码
        if isinstance(history_encoding, list) and all(isinstance(h, str) for h in history_encoding):
            max_length = getattr(self.config, 'MAX_LEN_HISTORY', 50)
            return self.tokenizer(
                history_encoding,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
        
        # 如果已经是正确的格式，直接返回
        return history_encoding
    
    def forward(
        self,
        visual_features,
        history_encoding,
        findings,
        attention_mask=None,
        labels=None,
        use_history=False,  # 添加use_history参数
    ):
        """
        适配MOE模型的前向传播接口
        
        Args:
            visual_features: 视觉特征 [batch_size, num_tokens, visual_dim]
            history_encoding: 历史文本的编码 {input_ids, attention_mask} 或 BatchEncoding 或 str list
            findings: 报告文本编码 {input_ids, attention_mask} 或 BatchEncoding
            attention_mask: 注意力掩码 [batch_size, seq_len]
            labels: 标签 [batch_size, seq_len]
            use_history: 是否使用历史文本作为prompt，如为False则仅使用视觉特征
        """
        # 准备历史文本输入
        prepared_history = self._prepare_history_input(history_encoding) if use_history else None
        
        # 调用BERT交叉解码器
        logits, hidden_states, decoded_texts, loss = self.decoder(
            visual_features=visual_features,
            history=prepared_history,
            target_text=findings,
            mode="train",
            use_history=use_history  # 传递use_history参数
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
        use_history=False,  # 添加use_history参数
    ):
        """
        适配MOE模型的生成接口
        
        Args:
            visual_features: 视觉特征 [batch_size, num_tokens, visual_dim]
            history_encoding: 历史文本的编码 {input_ids, attention_mask} 或 BatchEncoding 或 str list
            max_new_tokens: 生成文本的最大新token数量
            do_sample: 是否采样生成
            temperature: 温度参数
            top_p: 概率截断阈值
            repetition_penalty: 重复惩罚系数
            num_beams: beam search的宽度
            use_history: 是否使用历史文本
        """
        # 准备历史文本输入
        prepared_history = self._prepare_history_input(history_encoding) if use_history else None
        
        # 验证历史文本编码格式
        if use_history and prepared_history is not None:
            if not hasattr(prepared_history, 'input_ids') and not (isinstance(prepared_history, dict) and 'input_ids' in prepared_history):
                print(f"警告: 历史文本编码格式不正确，类型: {type(prepared_history)}")
                prepared_history = None
                use_history = False
            
        # 调用BERT交叉解码器生成文本，传递所有生成参数
        return self.decoder(
            visual_features=visual_features,
            history=prepared_history,
            mode="generate", 
            generation_params={
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": do_sample,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "num_beams": num_beams
            },
            use_history=use_history  # 传递use_history参数
        ) 