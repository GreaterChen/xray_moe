"""
Qwen2.5-VL-3B 解码器适配器
使用Qwen2.5-VL-3B作为decoder，视觉特征和疾病特征分别经过线性映射层
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer

# 获取logger
qwen_decoder_logger = logging.getLogger("train_logger")


class Qwen2VLDecoder(nn.Module):
    """
    Qwen2.5-VL-3B解码器适配器
    
    特点：
    1. 使用Qwen2.5-VL-3B作为生成模型
    2. 视觉特征和疾病特征分别经过独立的线性映射层
    3. 支持训练和生成两种模式
    """
    
    def __init__(
        self,
        config,
        tokenizer=None,
        visual_dim=768,
        disease_dim=768,
        num_visual_tokens=30,  # CLS + 29个区域
        num_disease_tokens=14,  # 14个疾病
        qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    ):
        super().__init__()
        
        self.config = config
        self.visual_dim = visual_dim
        self.disease_dim = disease_dim
        self.num_visual_tokens = num_visual_tokens
        self.num_disease_tokens = num_disease_tokens
        
        # 加载Qwen2.5-VL-3B模型
        qwen_decoder_logger.info(f"加载Qwen2.5-VL模型: {qwen_model_name}")
        self.qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
            qwen_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        # 加载tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
            qwen_decoder_logger.info("使用传入的tokenizer")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(qwen_model_name)
            qwen_decoder_logger.info("从Qwen2.5-VL加载tokenizer")
        
        # 获取Qwen模型的隐藏维度
        self.qwen_hidden_dim = self.qwen_model.config.hidden_size
        qwen_decoder_logger.info(f"Qwen模型隐藏维度: {self.qwen_hidden_dim}")
        
        # 视觉特征线性映射层：将视觉特征映射到Qwen的隐藏维度
        self.visual_projection = nn.Linear(visual_dim, self.qwen_hidden_dim)
        qwen_decoder_logger.info(f"视觉投影层: {visual_dim} -> {self.qwen_hidden_dim}")
        
        # 疾病特征线性映射层：将疾病特征映射到Qwen的隐藏维度
        self.disease_projection = nn.Linear(disease_dim, self.qwen_hidden_dim)
        qwen_decoder_logger.info(f"疾病投影层: {disease_dim} -> {self.qwen_hidden_dim}")
        
        qwen_decoder_logger.info("✅ Qwen2.5-VL解码器初始化完成")
    
    def _project_features(self, visual_features, disease_features=None):
        """
        将视觉特征和疾病特征分别投影到Qwen的隐藏维度
        
        Args:
            visual_features: [B, num_visual_tokens, visual_dim]
            disease_features: [B, num_disease_tokens, disease_dim] 或 None
            
        Returns:
            projected_features: [B, num_tokens, qwen_hidden_dim]
        """
        # 投影视觉特征
        projected_visual = self.visual_projection(visual_features)  # [B, num_visual, qwen_hidden_dim]
        
        if disease_features is not None:
            # 投影疾病特征
            projected_disease = self.disease_projection(disease_features)  # [B, num_disease, qwen_hidden_dim]
            
            # 拼接视觉特征和疾病特征
            projected_features = torch.cat([projected_visual, projected_disease], dim=1)  # [B, num_visual+num_disease, qwen_hidden_dim]
        else:
            projected_features = projected_visual
        
        return projected_features
    
    def _prepare_inputs_for_qwen(self, combined_features, text_input_ids=None, text_attention_mask=None):
        """
        准备Qwen模型的输入
        
        Args:
            combined_features: 投影后的组合特征 [B, num_tokens, qwen_hidden_dim]
            text_input_ids: 文本输入的token ids [B, text_len]
            text_attention_mask: 文本注意力掩码 [B, text_len]
            
        Returns:
            inputs_embeds: 嵌入向量 [B, total_len, qwen_hidden_dim]
            attention_mask: 注意力掩码 [B, total_len]
        """
        batch_size = combined_features.size(0)
        num_feature_tokens = combined_features.size(1)
        device = combined_features.device
        
        # 创建特征的attention mask (全为1)
        feature_attention_mask = torch.ones(
            batch_size, num_feature_tokens, dtype=torch.long, device=device
        )
        
        if text_input_ids is not None:
            # 获取文本嵌入
            text_embeds = self.qwen_model.get_input_embeddings()(text_input_ids)
            
            # 拼接特征嵌入和文本嵌入
            # 格式: [特征tokens] + [文本tokens]
            inputs_embeds = torch.cat([combined_features, text_embeds], dim=1)
            
            # 拼接attention mask
            attention_mask = torch.cat([feature_attention_mask, text_attention_mask], dim=1)
        else:
            # 只有特征，没有文本（生成模式）
            inputs_embeds = combined_features
            attention_mask = feature_attention_mask
        
        return inputs_embeds, attention_mask
    
    def forward(
        self,
        visual_features,
        history,
        target_text=None,
        mode="train",
        generation_params=None,
        use_history=False,
        disease_features=None,  # 新增：疾病特征
    ):
        """
        前向传播
        
        Args:
            visual_features: 视觉特征 [B, num_visual_tokens, visual_dim]
            history: 历史文本编码 (可选，暂不使用)
            target_text: 目标文本编码 {input_ids, attention_mask}
            mode: "train" 或 "generate"
            generation_params: 生成参数字典
            use_history: 是否使用历史文本
            disease_features: 疾病特征 [B, num_disease_tokens, disease_dim]
            
        Returns:
            训练模式: (logits, hidden_states, decoded_texts, loss)
            生成模式: 生成的文本列表
        """
        batch_size = visual_features.size(0)
        device = visual_features.device
        
        # 1. 投影视觉和疾病特征
        combined_features = self._project_features(visual_features, disease_features)
        
        if mode == "train" and target_text is not None:
            # 训练模式
            # 处理目标文本
            if hasattr(target_text, 'input_ids'):
                target_input_ids = target_text.input_ids.to(device)
                target_attention_mask = target_text.attention_mask.to(device)
            elif isinstance(target_text, dict):
                target_input_ids = target_text['input_ids'].to(device)
                target_attention_mask = target_text['attention_mask'].to(device)
            else:
                raise ValueError(f"target_text必须是BatchEncoding或字典，当前类型: {type(target_text)}")
            
            # 准备Qwen输入
            inputs_embeds, attention_mask = self._prepare_inputs_for_qwen(
                combined_features, target_input_ids, target_attention_mask
            )
            
            # 创建标签：只计算文本部分的损失
            # 特征部分的标签设为-100
            num_feature_tokens = combined_features.size(1)
            feature_labels = torch.full(
                (batch_size, num_feature_tokens), -100, dtype=torch.long, device=device
            )
            text_labels = target_input_ids.clone()
            # 将padding位置设为-100
            text_labels[target_attention_mask == 0] = -100
            
            # 拼接标签
            labels = torch.cat([feature_labels, text_labels], dim=1)
            
            # Qwen前向传播
            outputs = self.qwen_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True,
                return_dict=True,
            )
            
            logits = outputs.logits
            hidden_states = outputs.hidden_states[-1] if outputs.hidden_states else None
            loss = outputs.loss
            decoded_texts = None  # 训练时不解码，节省时间
            
            return logits, hidden_states, decoded_texts, loss
        
        else:
            # 生成模式
            params = generation_params or {}
            return self.generate(
                combined_features=combined_features,
                **params
            )
    
    def generate(
        self,
        combined_features,
        num_beams=3,
        max_new_tokens=150,
        min_length=100,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        repetition_penalty=1.0,
    ):
        """
        生成文本
        
        Args:
            combined_features: 投影后的组合特征 [B, num_tokens, qwen_hidden_dim]
            num_beams: beam search宽度
            max_new_tokens: 最大生成token数
            min_length: 最小生成长度
            do_sample: 是否采样
            top_p: nucleus sampling参数
            temperature: 温度参数
            repetition_penalty: 重复惩罚
            
        Returns:
            generated_texts: 生成的文本列表
        """
        batch_size = combined_features.size(0)
        device = combined_features.device
        
        # 准备输入（只有特征，没有文本）
        inputs_embeds, attention_mask = self._prepare_inputs_for_qwen(combined_features)
        
        # 生成配置
        generation_config = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "min_length": min_length,
            "num_beams": num_beams,
            "do_sample": do_sample if num_beams == 1 else False,  # beam search时不能采样
            "temperature": temperature if do_sample else 1.0,
            "top_p": top_p if do_sample else 1.0,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # 生成
        with torch.no_grad():
            outputs = self.qwen_model.generate(**generation_config)
        
        # 解码生成的文本
        # 注意：outputs包含了输入的特征tokens，我们只需要新生成的部分
        num_input_tokens = inputs_embeds.size(1)
        generated_texts = []
        
        for i in range(batch_size):
            # 只解码新生成的token（跳过输入的特征tokens）
            generated_tokens = outputs[i][num_input_tokens:]
            text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            generated_texts.append(text)
        
        return generated_texts


class Qwen2VLAdapter(nn.Module):
    """
    Qwen2.5-VL解码器适配器（与BertAdapter接口兼容）
    """
    
    def __init__(
        self,
        config,
        tokenizer=None,
        hidden_dim=768,
        max_length=196,
        qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    ):
        super().__init__()
        
        self.config = config
        
        # 创建Qwen2.5-VL解码器
        self.decoder = Qwen2VLDecoder(
            config=config,
            tokenizer=tokenizer,
            visual_dim=hidden_dim,
            disease_dim=hidden_dim,
            qwen_model_name=qwen_model_name,
        )
        
        self.tokenizer = self.decoder.tokenizer
    
    def forward(
        self,
        visual_features,
        history_encoding,
        findings,
        attention_mask=None,
        labels=None,
        use_history=False,
    ):
        """
        医学报告生成模型的前向传播接口（与BertAdapter兼容）
        
        Args:
            visual_features: 组合特征 [B, num_tokens, hidden_dim]
                            可能是 [B, 30, 768] (只有视觉) 或 [B, 44, 768] (视觉+疾病)
            history_encoding: 历史文本编码（暂不使用）
            findings: 报告文本编码
            attention_mask: 注意力掩码
            labels: 标签
            use_history: 是否使用历史文本
        """
        # 分离视觉特征和疾病特征
        # 前30个token是视觉特征(CLS + 29个区域)
        # 如果总长度>30，后14个是疾病特征
        if visual_features.size(1) > 30:
            visual_only = visual_features[:, :30, :]  # [B, 30, 768]
            disease_only = visual_features[:, 30:, :]  # [B, 14, 768]
        else:
            visual_only = visual_features
            disease_only = None
        
        # 调用Qwen解码器
        logits, hidden_states, decoded_texts, loss = self.decoder(
            visual_features=visual_only,
            history=history_encoding,
            target_text=findings,
            mode="train",
            use_history=use_history,
            disease_features=disease_only,
        )
        
        # 构造输出对象（与BertAdapter兼容）
        class Qwen2VLOutputs(dict):
            """同时支持字典操作和属性访问的输出类"""
            def __getattr__(self, key):
                try:
                    return self[key]
                except KeyError:
                    raise AttributeError(f"'Qwen2VLOutputs' object has no attribute '{key}'")
            
            def __setattr__(self, key, value):
                self[key] = value
        
        outputs = Qwen2VLOutputs()
        outputs.loss = loss
        outputs.logits = logits
        outputs.hidden_states = None  # 减少内存占用
        outputs.decoded_texts = decoded_texts
        
        return outputs
    
    def generate(
        self,
        visual_features,
        history_encoding,
        max_new_tokens=None,
        do_sample=None,
        temperature=None,
        top_p=None,
        repetition_penalty=None,
        num_beams=None,
        use_history=False,
    ):
        """
        医学报告生成模型的生成接口（与BertAdapter兼容）
        
        Args:
            visual_features: 组合特征 [B, num_tokens, hidden_dim]
            history_encoding: 历史文本编码
            max_new_tokens: 最大生成token数
            do_sample: 是否采样
            temperature: 温度参数
            top_p: nucleus sampling参数
            repetition_penalty: 重复惩罚
            num_beams: beam search宽度
            use_history: 是否使用历史文本
        """
        # 从config读取默认参数
        if max_new_tokens is None:
            max_new_tokens = getattr(self.config, 'GEN_MAX_NEW_TOKENS', 150)
        if do_sample is None:
            do_sample = getattr(self.config, 'GEN_DO_SAMPLE', False)
        if temperature is None:
            temperature = getattr(self.config, 'GEN_TEMPERATURE', 0.7)
        if top_p is None:
            top_p = getattr(self.config, 'GEN_TOP_P', 0.9)
        if repetition_penalty is None:
            repetition_penalty = getattr(self.config, 'GEN_REPETITION_PENALTY', 1.0)
        if num_beams is None:
            num_beams = getattr(self.config, 'GEN_NUM_BEAMS', 3)
        
        min_length = getattr(self.config, 'GEN_MIN_LENGTH', 100)
        
        # 分离视觉特征和疾病特征
        if visual_features.size(1) > 30:
            visual_only = visual_features[:, :30, :]
            disease_only = visual_features[:, 30:, :]
        else:
            visual_only = visual_features
            disease_only = None
        
        # 调用Qwen解码器生成
        return self.decoder(
            visual_features=visual_only,
            history=history_encoding,
            mode="generate",
            generation_params={
                "max_new_tokens": max_new_tokens,
                "min_length": min_length,
                "temperature": temperature,
                "do_sample": do_sample,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "num_beams": num_beams,
            },
            use_history=use_history,
            disease_features=disease_only,
        )

