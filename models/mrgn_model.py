import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from transformers import AutoModel, AutoTokenizer
import sentencepiece as spm
import math
from models.med import BertConfig, BertModel, BertLMHeadModel
from transformers import BertTokenizer
import torchvision.models as models
import torch
from torch import nn
import torch.nn.functional as F

from models.transformer import Transformer
from transformers import SwinModel


class BLIP_Decoder(nn.Module):
    def __init__(
        self,
        args,
        tokenizer,
        hidden_dim=768,
        max_length=196,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        # 加载BERT配置
        decoder_config = BertConfig.from_json_file(
            os.path.join(args.root_dir, "configs/bert_config.json")
        )
        decoder_config.encoder_width = hidden_dim
        decoder_config.add_cross_attention = True
        decoder_config.is_decoder = True

        # 初始化解码器
        self.text_decoder = BertLMHeadModel.from_pretrained(
            "bert-base-uncased", config=decoder_config, local_files_only=True
        )

        # 调整词表大小
        self.text_decoder.resize_token_embeddings(len(self.tokenizer))

    def forward(self, encoder_hidden_states, text=None):
        """
        Args:
            encoder_hidden_states: 视觉特征，形状 (B, N_v, C_v)
            target_ids: 目标文本的token ids，形状 (B, max_len)
        Returns:
            output: 生成的词汇分布，形状 (B, max_len, vocab_size)
            hidden_states: 文本特征，形状 (B, max_len, hidden_dim)
            decoded_texts: 生成的文本列表
        """
        if text is not None:
            # 输入保持不变: [CLS, A, B, C, SEP, PAD]
            decoder_inputs = text.input_ids

            # mask掉PAD位置
            decoder_targets = decoder_inputs.masked_fill(
                decoder_inputs == self.tokenizer.pad_token_id, -100
            )

            # mask掉CLS位置
            # decoder_targets = decoder_inputs.masked_fill(
            #     decoder_inputs == self.tokenizer.bos_token_id, -100
            # )

            # 前向传播
            outputs = self.text_decoder(
                input_ids=decoder_inputs,
                attention_mask=text.attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                labels=decoder_targets,
                output_hidden_states=True,  # 添加这个参数以获取隐藏状态
                return_dict=True,
            )

            # 获取logits和隐藏状态
            logits = outputs.logits  # (B, max_len, vocab_size)
            hidden_states = outputs.hidden_states[-1]  # (B, max_len, hidden_dim)

            # 解码生成的文本
            pred_tokens = torch.argmax(logits, dim=-1)
            decoded_texts = []
            for tokens in pred_tokens:
                text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                decoded_texts.append(text)

            loss_lm = outputs.loss

            return logits, hidden_states, decoded_texts, loss_lm

        else:
            # 使用generate方法生成文本
            logits, hidden_states, captions = self.generate(encoder_hidden_states)
            return logits, hidden_states, captions, 0

    def generate(
        self,
        image_embeds,
        sample=False,
        num_beams=3,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        batch_size = image_embeds.size(0)

        # 创建attention mask
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image_embeds.device
        )
        model_kwargs = {
            "encoder_hidden_states": image_embeds,
            "encoder_attention_mask": image_atts,
        }

        # 初始化输入
        input_ids = torch.ones((batch_size, 1), dtype=torch.long).to(
            image_embeds.device
        )
        # input_ids[:, 0] = self.tokenizer.bos_token_id
        input_ids[:, 0] = self.tokenizer.cls_token_id

        # 生成文本
        outputs = self.text_decoder.generate(
            input_ids=input_ids,
            max_new_tokens=self.max_length - 1,
            num_beams=num_beams,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            **model_kwargs
        )

        # 获取生成的序列
        # generated_tokens = outputs.sequences  # (batch_size, seq_len)
        generated_tokens = outputs
        padded_tokens = torch.full(
            (batch_size, self.max_length), self.tokenizer.pad_token_id, dtype=torch.long
        ).to(image_embeds.device)
        for i, tokens in enumerate(generated_tokens):
            seq_len = len(tokens)
            padded_tokens[i, :seq_len] = tokens
        generated_tokens = padded_tokens

        # 使用forward pass获取hidden states
        attention_mask = (generated_tokens != self.tokenizer.pad_token_id).long()

        decoder_outputs = self.text_decoder(
            input_ids=generated_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = decoder_outputs.hidden_states[-1]  # 获取最后一层的hidden states

        # 解码生成的文本
        captions = []
        for tokens in generated_tokens:
            caption = self.tokenizer.decode(tokens, skip_special_tokens=True)
            captions.append(caption)

        return None, hidden_states, captions

    def get_text_embeddings(self, texts):
        """
        获取文本的token embedding表示

        Args:
            texts: 文本字符串或文本字符串列表

        Returns:
            embeddings: token的嵌入表示 (batch_size, max_len, hidden_dim)
        """
        if isinstance(texts, list) and isinstance(texts[0], str):
            device = self.text_decoder.bert.embeddings.word_embeddings.weight.device
            # tokenization
            encoded = self.tokenizer(
                texts,
                max_length=196,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device)
        else:
            encoded = texts

        # 获取embeddings
        embeddings = self.text_decoder.bert.embeddings.word_embeddings(
            encoded.input_ids
        )
        return embeddings


class ResNet101(nn.Module):
    def __init__(self):
        super(ResNet101, self).__init__()
        model = getattr(models, "resnet101")(
            weights=models.ResNet101_Weights.IMAGENET1K_V1
        )
        modules = list(model.children())[:-3]
        self.model = nn.Sequential(*modules)

        # 添加线性映射层，将1024维降到768维
        self.dim_reduction = nn.Sequential(
            nn.Linear(1024, 512),  # 降到中间的低维度
            nn.ReLU(),  # 非线性激活
            nn.Linear(512, 768),  # 再升到目标维度
        )

    def forward(self, x):
        patch_feats = self.model(x)  # (batch_size, 1024, 14, 14)
        batch_size, feat_size, _, _ = patch_feats.shape

        # 重排并降维
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(
            0, 2, 1
        )  # (batch_size, 196, 1024)
        patch_feats = self.dim_reduction(patch_feats)  # (batch_size, 196, 768)

        patch_feats = F.normalize(patch_feats, p=2, dim=-1)

        return patch_feats


class HistoryEncoder(nn.Module):
    def __init__(self, args=None, bert_model_name="bert-base-uncased"):
        super().__init__()
        # 加载预训练BERT的embedding层
        self.args = args
        self.bert = BertModel.from_pretrained(bert_model_name, local_files_only=True)
        self.bert.resize_token_embeddings(args.tokenizer_max_len)

        self.embedding = self.bert.embeddings

    def forward(self, encoded_history):
        """
        Args:
            encoded_history: 包含input_ids, attention_mask等的字典
                           来自tokenizer的输出
        Returns:
            tensor: shape (batch_size, seq_len, 768)
        """
        # 获取embeddings
        embeddings = self.embedding(
            input_ids=encoded_history.input_ids,
            position_ids=None,
            inputs_embeds=None,
            past_key_values_length=0,
        )

        return embeddings


class CXR_BERT_FeatureExtractor(nn.Module):
    def __init__(self, device="cuda"):
        super(CXR_BERT_FeatureExtractor, self).__init__()
        self.device = device
        # 加载预训练的 CXR-BERT 模型和对应的分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/BiomedVLP-CXR-BERT-specialized", trust_remote_code=True, local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            "microsoft/BiomedVLP-CXR-BERT-specialized", trust_remote_code=True, local_files_only=True
        ).to(self.device)

        # Freeze all parameters in the BERT model
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()  # 设置模型为评估模式

    def forward(self, texts):
        """
        Args:
            texts: List[str], batch_size个生成的文本字符串
        Returns:
            cls_embeddings: 文本特征张量，形状为 (B, hidden_size)
            texts: 输入的文本列表
        """
        # 对输入文本进行编码
        inputs = self.tokenizer(
            texts, max_length=196, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # 获取 [CLS] 标记的嵌入表示
            cls_embeddings = outputs.last_hidden_state[:, 0, :]

        return cls_embeddings


class ModalityFusion(nn.Module):
    def __init__(
        self, hidden_size=768, num_heads=1, num_layers=1, mlp_ratio=4.0, dropout=0.1
    ):
        super().__init__()

        # Transformer Encoder层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=int(hidden_size * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_size),
        )

        # 模态类型嵌入
        self.modality_embeddings = nn.Parameter(torch.randn(2, hidden_size))

    def forward(self, visual_features, text_features):
        """
        Args:
            visual_features: [batch_size, 196, hidden_size] # 已经过MLP映射到768维
            text_features: [batch_size, text_len, hidden_size]
        """
        # 1. 直接添加模态类型嵌入
        visual_modality = self.modality_embeddings[0].unsqueeze(0).unsqueeze(0)
        text_modality = self.modality_embeddings[1].unsqueeze(0).unsqueeze(0)

        visual_features = visual_features + visual_modality
        text_features = text_features + text_modality

        # 2. 特征拼接
        concat_features = torch.cat([visual_features, text_features], dim=1)

        # 3. Transformer融合
        fused_features = self.transformer(concat_features)

        return fused_features


class FindingsGenerator(nn.Module):
    def __init__(self, text_decoder):
        """
        Findings Generator 使用 BLIP_Decoder 实现。
        Args:
            text_decoder: BLIP_Decoder 实例
        """
        super(FindingsGenerator, self).__init__()
        self.text_decoder = text_decoder

    def forward(self, F_v, target_embed=None):
        """
        Args:
            F_v: 输入的视觉特征，形状 (B, N_v, C_v)
            target_embed: 目标文本的token ids，形状 (B, max_len)
        Returns:
            logits: 生成的词汇分布，形状 (B, max_len, vocab_size)
            F_t: 解码器的隐藏状态，形状 (B, max_len, hidden_dim)
            findings_text: 生成的文本列表
            loss_lm: 损失值
        """
        # 使用 BLIP_Decoder 进行生成
        logits, F_t, findings_text, loss_lm = self.text_decoder(F_v, target_embed)

        return logits, F_t, findings_text, loss_lm


class CoAttentionBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=6):
        """
        单个 Co-Attention Block 的实现，包含以下步骤：
        1. 自注意力 (Self-Attention)
        2. 标准 Cross-Attention (Cross-Attention-1)
        3. 非对称 Cross-Attention (Cross-Attention-2)
        Args:
            embed_dim: 特征嵌入维度。
            num_heads: 注意力头的数量。
        """
        super(CoAttentionBlock, self).__init__()
        # 自注意力层
        self.self_attn_text = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads
        )
        self.self_attn_visual = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads
        )

        # 标准 Cross-Attention 层
        self.cross_attn_text_to_visual = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads
        )
        self.cross_attn_visual_to_text = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads
        )

        # 非对称 Cross-Attention 层
        self.cross_attn_asym_text = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads
        )
        self.cross_attn_asym_visual = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads
        )

        # 残差连接和归一化层
        self.norm_text1 = nn.LayerNorm(embed_dim)
        self.norm_visual1 = nn.LayerNorm(embed_dim)
        self.norm_text2 = nn.LayerNorm(embed_dim)
        self.norm_visual2 = nn.LayerNorm(embed_dim)
        self.norm_text3 = nn.LayerNorm(embed_dim)
        self.norm_visual3 = nn.LayerNorm(embed_dim)

    def forward(self, F_t, F_v):
        """
        Args:
            F_t: 文本特征，形状 (B, N_t, C_t)。
            F_v: 视觉特征，形状 (B, N_v, C_v)。
        Returns:
            F_t': 增强的文本特征，形状 (B, N_t, C_t)。
            F_v': 增强的视觉特征，形状 (B, N_v, C_v)。
        """
        # 转置到 (seq_len, batch_size, embed_dim) 格式，符合 MultiheadAttention 要求
        F_t = F_t.transpose(0, 1)  # (N_t, B, C_t)
        F_v = F_v.transpose(0, 1)  # (N_v, B, C_v)

        # Step 1: 自注意力
        F_t1, _ = self.self_attn_text(F_t, F_t, F_t)  # 文本自注意力
        F_t1 = self.norm_text1(F_t + F_t1)  # 残差连接 + 归一化

        F_v1, _ = self.self_attn_visual(F_v, F_v, F_v)  # 视觉自注意力
        F_v1 = self.norm_visual1(F_v + F_v1)  # 残差连接 + 归一化

        # Step 2: 标准 Cross-Attention
        F_t2, _ = self.cross_attn_visual_to_text(F_t1, F_v1, F_v1)  # 文本 -> 视觉
        F_t2 = self.norm_text2(F_t1 + F_t2)  # 残差连接 + 归一化

        F_v2, _ = self.cross_attn_text_to_visual(F_v1, F_t1, F_t1)  # 视觉 -> 文本
        F_v2 = self.norm_visual2(F_v1 + F_v2)  # 残差连接 + 归一化

        # # Step 3: 非对称 Cross-Attention
        # F_t3, _ = self.cross_attn_asym_text(
        #     F_t2, F_t2, F_v2
        # )  # Query 和 Key 是文本，Value 是视觉
        # F_t3 = self.norm_text3(F_t2 + F_t3)  # 残差连接 + 归一化

        # F_v3, _ = self.cross_attn_asym_visual(
        #     F_v2, F_v2, F_t2
        # )  # Query 和 Key 是视觉，Value 是文本
        # F_v3 = self.norm_visual3(F_v2 + F_v3)  # 残差连接 + 归一化

        # # 转回 (B, seq_len, embed_dim) 格式
        # F_t3 = F_t3.transpose(0, 1)  # (B, N_t, C_t)
        # F_v3 = F_v3.transpose(0, 1)  # (B, N_v, C_v)

        F_t3 = F_t2.transpose(0, 1)  # (B, N_t, C_t)
        F_v3 = F_v2.transpose(0, 1)  # (B, N_v, C_v)

        return F_t3, F_v3


class CoAttentionModule(nn.Module):
    def __init__(self, embed_dim=768, num_heads=6, num_blocks=6):
        """
        Co-Attention 模块，由多个 Co-Attention Block 组成。
        Args:
            embed_dim: 特征嵌入维度。
            num_heads: 注意力头的数量。
            num_blocks: Co-Attention Block 的数量。
        """
        super(CoAttentionModule, self).__init__()
        self.blocks = nn.ModuleList(
            [
                CoAttentionBlock(embed_dim=embed_dim, num_heads=num_heads)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, F_t, F_v):
        """
        Args:
            F_t: 文本特征，形状 (B, N_t, C_t)。
            F_v: 视觉特征，形状 (B, N_v, C_v)。
        Returns:
            F_t': 增强的文本特征，形状 (B, N_t, C_t)。
            F_v': 增强的视觉特征，形状 (B, N_v, C_v)。
        """
        for block in self.blocks:
            F_t, F_v = block(F_t, F_v)
        return F_t, F_v


class MultiLabelClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=384):
        """
        多标签分类器
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
        """
        super(MultiLabelClassifier, self).__init__()

        # 全局平均池化，用于将序列特征压缩为单个向量
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 分类器网络
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 添加层归一化提高稳定性
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 14),  # 14个类别
        )

    def forward(self, memory):
        """
        Args:
            memory: 拼接后的特征 (B, seq_len, hidden_dim)
        Returns:
            class_logits: 14个类别的预测概率 (B, 14)
        """
        # 全局平均池化
        pooled_features = self.global_pool(memory.transpose(1, 2)).squeeze(
            -1
        )  # (B, hidden_dim)

        # 分类预测
        class_logits = self.classifier(pooled_features)  # (B, 14)

        return class_logits


class ImpressionGenerator(nn.Module):
    def __init__(self, text_decoder):
        """
        Impression Generator 使用 BLIP_Decoder 实现。
        Args:
            text_decoder: BLIP_Decoder 实例
        """
        super(ImpressionGenerator, self).__init__()
        self.text_decoder = text_decoder

    def forward(self, F_v_prime, F_t_prime, F_v, target_embed=None, memory=None):
        """
        Args:
            F_v_prime: 增强的视觉特征 (B, N_v, C_v)
            F_t_prime: 增强的文本特征 (B, N_t, C_t)
            F_v: 原始的视觉特征 (B, N_v, C_v)
            target_embed: 目标文本的token ids，形状 (B, max_len)
        Returns:
            memory: 拼接后的特征 (B, N_v + 2*N_t, C)
            output: 生成的词汇分布 (B, max_len, vocab_size)
        """
        if memory is None:
            memory = torch.cat(
                [F_v, F_v_prime, F_t_prime], dim=1
            )  # (B, N_v + 2*N_t, C)

        # 使用 BLIP_Decoder 进行生成
        logits, _, impression_text, loss_lm = self.text_decoder(memory, target_embed)

        return memory, logits, impression_text, loss_lm


class HiMrGn(nn.Module):
    def __init__(
        self,
        args,
        image_encoder,
        history_encoder,
        modality_fusion,
        findings_decoder,
        multi_label_classifier,
        co_attention_module,
        impression_decoder,
        cxr_bert_feature_extractor,
    ):
        super().__init__()
        self.args = args
        self.image_encoder = image_encoder
        self.history_encoder = history_encoder
        self.modality_fusion = modality_fusion
        self.findings_decoder = findings_decoder
        self.co_attention_module = co_attention_module
        self.impression_decoder = impression_decoder
        self.cxr_bert_feature_extractor = cxr_bert_feature_extractor
        self.multi_label_classifier = multi_label_classifier

    def forward(
        self,
        image,
        findings=None,
        impression=None,
        history=None,
        findings_gt=None,
        train_stage=2,
        mode="train",
    ):
        if train_stage == 1:
            F_v = self.image_encoder(image)

            history = self.history_encoder(history)

            fusion_features = self.modality_fusion(F_v, history)

            logits, F_t, findings_text, loss_lm = self.findings_decoder(
                F_v=fusion_features,
                target_embed=findings if mode == "train" else None,
            )

            return {
                "findings_logits": logits,
                "findings_text": findings_text,
                "loss_lm": loss_lm,
                "impression_logits": None,
                "impression_text": None,
                "F_F": None,
                "F_I": None,
                "class_logits": None,
            }

        elif train_stage == 2:
            F_v = self.image_encoder(image)  # (B, C)


            # if mode != "train":
            #     history = self.history_encoder(history)

            #     fusion_features = self.modality_fusion(F_v, history)

            #     findings_logits, F_t, findings_text, findings_loss = self.findings_decoder(
            #         F_v=fusion_features,
            #         target_embed=findings if mode == "train" else None,
            #     )
            # else:
            #     findings_logits = None
            #     F_t = None
            #     findings_text = None
            #     findings_loss = torch.tensor(0.0)

            findings_logits = None
            F_t = None
            findings_text = None
            findings_loss = torch.tensor(0.0)

            # F_F = self.impression_decoder.text_decoder.get_text_embeddings(
            #     findings if mode == "train" else findings_text
            # )

            if self.args.CO:
                F_t_prime, F_v_prime = self.co_attention_module(F_F, F_v)
                memory = torch.cat([F_v, F_t_prime, F_v_prime], dim=1)
                memory, impression_logits, impression_text, impression_loss = (
                    self.impression_decoder(
                        F_t_prime,
                        F_v_prime,
                        F_v,
                        impression if mode == "train" else None,
                        memory=memory,
                    )
                )
            else:
                # memory = torch.cat([F_v, F_t], dim=1)
                memory = F_v
                memory, impression_logits, impression_text, impression_loss = (
                    self.impression_decoder(
                        F_t,
                        F_t,
                        F_v,
                        impression if mode == "train" else None,
                        memory,
                    )
                )

            if self.args.CLS:
                class_logits = self.multi_label_classifier(memory)
            else:
                class_logits = None

            if self.args.CL:
                F_F = self.cxr_bert_feature_extractor(findings_text)
                F_I = self.cxr_bert_feature_extractor(impression_text)
            else:
                F_F = None
                F_I = None

            return {
                "findings_logits": findings_logits,
                "findings_text": findings_text,
                "impression_logits": impression_logits,
                "impression_text": impression_text,
                "loss_lm": findings_loss + impression_loss,
                "findings_loss": findings_loss,
                "impression_loss": impression_loss,
                "F_F": F_F,
                "F_I": F_I,
                "class_logits": class_logits,
            }
        
        elif train_stage == 3:
            F_v = self.image_encoder(image)  # (B, C)

            history = self.history_encoder(history)

            fusion_features = self.modality_fusion(F_v, history)

            findings_logits, F_t, findings_text, findings_loss = self.findings_decoder(
                F_v=fusion_features,
                target_embed=findings if mode == "train" else None,
            )

            F_F = self.impression_decoder.text_decoder.get_text_embeddings(
                findings if mode == "train" else findings_text
            )

            if self.args.CO:
                F_t_prime, F_v_prime = self.co_attention_module(F_F, F_v)
                memory = torch.cat([F_v, F_t_prime, F_v_prime], dim=1)
                memory, impression_logits, impression_text, impression_loss = (
                    self.impression_decoder(
                        F_t_prime,
                        F_v_prime,
                        F_v,
                        impression if mode == "train" else None,
                        memory=memory,
                    )
                )
            else:
                memory = torch.cat([F_v, F_t], dim=1)
                memory, impression_logits, impression_text, impression_loss = (
                    self.impression_decoder(
                        F_t,
                        F_t,
                        F_v,
                        impression if mode == "train" else None,
                        memory,
                    )
                )

            if self.args.CLS:
                class_logits = self.multi_label_classifier(memory)
            else:
                class_logits = None

            if self.args.CL:
                F_F = self.cxr_bert_feature_extractor(findings_text)
                F_I = self.cxr_bert_feature_extractor(impression_text)
            else:
                F_F = None
                F_I = None

            return {
                "findings_logits": findings_logits,
                "findings_text": findings_text,
                "impression_logits": impression_logits,
                "impression_text": impression_text,
                "loss_lm": findings_loss + impression_loss,
                "findings_loss": findings_loss,
                "impression_loss": impression_loss,
                "F_F": F_F,
                "F_I": F_I,
                "class_logits": class_logits,
            }
    
