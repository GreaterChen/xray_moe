import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from utils import analyze_gpu_memory

from models.negativa_sample_pool import NegativeSamplePool
from models.text_enhancement import AnatomicalTextEnhancer


class MOE(nn.Module):
    def __init__(
        self,
        config,
        object_detector=None,
        image_encoder=None,
        modality_fusion=None,
        findings_decoder=None,
        cxr_bert=None,
    ):
        super(MOE, self).__init__()

        # 初始化各个组件
        self.object_detector = object_detector
        self.image_encoder = image_encoder
        self.modality_fusion = modality_fusion
        self.findings_decoder = findings_decoder
        self.cxr_bert = cxr_bert
        self.negative_pool = NegativeSamplePool(
            num_diseases=config.NUM_DISEASES if hasattr(config, "NUM_DISEASES") else 14
        )

        if self.image_encoder is not None:
            self.negative_pool.load(config.NEGATIVE_POOL_DIR)

        self.visual_projection = nn.Linear(768, 768)
        self.text_projection = nn.Linear(768, 768)

        # 保存参数配置
        self.config = config
        
        # 初始化文本增强模块（仅在FINETUNE_BERT阶段使用）
        self.text_enhancer = None
        if hasattr(config, 'PHASE') and config.PHASE == "FINETUNE_BERT":
            try:
                # 获取tokenizer（从findings_decoder中获取）
                if hasattr(self.findings_decoder, 'tokenizer'):
                    tokenizer = self.findings_decoder.tokenizer
                elif hasattr(self.findings_decoder, 'decoder') and hasattr(self.findings_decoder.decoder, 'tokenizer'):
                    tokenizer = self.findings_decoder.decoder.tokenizer
                else:
                    print("警告: 无法获取tokenizer，文本增强功能将被禁用")
                    tokenizer = None
                
                if tokenizer is not None:
                    self.text_enhancer = AnatomicalTextEnhancer(
                        tokenizer=tokenizer,
                        visual_projection=self.visual_projection,
                        text_projection=self.text_projection,
                        device="cuda" if torch.cuda.is_available() else "cpu"
                    )
                    print("文本增强模块已初始化，使用共同空间映射")
            except Exception as e:
                print(f"文本增强模块初始化失败: {e}")
                self.text_enhancer = None

    def forward(
        self,
        image,
        bbox_targets=None,
        findings=None,
        history=None,
        targets=None,
        label=None,
        phase="UNK",
        current_epoch=0,
        total_epochs=20,
        mode="train",
        image_ids=None,  # 添加image_ids参数用于文本增强
    ):
        # 在这里实现前向传播逻辑
        if phase == "TRAIN_DETECTION":
            return self.object_detector(image, bbox_targets)
        elif phase == "PRETRAIN_VIT":
            # 第一步：使用目标检测器提取区域特征
            with torch.no_grad():  # 在阶段2冻结目标检测器
                detection_outputs = self.object_detector(
                    image,
                    bbox_targets,
                    current_epoch=current_epoch,
                    total_epochs=total_epochs,
                )
                region_features = detection_outputs["region_features"]
                region_detected = detection_outputs["region_detected"]

            # 第二步：通过ViT处理区域特征
            image_encoder_outputs = self.image_encoder(
                region_features, region_detected=region_detected, image_labels=label, use_moe=False
            )

            # 获取ViT的完整输出，在需要时通过索引提取cls_token
            visual_features = image_encoder_outputs["visual_features"]  # [B, 1+num_regions, hidden_size]
            cls_token = visual_features[:, 0]  # 提取cls_token [B, hidden_size]

            if mode == "train":
                # 使用CXR-BERT编码findings，获取text_cls_token
                with torch.no_grad():  # 冻结CXR-BERT
                    text_cls_token = self.cxr_bert(findings)

                # 将文本和视觉特征映射到共享空间
                mapped_visual_cls = self.visual_projection(cls_token)
                mapped_text_cls = self.text_projection(text_cls_token)

                # 获取当前批次的疾病标签/预测
                disease_labels = label

                # 计算全局对比损失(LTC)
                if disease_labels is not None and self.negative_pool is not None:
                    # 使用negative pool获取困难负样本
                    batch_size = mapped_visual_cls.size(0)
                    neg_samples_per_instance = batch_size - 1

                    # 为每个样本获取对应的负样本并映射到共享空间
                    negative_samples = self.negative_pool.get_negative_samples_batch(
                        disease_labels, k=neg_samples_per_instance
                    )
                    mapped_negative_samples = self.text_projection(
                        negative_samples
                    )  # [B, K, hidden_size]

                    # 使用困难负样本计算对比损失
                    ltc_loss = self.compute_global_ltc_loss(
                        mapped_visual_cls, mapped_text_cls, mapped_negative_samples
                    )
                else:
                    # 如果没有负样本池，使用批内对比
                    ltc_loss = self.compute_batch_ltc_loss(
                        mapped_visual_cls, mapped_text_cls
                    )

                # 返回包含LTC损失的结果
                results = {
                    "disease_preds": image_encoder_outputs["disease_preds"],
                    "final_disease_preds": image_encoder_outputs["final_disease_preds"],
                    "ltc_loss": ltc_loss,
                    "cls_loss": image_encoder_outputs["loss"],
                }
                return results

            # 如果是评估/推理模式
            else:
                # 为测试模式计算ltc_loss和cls_loss
                with torch.no_grad():  # 确保不计算梯度
                    # 使用CXR-BERT编码findings，获取text_cls_token
                    text_cls_token = (
                        self.cxr_bert(findings) if findings is not None else None
                    )

                    # 只有当findings可用时才计算ltc_loss
                    ltc_loss = None
                    if text_cls_token is not None:
                        # 将文本和视觉特征映射到共享空间
                        mapped_visual_cls = self.visual_projection(cls_token)
                        mapped_text_cls = self.text_projection(text_cls_token)

                        # 计算批内对比损失
                        ltc_loss = self.compute_batch_ltc_loss(
                            mapped_visual_cls, mapped_text_cls
                        )

                    # 获取分类损失
                    cls_loss = image_encoder_outputs.get("loss", None)

                # 返回简化的结果，只包含需要的字段
                return {
                    "disease_preds": image_encoder_outputs["disease_preds"],
                    "final_disease_preds": image_encoder_outputs["final_disease_preds"],
                    "ltc_loss": ltc_loss,
                    "cls_loss": cls_loss,
                }

        elif phase == "INFER_BERT":
            with torch.no_grad():
                # 获取文本的CLS token
                text_cls_token = self.cxr_bert(findings)

                # 如果有标签，更新负样本池
                if label is not None:
                    self.negative_pool.update(text_cls_token, label)

                # 返回文本特征
                return {"text_cls_token": text_cls_token}

        elif phase == "FINETUNE_MISTRAL" or phase == "FINETUNE_LLAMA" or phase == "FINETUNE_BERT":
            # 第一步：使用目标检测器提取区域特征（冻结）
            with torch.no_grad():
                detection_outputs = self.object_detector(
                    image,
                    bbox_targets,
                    current_epoch=current_epoch,
                    total_epochs=total_epochs,
                )
                region_features = detection_outputs["region_features"]
                region_detected = detection_outputs["region_detected"]

            # 第二步：通过ViT处理区域特征
            image_encoder_outputs = self.image_encoder(
                region_features, 
                region_detected=region_detected, 
                image_labels=label,
                phase=phase,  # 传递phase参数给ViT
                use_moe=True
            )
 
            # 直接使用ViT输出的完整视觉特征（已包含cls_token和region特征）
            visual_features = image_encoder_outputs["visual_features"]  # [B, 1+num_regions, hidden_size]

            # 第三步：处理历史文本并应用文本增强（仅在FINETUNE_BERT阶段）
            enhanced_history = history  # 默认使用原始历史文本
            similarity_scores = None
            
            if phase == "FINETUNE_BERT" and self.text_enhancer is not None:
                try:
                    # 准备历史文本输入
                    from transformers.tokenization_utils_base import BatchEncoding
                    
                    if isinstance(history, BatchEncoding) and hasattr(history, 'input_ids'):
                        # 如果history是BatchEncoding格式，解码为文本
                        batch_size = history.input_ids.size(0)
                        history_texts = []
                        for i in range(batch_size):
                            hist_ids = history.input_ids[i]
                            hist_text = self.text_enhancer.tokenizer.decode(hist_ids, skip_special_tokens=True)
                            history_texts.append(hist_text)
                    elif isinstance(history, dict) and 'input_ids' in history:
                        # 如果history是普通字典格式
                        batch_size = history['input_ids'].size(0)
                        history_texts = []
                        for i in range(batch_size):
                            hist_ids = history['input_ids'][i]
                            hist_text = self.text_enhancer.tokenizer.decode(hist_ids, skip_special_tokens=True)
                            history_texts.append(hist_text)
                    elif isinstance(history, list):
                        # 如果history是文本列表
                        history_texts = history
                    else:
                        # 其他情况，创建空列表
                        batch_size = visual_features.size(0)
                        history_texts = [""] * batch_size
                    
                    # 应用文本增强（基于视觉特征检索）
                    enhanced_history_texts, similarity_scores = self.text_enhancer(
                        visual_features, 
                        history_texts, 
                        query_image_ids=image_ids,  # 传入image_ids用于排除自身
                        similarity_threshold=0.3,  # 提高阈值，因为视觉到视觉的相似度更可靠
                        top_k=1  # 只取最相似的一个结果
                    )
                    
                    # 将增强的文本重新编码
                    if enhanced_history_texts:
                        enhanced_history = self.text_enhancer.tokenizer(
                            enhanced_history_texts,
                            max_length=getattr(self.config, 'MAX_LEN_HISTORY', 50),
                            padding="max_length",
                            truncation=True,
                            return_tensors="pt",
                        ).to(visual_features.device)
                    
                except Exception as e:
                    print(f"文本增强过程中出错: {e}")
                    # 出错时使用原始历史文本
                    pass

            # 第四步：通过生成模型进行文本生成（可训练）
            if mode == "train":
                # 训练模式：使用findings计算损失
                outputs = self.findings_decoder(
                    visual_features=visual_features,
                    history_encoding=enhanced_history,  # 使用增强后的历史文本
                    findings=findings,
                    use_history=True
                )
                
                # 如果有相似度分数，可以添加到输出中用于分析
                if similarity_scores is not None:
                    outputs.similarity_scores = similarity_scores
                
                return outputs
            else:
                # 纯生成模式：不计算损失，只生成文本
                with torch.no_grad():
                    generated_texts = self.findings_decoder.generate(
                        visual_features=visual_features,
                        history_encoding=enhanced_history,  # 使用增强后的历史文本
                        use_history=True
                    )
                return {"generated_texts": generated_texts}

        elif phase == "BUILD_DATABASE":
            # BUILD_DATABASE阶段：提取解剖区域特征用于构建数据库
            with torch.no_grad():
                # 第一步：使用目标检测器提取区域特征
                detection_outputs = self.object_detector(
                    image,
                    bbox_targets,
                    current_epoch=current_epoch,
                    total_epochs=total_epochs,
                )
                region_features = detection_outputs["region_features"]  # [B, 29, 768]
                region_detected = detection_outputs["region_detected"]  # [B, 29]

                # 第二步：通过ViT处理区域特征
                image_encoder_outputs = self.image_encoder(
                    region_features, 
                    region_detected=region_detected, 
                    image_labels=label,
                    phase="PRETRAIN_VIT",  # 使用PRETRAIN_VIT模式，不启用MOE
                    use_moe=False
                )

                # 获取完整的视觉特征
                visual_features = image_encoder_outputs["visual_features"]  # [B, 1+num_regions, hidden_size]
                
                # 提取区域特征（去除CLS token）
                region_visual_features = visual_features[:, 1:, :]  # [B, 29, 768]

                return {
                    "region_features": region_visual_features,  # ViT处理后的区域特征
                    "region_detected": region_detected,  # 区域检测掩码
                    "raw_region_features": region_features,  # 检测器原始区域特征
                }

    def compute_global_ltc_loss(self, visual_cls, text_cls, negative_samples):
        """
        计算使用全局负样本池的语言-视觉对比损失(LTC)，完全并行处理

        参数:
            visual_cls: 映射后的视觉特征 [B, hidden_size]
            text_cls: 映射后的文本特征 [B, hidden_size]
            negative_samples: 负样本tensor [B, K, hidden_size]

        返回:
            全局对比损失值
        """
        batch_size = visual_cls.size(0)
        temperature = 0.07  # 温度参数
        device = visual_cls.device

        # 归一化特征
        visual_cls = F.normalize(visual_cls, p=2, dim=1)  # [B, hidden_size]
        text_cls = F.normalize(text_cls, p=2, dim=1)  # [B, hidden_size]
        negative_samples = F.normalize(
            negative_samples, p=2, dim=2
        )  # [B, K, hidden_size]

        # 计算正样本相似度 [B]
        pos_similarities = torch.sum(visual_cls * text_cls, dim=1)

        # 批量计算负样本相似度 [B, K]
        neg_similarities = torch.bmm(
            visual_cls.unsqueeze(1),  # [B, 1, hidden_size]
            negative_samples.transpose(1, 2),  # [B, hidden_size, K]
        ).squeeze(
            1
        )  # [B, K]

        # 合并相似度并应用温度缩放 [B, K+1]
        all_similarities = (
            torch.cat([pos_similarities.unsqueeze(1), neg_similarities], dim=1)
            / temperature
        )

        # 使用交叉熵计算损失
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        return F.cross_entropy(all_similarities, labels)

    def compute_batch_ltc_loss(self, visual_cls, text_cls):
        """
        在批次内计算语言-视觉对比损失(LTC)

        参数:
            visual_cls: 映射后的视觉特征 [B, hidden_size]
            text_cls: 映射后的文本特征 [B, hidden_size]

        返回:
            批内对比损失值
        """
        batch_size = visual_cls.size(0)
        temperature = (
            self.config.TEMPERATURE if hasattr(self.config, "TEMPERATURE") else 0.07
        )  # 添加默认值

        # 归一化特征
        visual_cls = F.normalize(visual_cls, p=2, dim=1)
        text_cls = F.normalize(text_cls, p=2, dim=1)

        # 计算所有视觉-文本对的相似度矩阵
        logits = torch.matmul(visual_cls, text_cls.t()) / temperature  # [B, B]

        # 对角线上的元素是正样本对
        labels = torch.arange(batch_size, device=visual_cls.device)

        # 计算视觉->文本方向的损失
        loss_v2t = F.cross_entropy(logits, labels)

        # 计算文本->视觉方向的损失
        loss_t2v = F.cross_entropy(logits.t(), labels)

        # 总损失是两个方向损失的平均
        loss = (loss_v2t + loss_t2v) / 2

        return loss
