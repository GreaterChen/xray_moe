import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import pickle
import os
import logging
from models.fast_rcnn_classifier import DetectionOnlyFastRCNN
from models.vit import MedicalVisionTransformer
from models.rgat import ThreeStageRGAT
from utils import analyze_gpu_memory

# 获取logger
model_logger = logging.getLogger("train_logger")


class MedicalReportGenerator(nn.Module):
    def __init__(
        self,
        config,
        object_detector=None,
        image_encoder=None,
        modality_fusion=None,
        findings_decoder=None,
        cxr_bert=None,
    ):
        super(MedicalReportGenerator, self).__init__()

        # 初始化各个组件
        self.object_detector = object_detector
        self.image_encoder = image_encoder
        self.modality_fusion = modality_fusion
        self.findings_decoder = findings_decoder
        self.cxr_bert = cxr_bert
        # 保存参数配置
        self.config = config
        
        # 添加检测结果缓存支持
        self.use_detection_cache = False
        self.detection_cache = {}
        
        # 为区域级别对比学习添加独立的投影层
        self.region_visual_projection = nn.Linear(768, 768)
        self.region_text_projection = nn.Linear(768, 768)
        
        # 在微调阶段初始化RGAT模块（根据配置决定是否启用）
        self.rgat = None
        enable_rgat = getattr(config, 'ENABLE_RGAT', True)
        if config.PHASE == "FINETUNE_BERT" and enable_rgat:
            # 获取图矩阵路径
            aa_adj_path = getattr(config, 'AA_ADJ_PATH', None)
            dd_adj_path = getattr(config, 'DD_ADJ_PATH', None)
            da_adj_path = getattr(config, 'DA_ADJ_PATH', None)
            
            self.rgat = ThreeStageRGAT(
                num_anatomy=29,
                num_disease=14,
                anatomy_dim=768,
                disease_dim=768,
                hidden_dim_1=768,
                hidden_dim_2=768,
                hidden_dim_3=768,
                dropout=getattr(config, 'RGAT_DROPOUT', 0.1),
                aa_adj_path=aa_adj_path,
                dd_adj_path=dd_adj_path,
                da_adj_path=da_adj_path,
            )
            model_logger.info("✅ RGAT模块已在微调阶段初始化")
        elif config.PHASE == "FINETUNE_BERT" and not enable_rgat:
            model_logger.info("ℹ️  RGAT模块已禁用，decoder将直接使用视觉特征")

    def train(self, mode: bool = True):
        """
        自定义train，以便在整体训练模式下保持检测器骨干处于eval，
        仅开放feature projector等可训练分支。
        """
        super().train(mode)
        if self.object_detector is not None and hasattr(self.object_detector, 'detector'):
            self.object_detector.detector.eval()
        return self


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
        use_consistent_eval=False,  # 新增参数：是否在测试时保持训练模式以确保一致性
        anatomical_embeddings_batch=None,  # 新增：批次中每个样本的解剖区域嵌入
        anatomical_nlp_status_batch=None,  # 新增：批次中每个样本的解剖区域NLP状态
        same_text_region_groups_batch=None,  # 新增：批次中每个样本的同文本区域分组
        **kwargs
    ):
        # 在这里实现前向传播逻辑
        if phase == "TRAIN_DETECTION":
            return self.object_detector(image, bbox_targets)
        elif phase == "PRETRAIN_VIT":
            # 第一步：使用目标检测器提取区域特征
            detection_outputs = self.object_detector(
                image,
                bbox_targets,
                current_epoch=current_epoch,
                total_epochs=total_epochs,
            )
            region_features = detection_outputs["region_features"]
            region_detected = detection_outputs["region_detected"]

            # 第二步：通过标准ViT处理区域特征
            visual_features = self.image_encoder(region_features)  # [B, 1+num_regions, hidden_size]

            if mode == "train":
                # 按配置选择对比损失类型
                loss_type = getattr(self.config, 'CONTRASTIVE_LOSS_TYPE', 'region')
                region_itc_loss = None
                clip_itc_loss = None
                simple_region_clip_loss = None
                
                if loss_type == 'region':
                    # 复杂的region对比学习（考虑NLP状态、同文本区域等）
                    if getattr(self.config, 'ENABLE_REGION_ITC', True):
                        region_itc_loss = self.compute_region_itc_loss(
                            visual_features, region_detected, anatomical_embeddings_batch,
                            anatomical_nlp_status_batch, same_text_region_groups_batch, image_ids
                        )
                elif loss_type == 'clip':
                    # 原生CLIP（image-report层面）
                    clip_itc_loss = self.compute_clip_itc_loss(
                        visual_features=visual_features,
                        findings=findings
                    )
                elif loss_type == 'simple_region_clip':
                    # 简化的region CLIP（patch-sentence层面，简单配对定义）
                    simple_region_clip_loss = self.compute_simple_region_clip_loss(
                        visual_features=visual_features,
                        region_detected=region_detected,
                        anatomical_embeddings_batch=anatomical_embeddings_batch
                    )

                # 返回结果
                results = {
                    "region_itc_loss": region_itc_loss,
                    "clip_itc_loss": clip_itc_loss,
                    "simple_region_clip_loss": simple_region_clip_loss,
                    "visual_features": visual_features,
                }
                return results

            # 如果是评估/推理模式
            else:
                # 为测试模式计算对比损失
                with torch.no_grad():
                    loss_type = getattr(self.config, 'CONTRASTIVE_LOSS_TYPE', 'region')
                    region_itc_loss = None
                    clip_itc_loss = None
                    simple_region_clip_loss = None
                    
                    if loss_type == 'region':
                        if getattr(self.config, 'ENABLE_REGION_ITC', True):
                            region_itc_loss = self.compute_region_itc_loss(
                                visual_features, region_detected, anatomical_embeddings_batch,
                                anatomical_nlp_status_batch, same_text_region_groups_batch, image_ids
                            )
                    elif loss_type == 'clip':
                        clip_itc_loss = self.compute_clip_itc_loss(
                            visual_features=visual_features,
                            findings=findings
                        )
                    elif loss_type == 'simple_region_clip':
                        simple_region_clip_loss = self.compute_simple_region_clip_loss(
                            visual_features=visual_features,
                            region_detected=region_detected,
                            anatomical_embeddings_batch=anatomical_embeddings_batch
                        )

                # 返回简化的结果
                return {
                    "region_itc_loss": region_itc_loss,
                    "clip_itc_loss": clip_itc_loss,
                    "simple_region_clip_loss": simple_region_clip_loss,
                    "visual_features": visual_features,
                }

        elif phase == "INFER_BERT":
            # INFER_BERT阶段：使用完整的模型进行推理（和FINETUNE_BERT类似但在推理模式）
            with torch.no_grad():
                # 第一步：使用目标检测器提取区域特征（冻结）
                detection_outputs = self.object_detector(
                    image,
                    bbox_targets,
                    current_epoch=current_epoch,
                    total_epochs=total_epochs,
                )
                region_features = detection_outputs["region_features"]
                region_detected = detection_outputs["region_detected"]

                # 第二步：通过标准ViT处理区域特征（冻结）
                visual_features = self.image_encoder(region_features)  # [B, 1+num_regions, hidden_size]
                
                # 第三步：使用RGAT进行三阶段推理（如果在微调阶段已训练）
                disease_features = None
                disease_preds = None
                
                if self.rgat is not None:
                    # 提取区域特征(去除CLS token)
                    anatomy_features = visual_features[:, 1:, :]  # [B, 29, 768]
                    
                    # RGAT三阶段推理
                    disease_features, disease_preds = self.rgat(anatomy_features)  # [B, 14, 768], [B, 14]
                
                # 第四步：拼接ViT特征和疾病特征
                if disease_features is not None:
                    combined_features = torch.cat([visual_features, disease_features], dim=1)  # [B, 44, 768]
                else:
                    combined_features = visual_features

                # 第五步：通过BERT解码器进行推理
                if mode == "train":
                    # 如果是训练模式（用于构建负样本池等）
                    outputs = self.findings_decoder(
                        visual_features=combined_features,
                        history_encoding=history,
                        findings=findings,
                    )
                    if disease_preds is not None:
                        outputs["disease_preds"] = disease_preds
                    return outputs
                else:
                    # 纯推理模式：只生成文本
                    generated_texts = self.findings_decoder.generate(
                        visual_features=combined_features,
                        history_encoding=history,
                    )
                    results = {"findings_text": generated_texts}
                    if disease_preds is not None:
                        results["disease_preds"] = disease_preds
                    return results

        elif phase == "FINETUNE_BERT":
            # 第一步：使用目标检测器提取区域特征
            detection_outputs = self.object_detector(
                image,
                bbox_targets,
                current_epoch=current_epoch,
                total_epochs=total_epochs,
            )
            region_features = detection_outputs["region_features"]
            region_detected = detection_outputs["region_detected"]

            # 第二步：通过标准ViT处理区域特征（可训练）
            visual_features = self.image_encoder(region_features)  # [B, 1+num_regions, hidden_size]
            
            # 第三步：使用RGAT进行三阶段推理（可训练）
            disease_features = None
            disease_preds = None
            rgat_loss = None
            
            if self.rgat is not None:
                # 提取区域特征(去除CLS token)
                anatomy_features = visual_features[:, 1:, :]  # [B, 29, 768]
                
                # RGAT三阶段推理
                disease_features, disease_preds = self.rgat(anatomy_features)  # [B, 14, 768], [B, 14]
                
                # 计算疾病分类损失(如果有标签)
                if label is not None and mode == "train":
                    rgat_loss = self.rgat.compute_classification_loss(disease_preds, label)
            
            # 第四步：拼接ViT特征和疾病特征
            if disease_features is not None:
                # 将14个疾病特征拼接到visual_features后面
                # visual_features: [B, 30, 768] (1个CLS + 29个区域)
                # disease_features: [B, 14, 768]
                combined_features = torch.cat([visual_features, disease_features], dim=1)  # [B, 44, 768]
            else:
                combined_features = visual_features
            
            # 第五步：通过生成模型进行文本生成（可训练）
            if mode == "train":
                # 训练模式：使用findings计算损失
                outputs = self.findings_decoder(
                    visual_features=combined_features,
                    history_encoding=history,
                    findings=findings,
                )
                
                # 添加RGAT损失到输出
                if rgat_loss is not None:
                    outputs["rgat_loss"] = rgat_loss
                if disease_preds is not None:
                    outputs["disease_preds"] = disease_preds
                
                return outputs
            else:
                # 纯生成模式：不计算损失，只生成文本
                with torch.no_grad():
                    generated_texts = self.findings_decoder.generate(
                        visual_features=combined_features,
                        history_encoding=history,
                    )
                
                results = {"findings_text": generated_texts}
                if disease_preds is not None:
                    results["disease_preds"] = disease_preds
                return results

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

                # 第二步：通过标准ViT处理区域特征
                visual_features = self.image_encoder(region_features)  # [B, 1+num_regions, hidden_size]
                
                # 提取区域特征（去除CLS token）
                region_visual_features = visual_features[:, 1:, :]  # [B, 29, 768]

                return {
                    "region_features": region_visual_features,  # ViT处理后的区域特征
                    "region_detected": region_detected,  # 区域检测掩码
                    "raw_region_features": region_features,  # 检测器原始区域特征
                }


    def compute_region_itc_loss(self, visual_features, region_detected, anatomical_embeddings_batch, 
                                anatomical_nlp_status_batch=None, same_text_region_groups_batch=None, image_ids=None):
        """
        计算区域级别的图像-文本对比损失(ITC) - 内存优化版本，支持NLP状态和同文本区域分组
        
        参数:
            visual_features: ViT输出的视觉特征 [B, 1+num_regions, hidden_size]
            region_detected: 区域检测掩码 [B, num_regions]
            anatomical_embeddings_batch: 批次中每个样本的解剖区域嵌入
            anatomical_nlp_status_batch: 批次中每个样本的解剖区域NLP状态 (normal/abnormal)
            same_text_region_groups_batch: 批次中每个样本的同文本区域分组
            image_ids: 图像ID列表（可选，用于调试）
            
        返回:
            region_itc_loss: 区域级别的对比损失，如果无法计算则返回None
        """
        if not anatomical_embeddings_batch:
            return None
            
        batch_size = visual_features.size(0)
        device = visual_features.device
        
        try:
            # 高效数据收集：避免重复列表操作
            valid_pairs = []
            text_embeds_list = []
            nlp_status_list = []  # 新增：收集NLP状态
            same_text_group_ids = []  # 新增：收集同文本分组ID
            
            # 预计算所有检测mask，减少GPU查询次数
            detected_masks = region_detected > 0.5  # [B, 29]
            
            # 收集有效的视觉-文本对
            for batch_idx in range(batch_size):
                anatomical_embeddings = anatomical_embeddings_batch[batch_idx]
                if not anatomical_embeddings:
                    continue
                
                # 获取该样本的NLP状态字典
                anatomical_nlp_status = {}
                if anatomical_nlp_status_batch and batch_idx < len(anatomical_nlp_status_batch):
                    anatomical_nlp_status = anatomical_nlp_status_batch[batch_idx] or {}
                
                # 获取该样本的同文本区域分组
                same_text_groups = []
                if same_text_region_groups_batch and batch_idx < len(same_text_region_groups_batch):
                    same_text_groups = same_text_region_groups_batch[batch_idx] or []
                
                # 构建区域索引到分组ID的映射
                region_to_group = {}
                for group_id, group in enumerate(same_text_groups):
                    for region_idx in group:
                        region_to_group[region_idx] = group_id
                
                batch_mask = detected_masks[batch_idx]  # [29]
                
                for region_idx, text_embed in anatomical_embeddings.items():
                    if batch_mask[region_idx - 1]:  # 0-based索引
                        valid_pairs.append((batch_idx, region_idx - 1))
                        # 统一转换为torch张量，支持上游提供numpy数组/torch张量
                        text_embeds_list.append(torch.as_tensor(text_embed, dtype=torch.float32))
                        
                        # 新增：收集NLP状态，如果没有则默认为None
                        status = anatomical_nlp_status.get(region_idx, None)
                        nlp_status_list.append(status)
                        
                        # 新增：收集同文本分组ID，格式为 (batch_idx, group_id)
                        group_id = region_to_group.get(region_idx, -1)  # -1表示不在任何分组中
                        same_text_group_ids.append((batch_idx, group_id))
            
            # 检查样本数量
            total_valid = len(valid_pairs)
            if total_valid < 2:
                return None
            
            # 一次性计算所有特征
            return self._compute_region_itc_direct(
                visual_features, valid_pairs, text_embeds_list, nlp_status_list, same_text_group_ids, device
            )
                
        except Exception as e:
            model_logger.warning(f"⚠️  区域ITC损失计算出错: {e}")
            return None

    def _compute_region_itc_direct(self, visual_features, valid_pairs, text_embeds_list, nlp_status_list, same_text_group_ids, device):
        """
        直接计算区域ITC损失，内存优化版本
        
        正负样本定义（基于NLP状态和同文本分组）:
        - 正样本: 
          同一解剖区域 且 相同NLP状态（都是normal或都是abnormal）
        
        - 负样本: 
          情况1: 同一解剖区域 但 不同NLP状态（必然是不同样本，不需要考虑同文本）
          情况2a: 同样本的不同解剖区域 且 非同文本区域
          情况2b: 不同样本的不同解剖区域（全都是负样本）
        
        - 忽略: 
          对角线（自己与自己）
          同一样本中具有相同文本的不同区域（同一分组内的区域）
        """
        N = len(valid_pairs)
        
        # 批量构建索引
        batch_indices = torch.tensor([pair[0] for pair in valid_pairs], device=device)
        region_indices = torch.tensor([pair[1] for pair in valid_pairs], device=device)
        
        # 提取区域视觉特征 - 避免重复切片
        region_visual = visual_features[:, 1:30, :]  # [B, 29, hidden_size]
        visual_feats = region_visual[batch_indices, region_indices]  # [N, hidden_size]
        
        # 数值稳定性检查
        if torch.isnan(visual_feats).any() or torch.isinf(visual_feats).any():
            model_logger.warning("⚠️  区域视觉特征包含NaN或Inf值")
            visual_feats = torch.nan_to_num(visual_feats, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 批量转换文本特征
        text_embeds = torch.stack(text_embeds_list).to(device, non_blocking=True)
        
        # 数值稳定性检查
        if torch.isnan(text_embeds).any() or torch.isinf(text_embeds).any():
            model_logger.warning("⚠️  区域文本特征包含NaN或Inf值")
            text_embeds = torch.nan_to_num(text_embeds, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 投影和归一化 - 合并操作减少内存分配
        eps = 1e-8
        mapped_visual = F.normalize(self.region_visual_projection(visual_feats), p=2, dim=1, eps=eps)
        mapped_text = F.normalize(self.region_text_projection(text_embeds), p=2, dim=1, eps=eps)
        
        # 计算相似度矩阵
        temperature = getattr(self.config, 'REGION_ITC_TEMPERATURE', 0.07)
        logits = torch.matmul(mapped_visual, mapped_text.t()) / temperature  # [N, N]
        
        # 数值稳定性：限制logits范围
        logits = torch.clamp(logits, min=-10.0, max=10.0)
        
        # ========== 构建基于NLP状态和同文本分组的正负样本掩码 ==========
        # 1. 基本掩码：相同/不同解剖区域
        same_region_mask = (region_indices.unsqueeze(1) == region_indices.unsqueeze(0))  # [N, N]
        diff_region_mask = (region_indices.unsqueeze(1) != region_indices.unsqueeze(0))  # [N, N]
        diff_sample_mask = (batch_indices.unsqueeze(1) != batch_indices.unsqueeze(0))  # [N, N]
        
        # 2. 构建NLP状态掩码
        # 将状态列表转换为数值编码：normal=0, abnormal=1, None=-1
        status_codes = []
        for status in nlp_status_list:
            if status == "normal":
                status_codes.append(0)
            elif status == "abnormal":
                status_codes.append(1)
            else:
                status_codes.append(-1)  # 未知状态
        
        status_tensor = torch.tensor(status_codes, device=device)  # [N]
        
        # 3. 相同/不同状态掩码
        same_status_mask = (status_tensor.unsqueeze(1) == status_tensor.unsqueeze(0))  # [N, N]
        diff_status_mask = (status_tensor.unsqueeze(1) != status_tensor.unsqueeze(0))  # [N, N]
        
        # 4. 有效状态掩码（排除None状态的样本）
        valid_status_mask = (status_tensor >= 0)  # [N]
        valid_pair_mask = valid_status_mask.unsqueeze(1) & valid_status_mask.unsqueeze(0)  # [N, N]
        
        # 5. 构建同文本分组掩码（新增）
        # 将 same_text_group_ids 转换为张量 [(batch_idx, group_id), ...]
        group_batch_indices = torch.tensor([gid[0] for gid in same_text_group_ids], device=device)  # [N]
        group_ids = torch.tensor([gid[1] for gid in same_text_group_ids], device=device)  # [N]
        
        # 同一分组掩码：同一样本 且 同一分组ID 且 分组ID有效（>= 0）
        same_sample_mask = (group_batch_indices.unsqueeze(1) == group_batch_indices.unsqueeze(0))  # [N, N]
        same_group_mask = (group_ids.unsqueeze(1) == group_ids.unsqueeze(0))  # [N, N]
        valid_group_mask = (group_ids >= 0)  # [N]
        valid_group_pair_mask = valid_group_mask.unsqueeze(1) & valid_group_mask.unsqueeze(0)  # [N, N]
        
        # 同文本区域掩码：同一样本 且 同一分组 且 分组有效
        same_text_mask = same_sample_mask & same_group_mask & valid_group_pair_mask  # [N, N]
        
        # 6. 正样本掩码：同一解剖区域 且 相同NLP状态
        positive_mask = same_region_mask & same_status_mask & valid_pair_mask  # [N, N]
        
        # 7. 负样本掩码：
        #    情况1: 同一解剖区域 但 不同NLP状态（必然是不同样本，不需要考虑同文本）
        #    情况2a: 同样本的不同解剖区域 且 不是同文本区域
        #    情况2b: 不同样本的不同解剖区域（都是负样本）
        
        # 情况1：同一解剖区域 但 不同NLP状态
        negative_mask_same_region_diff_status = same_region_mask & diff_status_mask & valid_pair_mask
        
        # 情况2：不同解剖区域
        # 2a: 同样本的不同区域 且 不是同文本区域
        negative_mask_same_sample_diff_region = (~diff_sample_mask) & diff_region_mask & (~same_text_mask)
        # 2b: 不同样本的不同区域（都是负样本）
        negative_mask_diff_sample_diff_region = diff_sample_mask & diff_region_mask
        
        # 合并所有负样本
        negative_mask = negative_mask_same_region_diff_status | negative_mask_same_sample_diff_region | negative_mask_diff_sample_diff_region
        
        # 8. 移除对角线（自己不和自己对比）
        eye_mask = torch.eye(N, device=device, dtype=torch.bool)
        positive_mask = positive_mask & (~eye_mask)
        
        # 计算监督对比损失 (Supervised Contrastive Loss)
        loss = self._supervised_contrastive_loss(
            logits, positive_mask, negative_mask, device
        )
        
        # 最终检查
        if torch.isnan(loss) or torch.isinf(loss):
            model_logger.warning("⚠️  区域ITC损失计算出现NaN/Inf，返回零损失")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return loss
    
    def _supervised_contrastive_loss(self, logits, positive_mask, negative_mask, device):
        """
        监督对比学习损失
        
        参数:
            logits: 相似度矩阵 [N, N]
            positive_mask: 正样本掩码 [N, N]
            negative_mask: 负样本掩码 [N, N]
            device: 设备
            
        返回:
            loss: 对比损失标量
        """
        N = logits.size(0)
        
        # 检查每个样本是否有正样本
        num_positives = positive_mask.sum(dim=1)  # [N]
        has_positive = num_positives > 0  # [N]
        
        if not has_positive.any():
            # 如果没有任何样本有正样本，返回零损失
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # 只对有正样本的样本计算损失
        valid_indices = torch.where(has_positive)[0]
        
        # 对每个有效样本计算损失
        losses = []
        for i in valid_indices:
            # 获取第i个样本的正样本和负样本
            pos_mask_i = positive_mask[i]  # [N]
            neg_mask_i = negative_mask[i]  # [N]
            
            num_pos = pos_mask_i.sum()
            num_neg = neg_mask_i.sum()
            
            if num_pos == 0 or num_neg == 0:
                continue
            
            # 提取正样本和负样本的logits
            pos_logits = logits[i][pos_mask_i]  # [num_pos]
            neg_logits = logits[i][neg_mask_i]  # [num_neg]
            
            # 计算InfoNCE损失: -log(sum(exp(pos)) / (sum(exp(pos)) + sum(exp(neg))))
            # 数值稳定版本
            pos_exp = torch.exp(pos_logits)
            neg_exp = torch.exp(neg_logits)
            
            pos_sum = pos_exp.sum()
            neg_sum = neg_exp.sum()
            
            # 避免除零
            denominator = pos_sum + neg_sum + 1e-8
            
            # 损失: -log(pos_sum / denominator)
            loss_i = -torch.log(pos_sum / denominator + 1e-8)
            losses.append(loss_i)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # 平均所有样本的损失
        loss = torch.stack(losses).mean()
        
        return loss

    def compute_simple_region_clip_loss(self, visual_features, region_detected, anatomical_embeddings_batch):
        """
        计算简化的区域级CLIP对比损失（patch-sentence层面，简单配对定义）
        
        正负样本定义（最简单的CLIP方式）：
        - 正样本：配对的region-sentence（同一样本的同一解剖区域）
        - 负样本：batch内所有其他region-sentence对
        
        这是region层面的对比学习，但使用最简单的CLIP配对规则，不考虑NLP状态、同文本区域等复杂情况。
        
        参数:
            visual_features: ViT输出的视觉特征 [B, 1+num_regions, hidden_size]
            region_detected: 区域检测掩码 [B, num_regions]
            anatomical_embeddings_batch: 批次中每个样本的解剖区域嵌入
            
        返回:
            标量对比损失 (对称 InfoNCE, region->text 与 text->region 平均)
        """
        if not anatomical_embeddings_batch:
            return None
            
        batch_size = visual_features.size(0)
        device = visual_features.device
        
        try:
            # 收集所有有效的region-sentence配对
            valid_pairs = []  # [(batch_idx, region_idx), ...]
            text_embeds_list = []
            
            detected_masks = region_detected > 0.5  # [B, 29]
            
            for batch_idx in range(batch_size):
                anatomical_embeddings = anatomical_embeddings_batch[batch_idx]
                if not anatomical_embeddings:
                    continue
                
                batch_mask = detected_masks[batch_idx]  # [29]
                
                for region_idx, text_embed in anatomical_embeddings.items():
                    if batch_mask[region_idx - 1]:  # 0-based索引
                        valid_pairs.append((batch_idx, region_idx - 1))
                        text_embeds_list.append(torch.as_tensor(text_embed, dtype=torch.float32))
            
            # 检查样本数量
            N = len(valid_pairs)
            if N < 2:
                return None
            
            # 提取region视觉特征
            batch_indices = torch.tensor([pair[0] for pair in valid_pairs], device=device)
            region_indices = torch.tensor([pair[1] for pair in valid_pairs], device=device)
            
            region_visual = visual_features[:, 1:30, :]  # [B, 29, hidden_size]
            visual_feats = region_visual[batch_indices, region_indices]  # [N, hidden_size]
            
            # 文本特征
            text_embeds = torch.stack(text_embeds_list).to(device, non_blocking=True)  # [N, hidden_size]
            
            # 数值稳定性检查
            if torch.isnan(visual_feats).any() or torch.isinf(visual_feats).any():
                visual_feats = torch.nan_to_num(visual_feats, nan=0.0, posinf=1.0, neginf=-1.0)
            if torch.isnan(text_embeds).any() or torch.isinf(text_embeds).any():
                text_embeds = torch.nan_to_num(text_embeds, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 投影 + 归一化
            eps = 1e-8
            mapped_visual = F.normalize(self.region_visual_projection(visual_feats), p=2, dim=1, eps=eps)
            mapped_text = F.normalize(self.region_text_projection(text_embeds), p=2, dim=1, eps=eps)
            
            # 计算相似度矩阵
            temperature = getattr(self.config, 'REGION_ITC_TEMPERATURE', 0.07)
            logits_region_to_text = torch.matmul(mapped_visual, mapped_text.t()) / temperature  # [N, N]
            logits_text_to_region = torch.matmul(mapped_text, mapped_visual.t()) / temperature  # [N, N]
            
            # 数值稳定
            logits_region_to_text = torch.clamp(logits_region_to_text, min=-10.0, max=10.0)
            logits_text_to_region = torch.clamp(logits_text_to_region, min=-10.0, max=10.0)
            
            # 构建目标：对角线为正样本（配对的region-sentence）
            # 即：第i个region应该匹配第i个sentence（因为它们来自同一个样本的同一个区域）
            targets = torch.arange(N, device=device)
            
            # 对称的InfoNCE损失
            loss_r2t = F.cross_entropy(logits_region_to_text, targets)
            loss_t2r = F.cross_entropy(logits_text_to_region, targets)
            loss = 0.5 * (loss_r2t + loss_t2r)
            
            if torch.isnan(loss) or torch.isinf(loss):
                model_logger.warning("⚠️  简化区域CLIP损失计算出现NaN/Inf，返回零损失")
                return torch.tensor(0.0, device=device, requires_grad=True)
            
            return loss
                
        except Exception as e:
            model_logger.warning(f"⚠️  简化区域CLIP损失计算出错: {e}")
            return None

    def compute_clip_itc_loss(self, visual_features, findings):
        """
        计算batch内最普通的CLIP式图文对比损失。
        使用CLS全局视觉特征与CXR-BERT的文本CLS特征作为一一对应的正样本，
        其他(batch内)为负样本。复用与region相同的投影层与归一化。

        参数:
            visual_features: ViT输出视觉特征 [B, 1+num_regions, hidden_size]
            findings: 批次的报告文本（tokenized或字符串列表，交给cxr_bert处理）

        返回:
            标量对比损失 (对称 InfoNCE, i->t 与 t->i 平均)
        """
        if self.cxr_bert is None:
            return None

        device = visual_features.device
        batch_size = visual_features.size(0)
        if batch_size < 2:
            return None

        # 全局视觉特征：CLS token
        global_visual = visual_features[:, 0, :]  # [B, hidden_size]

        # 文本特征：CXR-BERT CLS
        text_cls = self.cxr_bert(findings)  # [B, hidden_size]

        # 投影 + 归一化（复用region的映射层）
        eps = 1e-8
        mapped_visual = F.normalize(self.region_visual_projection(global_visual), p=2, dim=1, eps=eps)
        mapped_text = F.normalize(self.region_text_projection(text_cls), p=2, dim=1, eps=eps)

        # 相似度与温度
        temperature = getattr(self.config, 'TEMPERATURE', 0.07)
        logits_per_image = torch.matmul(mapped_visual, mapped_text.t()) / temperature  # [B, B]
        logits_per_text = torch.matmul(mapped_text, mapped_visual.t()) / temperature   # [B, B]

        # 数值稳定
        logits_per_image = torch.clamp(logits_per_image, min=-10.0, max=10.0)
        logits_per_text = torch.clamp(logits_per_text, min=-10.0, max=10.0)

        # 目标：对角为正样本
        targets = torch.arange(batch_size, device=device)
        loss_i2t = F.cross_entropy(logits_per_image, targets)
        loss_t2i = F.cross_entropy(logits_per_text, targets)
        loss = 0.5 * (loss_i2t + loss_t2i)
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=device, requires_grad=True)
        return loss
