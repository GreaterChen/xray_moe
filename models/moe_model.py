import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import pickle
import os
from models.fast_rcnn_classifier import DetectionOnlyFastRCNN
from models.vit import MedicalVisionTransformer
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
        # 保存参数配置
        self.config = config
        
        # 添加检测结果缓存支持
        self.use_detection_cache = False
        self.detection_cache = {}
        
        self.visual_projection = nn.Linear(768, 768)
        self.text_projection = nn.Linear(768, 768)
        
        # 为区域级别对比学习添加独立的投影层
        self.region_visual_projection = nn.Linear(768, 768)
        self.region_text_projection = nn.Linear(768, 768)
        
        # 添加Cross-Attention模块用于文本增强（只在支持的阶段中初始化）
        enable_text_enhancement = getattr(config, 'ENABLE_TEXT_ENHANCEMENT', False)
        use_cross_attention = getattr(config, 'TEXT_ENHANCEMENT_USE_CROSS_ATTENTION', True)
        
        if enable_text_enhancement and use_cross_attention:
            supported_phases = getattr(config, 'TEXT_ENHANCEMENT_PHASES', ["FINETUNE_BERT"])
            current_phase = getattr(config, 'PHASE', None)
            
            if current_phase in supported_phases:
                # 从配置中获取Cross-Attention参数
                num_heads = getattr(config, 'TEXT_ENHANCEMENT_CROSS_ATTN_HEADS', 12)
                dropout_rate = getattr(config, 'TEXT_ENHANCEMENT_CROSS_ATTN_DROPOUT', 0.1)
                
                # Cross-Attention模块：文本特征attend to视觉特征
                self.text_to_visual_cross_attn = nn.MultiheadAttention(
                    embed_dim=768, 
                    num_heads=num_heads, 
                    dropout=dropout_rate, 
                    batch_first=True
                )
                
                print(f"✅ Cross-Attention文本增强模块已在 {current_phase} 阶段初始化")
                print(f"   - 注意力头数: {num_heads}")
                print(f"   - Dropout率: {dropout_rate}")
            else:
                print(f"📝 当前阶段 {current_phase} 不在Cross-Attention文本增强支持阶段 {supported_phases} 中")
        elif enable_text_enhancement and not use_cross_attention:
            print("📝 文本增强功能启用，但使用传统拼接方式（非Cross-Attention）")
        else:
            print("📝 Cross-Attention文本增强功能未启用")
        
        # 只在PRETRAIN_VIT阶段加载负样本池
        if config.PHASE == "PRETRAIN_VIT":
            self.negative_pool = NegativeSamplePool(
                num_diseases=config.NUM_DISEASES if hasattr(config, "NUM_DISEASES") else 14
            )
            self.negative_pool.load(config.NEGATIVE_POOL_DIR)
        
        # 初始化文本增强模块（根据配置决定是否启用）
        self.text_enhancer = None
        
        # 检查是否启用文本增强功能
        enable_text_enhancement = getattr(config, 'ENABLE_TEXT_ENHANCEMENT', False)
        
        if enable_text_enhancement:
            # 检查是否在支持的阶段
            supported_phases = getattr(config, 'TEXT_ENHANCEMENT_PHASES', ["FINETUNE_BERT"])
            current_phase = getattr(config, 'PHASE', None)
            
            if current_phase in supported_phases:
                try:
                    # 获取tokenizer（从findings_decoder中获取）
                    tokenizer = None
                    if hasattr(self.findings_decoder, 'tokenizer'):
                        tokenizer = self.findings_decoder.tokenizer
                    elif hasattr(self.findings_decoder, 'decoder') and hasattr(self.findings_decoder.decoder, 'tokenizer'):
                        tokenizer = self.findings_decoder.decoder.tokenizer
                    else:
                        print("⚠️  警告: 无法获取tokenizer，文本增强功能将被禁用")
                    
                    if tokenizer is not None:
                        self.text_enhancer = AnatomicalTextEnhancer(
                            tokenizer=tokenizer,
                            visual_projection=self.region_visual_projection,
                            text_projection=self.region_text_projection,
                            device="cuda" if torch.cuda.is_available() else "cpu",
                            config=config  # 传递配置参数
                        )
                        
                        # 检查是否成功启用
                        if self.text_enhancer.enabled:
                            print(f"✅ 文本增强模块已在 {current_phase} 阶段启用")
                        else:
                            print(f"❌ 文本增强模块启动失败")
                            self.text_enhancer = None
                except Exception as e:
                    print(f"❌ 文本增强模块初始化失败: {e}")
                    self.text_enhancer = None
            else:
                print(f"📝 当前阶段 {current_phase} 不在文本增强支持阶段 {supported_phases} 中")
        else:
            print("📝 文本增强功能未在配置中启用")



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
        **kwargs
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
            # 如果是测试模式且要求一致性评估，临时切换到训练模式
            original_training = self.training
            if mode != "train" and use_consistent_eval:
                self.train()  # 切换到训练模式以保持dropout等行为一致
                print("⚠️  为了一致性比较，在测试时使用训练模式（dropout等保持激活）")
            
            image_encoder_outputs = self.image_encoder(
                region_features, region_detected=region_detected, image_labels=label, use_moe=False
            )

            # 获取ViT的完整输出，在需要时通过索引提取cls_token
            visual_features = image_encoder_outputs["visual_features"]  # [B, 1+num_regions, hidden_size]
            cls_token = visual_features[:, 0]  # 提取cls_token [B, hidden_size]

            # 恢复原始训练状态
            if mode != "train" and use_consistent_eval:
                self.train(original_training)

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
                    # 固定负样本数量，避免batch size影响
                    fixed_neg_samples = 63  # 固定使用63个负样本
                    
                    # 为每个样本获取对应的负样本并映射到共享空间
                    negative_samples = self.negative_pool.get_negative_samples_batch(
                        disease_labels, k=fixed_neg_samples
                    )
                    mapped_negative_samples = self.text_projection(
                        negative_samples
                    )  # [B, K, hidden_size]

                    # 使用困难负样本计算对比损失
                    ltc_loss = self.compute_global_ltc_loss(
                        mapped_visual_cls, mapped_text_cls, mapped_negative_samples
                    )
                    # print(f"[TRAIN] 使用全局负样本池计算LTC loss: {ltc_loss.item():.4f} (neg_samples={fixed_neg_samples})")
                else:
                    # 如果没有负样本池，使用批内对比
                    ltc_loss = self.compute_batch_ltc_loss(
                        mapped_visual_cls, mapped_text_cls
                    )
                    # print(f"[TRAIN] 使用批内对比计算LTC loss: {ltc_loss.item():.4f}")

                # 计算区域级别的ITC损失（新增）
                region_itc_loss = None
                if getattr(self.config, 'ENABLE_REGION_ITC', True):
                    region_itc_loss = self.compute_region_itc_loss(
                        visual_features, region_detected, anatomical_embeddings_batch, image_ids
                    )

                # 返回包含LTC损失和区域ITC损失的结果
                results = {
                    "disease_preds": image_encoder_outputs["disease_preds"],
                    "final_disease_preds": image_encoder_outputs["final_disease_preds"],
                    "ltc_loss": ltc_loss,
                    "region_itc_loss": region_itc_loss,
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

                        # 测试时也使用与训练时相同的loss计算方式
                        disease_labels = label
                        if disease_labels is not None and self.negative_pool is not None:
                            # 使用negative pool获取困难负样本（与训练时相同）
                            batch_size = mapped_visual_cls.size(0)
                            # 固定负样本数量，避免batch size影响
                            fixed_neg_samples = 64  # 固定使用64个负样本
                            
                            # 为每个样本获取对应的负样本并映射到共享空间
                            negative_samples = self.negative_pool.get_negative_samples_batch(
                                disease_labels, k=fixed_neg_samples
                            )
                            mapped_negative_samples = self.text_projection(
                                negative_samples
                            )  # [B, K, hidden_size]

                            # 使用困难负样本计算对比损失
                            ltc_loss = self.compute_global_ltc_loss(
                                mapped_visual_cls, mapped_text_cls, mapped_negative_samples
                            )
                            # print(f"[TEST] 使用全局负样本池计算LTC loss: {ltc_loss.item():.4f} (neg_samples={fixed_neg_samples})")
                        else:
                            # 如果没有负样本池，使用批内对比
                            ltc_loss = self.compute_batch_ltc_loss(
                                mapped_visual_cls, mapped_text_cls
                            )
                            # print(f"[TEST] 使用批内对比计算LTC loss: {ltc_loss.item():.4f}")

                    # 获取分类损失
                    cls_loss = image_encoder_outputs.get("loss", None)
                    
                    # 计算区域级别的ITC损失（测试模式）
                    region_itc_loss = None
                    if getattr(self.config, 'ENABLE_REGION_ITC', True):
                        region_itc_loss = self.compute_region_itc_loss(
                            visual_features, region_detected, anatomical_embeddings_batch, image_ids
                        )

                # 返回简化的结果，只包含需要的字段
                return {
                    "disease_preds": image_encoder_outputs["disease_preds"],
                    "final_disease_preds": image_encoder_outputs["final_disease_preds"],
                    "ltc_loss": ltc_loss,
                    "region_itc_loss": region_itc_loss,
                    "cls_loss": cls_loss,   
                }

        elif phase == "INFER_BERT":
            # INFER_BERT阶段：使用完整的模型进行推理（和FINETUNE_BERT类似但在推理模式）
            with torch.no_grad():
                # 检查是否使用缓存的检测结果
                if self.use_detection_cache and hasattr(self, 'detection_cache'):
                    # 从缓存中获取检测结果
                    batch_size = image.shape[0]
                    device = image.device
                    
                    # 获取当前batch的image_ids
                    if 'image_ids' in kwargs:
                        image_ids = kwargs['image_ids']
                    elif image_ids is not None:
                        pass  # 使用传入的image_ids
                    else:
                        # 如果没有image_ids，回退到正常检测
                        print("⚠️  警告：缓存模式下未提供image_ids，回退到正常检测")
                        detection_outputs = self.object_detector(
                            image, bbox_targets, current_epoch=current_epoch, total_epochs=total_epochs,
                        )
                        region_features = detection_outputs["region_features"]
                        region_detected = detection_outputs["region_detected"]
                    
                    if image_ids is not None:
                        # 从缓存构建bbox targets（作为"ground truth"）
                        cached_targets = []
                        
                        for i, img_id in enumerate(image_ids):
                            if img_id in self.detection_cache:
                                cache_data = self.detection_cache[img_id]
                                # 构建目标格式，使用缓存的bbox作为"ground truth"
                                target = {
                                    "boxes": torch.tensor(cache_data["boxes"], device=device, dtype=torch.float32),
                                    "labels": torch.tensor(cache_data["labels"], device=device, dtype=torch.long),
                                    "image_id": torch.tensor(i, device=device),
                                    "area": torch.tensor([0.0] * len(cache_data["boxes"]), device=device),  # 占位符
                                    "iscrowd": torch.tensor([0] * len(cache_data["boxes"]), device=device, dtype=torch.long)
                                }
                                cached_targets.append(target)
                            else:
                                print(f"⚠️  警告：图像 {img_id} 不在检测缓存中")
                                # 回退到实时检测
                                single_detection = self.object_detector.predict_regions(image[i:i+1])
                                target = {
                                    "boxes": single_detection[0]["boxes"],
                                    "labels": single_detection[0]["labels"], 
                                    "image_id": torch.tensor(i, device=device),
                                    "area": torch.tensor([0.0] * len(single_detection[0]["boxes"]), device=device),
                                    "iscrowd": torch.tensor([0] * len(single_detection[0]["boxes"]), device=device, dtype=torch.long)
                                }
                                cached_targets.append(target)
                        
                        # 使用缓存的bbox作为"ground truth"提取特征（相当于use_gt=True）
                        detection_outputs = self.object_detector(
                            image,
                            cached_targets,  # 使用缓存的bbox作为targets
                            current_epoch=current_epoch,
                            total_epochs=total_epochs,
                        )
                        region_features = detection_outputs["region_features"]
                        region_detected = detection_outputs["region_detected"]
                        
                        print(f"✅ 使用 {len(image_ids)} 个样本的缓存bbox提取特征")
                else:
                    # 第一步：使用目标检测器提取区域特征（冻结）
                    detection_outputs = self.object_detector(
                        image,
                        bbox_targets,
                        current_epoch=current_epoch,
                        total_epochs=total_epochs,
                    )
                    region_features = detection_outputs["region_features"]
                    region_detected = detection_outputs["region_detected"]

                # 第二步：通过ViT处理区域特征（冻结）
                image_encoder_outputs = self.image_encoder(
                    region_features, 
                    region_detected=region_detected, 
                    image_labels=label,
                    phase=phase,  # 传递phase参数给ViT
                    use_moe=True
                )
 
                # 直接使用ViT输出的完整视觉特征（已包含cls_token和region特征）
                visual_features = image_encoder_outputs["visual_features"]  # [B, 1+num_regions, hidden_size]

                # 第三步：通过BERT解码器进行推理
                if mode == "train":
                    # 如果是训练模式（用于构建负样本池等）
                    outputs = self.findings_decoder(
                        visual_features=visual_features,
                        history_encoding=history,
                        findings=findings,
                        use_history=True
                    )
                    return outputs
                else:
                    # 纯推理模式：只生成文本
                    generated_texts = self.findings_decoder.generate(
                        visual_features=visual_features,
                        history_encoding=history,
                        use_history=True
                    )
                    return {"findings_text": generated_texts}

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

            # 第三步：应用基于Cross-Attention的文本增强
            enhanced_visual_features = visual_features  # 默认使用原始视觉特征
            enhanced_history = history  # 默认使用原始历史文本
            
            # 检查是否应该在当前阶段使用新的Cross-Attention文本增强
            should_use_cross_attention = (
                hasattr(self, 'text_to_visual_cross_attn') and
                self.text_enhancer is not None and 
                self.text_enhancer.enabled and
                getattr(self.config, 'ENABLE_TEXT_ENHANCEMENT', False) and
                getattr(self.config, 'TEXT_ENHANCEMENT_USE_CROSS_ATTENTION', True)
            )

            should_use_cross_attention = False
            
            if should_use_cross_attention:
                # 提取区域特征（去除CLS token）用于文本检索
                region_features = visual_features[:, 1:30, :]  # [batch_size, 29, 768]
                
                # 应用Cross-Attention文本增强，不再处理history拼接
                enhanced_visual_features = self.apply_text_enhancement(
                    visual_features=visual_features,
                    region_features=region_features,
                    image_ids=image_ids
                )
                # 使用Cross-Attention时，history保持原样，不进行文本拼接
                enhanced_history = history
            else:
                # 检查是否使用传统的文本拼接增强方式（向后兼容）
                should_use_legacy_enhancement = (
                    self.text_enhancer is not None and 
                    self.text_enhancer.enabled and
                    getattr(self.config, 'ENABLE_TEXT_ENHANCEMENT', False) and
                    not should_use_cross_attention
                )
                
                if should_use_legacy_enhancement:
                    try:
                        # 从配置中获取文本增强参数
                        similarity_threshold = getattr(self.config, 'TEXT_ENHANCEMENT_SIMILARITY_THRESHOLD', 0.5)
                        top_k = getattr(self.config, 'TEXT_ENHANCEMENT_TOP_K', 1)
                        top_sentences = getattr(self.config, 'TEXT_ENHANCEMENT_TOP_SENTENCES', 5)
                        
                        # 提取区域特征（去除CLS token）
                        region_features = visual_features[:, 1:30, :]  # [batch_size, 29, 768]
                        
                        # 检索增强文本（返回(文本, 分数)元组列表）
                        enhanced_texts = self.text_enhancer(
                            visual_features=region_features,
                            query_image_ids=image_ids,
                            similarity_threshold=similarity_threshold,
                            top_k=top_k,
                            top_sentences=top_sentences,
                            return_features=False  # 使用传统的文本返回方式
                        )
                        
                        # 直接在embedding层面增强history（避免解码-编码往返）
                        if enhanced_texts is not None:
                            # 构造包含history的source字典
                            source_dict = {"history": history}
                            
                            # 应用embedding层面的文本增强
                            enhanced_source = self.text_enhancer.create_enhanced_prompt(
                                source=source_dict,
                                enhanced_texts=enhanced_texts,
                                top_sentences=top_sentences
                            )
                            
                            # 提取增强后的history
                            enhanced_history = enhanced_source["history"]
                        else:
                            enhanced_history = history
                        
                    except Exception as e:
                        print(f"❌ 传统文本增强过程中出错: {e}")
                        import traceback
                        traceback.print_exc()
                        # 出错时使用原始历史文本
                        enhanced_history = history

            # 第四步：通过生成模型进行文本生成（可训练）
            if mode == "train":
                # 训练模式：使用findings计算损失
                outputs = self.findings_decoder(
                    visual_features=enhanced_visual_features,  # 使用Cross-Attention增强后的视觉特征
                    history_encoding=enhanced_history,  # 使用增强后的历史文本
                    findings=findings,
                    use_history=True
                )
                
                return outputs
            else:
                # 纯生成模式：不计算损失，只生成文本
                with torch.no_grad():
                    generated_texts = self.findings_decoder.generate(
                        visual_features=enhanced_visual_features,  # 使用Cross-Attention增强后的视觉特征
                        history_encoding=enhanced_history,  # 使用增强后的历史文本
                        use_history=True
                    )
                return {"findings_text": generated_texts}

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

        # 数值稳定性检查
        if torch.isnan(visual_cls).any() or torch.isinf(visual_cls).any():
            print("⚠️  visual_cls包含NaN或Inf值")
            visual_cls = torch.nan_to_num(visual_cls, nan=0.0, posinf=1.0, neginf=-1.0)
            
        if torch.isnan(text_cls).any() or torch.isinf(text_cls).any():
            print("⚠️  text_cls包含NaN或Inf值")
            text_cls = torch.nan_to_num(text_cls, nan=0.0, posinf=1.0, neginf=-1.0)
            
        if torch.isnan(negative_samples).any() or torch.isinf(negative_samples).any():
            print("⚠️  negative_samples包含NaN或Inf值")
            negative_samples = torch.nan_to_num(negative_samples, nan=0.0, posinf=1.0, neginf=-1.0)

        # 归一化特征，添加小的epsilon避免除零
        eps = 1e-8
        visual_cls = F.normalize(visual_cls, p=2, dim=1, eps=eps)  # [B, hidden_size]
        text_cls = F.normalize(text_cls, p=2, dim=1, eps=eps)  # [B, hidden_size]
        negative_samples = F.normalize(
            negative_samples, p=2, dim=2, eps=eps
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
        
        # 数值稳定性：限制相似度范围以防止溢出
        all_similarities = torch.clamp(all_similarities, min=-10.0, max=10.0)

        # 使用交叉熵计算损失
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        loss = F.cross_entropy(all_similarities, labels)
        
        # 最终检查
        if torch.isnan(loss) or torch.isinf(loss):
            print("⚠️  LTC损失计算出现NaN/Inf，返回零损失")
            return torch.tensor(0.0, device=device, requires_grad=True)
            
        return loss

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

        # 数值稳定性检查
        if torch.isnan(visual_cls).any() or torch.isinf(visual_cls).any():
            print("⚠️  visual_cls包含NaN或Inf值")
            visual_cls = torch.nan_to_num(visual_cls, nan=0.0, posinf=1.0, neginf=-1.0)
            
        if torch.isnan(text_cls).any() or torch.isinf(text_cls).any():
            print("⚠️  text_cls包含NaN或Inf值")
            text_cls = torch.nan_to_num(text_cls, nan=0.0, posinf=1.0, neginf=-1.0)

        # 归一化特征，添加小的epsilon避免除零
        eps = 1e-8
        visual_cls = F.normalize(visual_cls, p=2, dim=1, eps=eps)
        text_cls = F.normalize(text_cls, p=2, dim=1, eps=eps)

        # 计算所有视觉-文本对的相似度矩阵
        logits = torch.matmul(visual_cls, text_cls.t()) / temperature  # [B, B]
        
        # 数值稳定性：限制logits范围
        logits = torch.clamp(logits, min=-10.0, max=10.0)

        # 对角线上的元素是正样本对
        labels = torch.arange(batch_size, device=visual_cls.device)

        # 计算视觉->文本方向的损失
        loss_v2t = F.cross_entropy(logits, labels)

        # 计算文本->视觉方向的损失
        loss_t2v = F.cross_entropy(logits.t(), labels)

        # 总损失是两个方向损失的平均
        loss = (loss_v2t + loss_t2v) / 2
        
        # 最终检查
        if torch.isnan(loss) or torch.isinf(loss):
            print("⚠️  批内LTC损失计算出现NaN/Inf，返回零损失")
            return torch.tensor(0.0, device=visual_cls.device, requires_grad=True)

        return loss

    def compute_region_itc_loss(self, visual_features, region_detected, anatomical_embeddings_batch, image_ids=None):
        """
        计算区域级别的图像-文本对比损失(ITC) - 内存优化版本
        
        参数:
            visual_features: ViT输出的视觉特征 [B, 1+num_regions, hidden_size]
            region_detected: 区域检测掩码 [B, num_regions]
            anatomical_embeddings_batch: 批次中每个样本的解剖区域嵌入
            image_ids: 图像ID列表（可选，用于调试）
            
        返回:
            region_itc_loss: 区域级别的对比损失，如果无法计算则返回None
        """
        if not anatomical_embeddings_batch:
            return None
            
        batch_size = visual_features.size(0)
        device = visual_features.device
        
        # 最大样本数量控制
        max_samples = getattr(self.config, 'MAX_REGION_ITC_SAMPLES', 64)
        
        try:
            # 高效数据收集：避免重复列表操作
            valid_pairs = []
            text_embeds_list = []
            
            # 预计算所有检测mask，减少GPU查询次数
            detected_masks = region_detected > 0.5  # [B, 29]
            
            # 收集有效的视觉-文本对
            total_candidates = 0
            for batch_idx in range(batch_size):
                anatomical_embeddings = anatomical_embeddings_batch[batch_idx]
                if not anatomical_embeddings:
                    continue
                
                batch_mask = detected_masks[batch_idx]  # [29]
                
                for region_idx, text_embed in anatomical_embeddings.items():
                    if batch_mask[region_idx - 1]:  # 0-based索引
                        total_candidates += 1
                        # 早期随机采样控制内存
                        if total_candidates > max_samples:
                            if torch.rand(1).item() > (max_samples / total_candidates):
                                continue
                        
                        valid_pairs.append((batch_idx, region_idx - 1))
                        text_embeds_list.append(text_embed)
            
            # 检查样本数量
            total_valid = len(valid_pairs)
            if total_valid < 2:
                return None
            
            # 最终采样确保不超限
            if total_valid > max_samples:
                indices = torch.randperm(total_valid)[:max_samples]
                valid_pairs = [valid_pairs[i] for i in indices]
                text_embeds_list = [text_embeds_list[i] for i in indices]
                total_valid = max_samples
            
            # 一次性计算所有特征
            return self._compute_region_itc_direct(
                visual_features, valid_pairs, text_embeds_list, device
            )
                
        except Exception as e:
            print(f"⚠️  区域ITC损失计算出错: {e}")
            return None

    def _compute_region_itc_direct(self, visual_features, valid_pairs, text_embeds_list, device):
        """直接计算区域ITC损失，内存优化版本"""
        N = len(valid_pairs)
        
        # 批量构建索引
        batch_indices = torch.tensor([pair[0] for pair in valid_pairs], device=device)
        region_indices = torch.tensor([pair[1] for pair in valid_pairs], device=device)
        
        # 提取区域视觉特征 - 避免重复切片
        region_visual = visual_features[:, 1:30, :]  # [B, 29, hidden_size]
        visual_feats = region_visual[batch_indices, region_indices]  # [N, hidden_size]
        
        # 数值稳定性检查
        if torch.isnan(visual_feats).any() or torch.isinf(visual_feats).any():
            print("⚠️  区域视觉特征包含NaN或Inf值")
            visual_feats = torch.nan_to_num(visual_feats, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 批量转换文本特征
        text_embeds = torch.stack(text_embeds_list).to(device, non_blocking=True)
        
        # 数值稳定性检查
        if torch.isnan(text_embeds).any() or torch.isinf(text_embeds).any():
            print("⚠️  区域文本特征包含NaN或Inf值")
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
        
        # 构建标签
        labels = torch.arange(N, device=device, dtype=torch.long)
        
        # 计算双向损失
        loss_v2t = F.cross_entropy(logits, labels)
        loss_t2v = F.cross_entropy(logits.t(), labels)
        
        loss = (loss_v2t + loss_t2v) / 2
        
        # 最终检查
        if torch.isnan(loss) or torch.isinf(loss):
            print("⚠️  区域ITC损失计算出现NaN/Inf，返回零损失")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return loss

    def apply_text_enhancement(self, visual_features, region_features, image_ids):
        """
        使用检索到的文本特征通过cross-attention增强视觉特征
        
        Args:
            visual_features: 原始视觉特征 [B, 30, 768] (包含CLS token)
            region_features: 区域视觉特征 [B, 29, 768] (不包含CLS token)
            image_ids: 图像ID列表，用于文本检索
            
        Returns:
            enhanced_visual_features: 增强后的视觉特征 [B, 30, 768]
        """
        if not hasattr(self, 'text_to_visual_cross_attn') or self.text_enhancer is None:
            # 如果没有cross-attention模块或文本增强器，直接返回原始特征
            return visual_features
        
        try:
            # 从配置中获取文本增强参数
            similarity_threshold = getattr(self.config, 'TEXT_ENHANCEMENT_SIMILARITY_THRESHOLD', 0.5)
            top_k = getattr(self.config, 'TEXT_ENHANCEMENT_TOP_K', 1)
            top_sentences = getattr(self.config, 'TEXT_ENHANCEMENT_TOP_SENTENCES', 5)
            
            # 提取文本特征而不是文本字符串 - 每个区域top1
            text_features, valid_mask = self.text_enhancer(
                visual_features=region_features,
                query_image_ids=image_ids,
                similarity_threshold=similarity_threshold,
                top_k=top_k,
                return_features=True  # 关键：返回特征而不是文本
            )
            
            if text_features is None or not valid_mask.any():
                # 如果没有检索到有效的文本特征，返回原始视觉特征
                return visual_features
            
            batch_size = visual_features.size(0)
            device = visual_features.device
            
            # 提取区域特征（去除CLS token）进行cross-attention
            region_visual_features = visual_features[:, 1:, :]  # [B, 29, 768]
            
            # 检查哪些样本有有效的文本特征
            sample_valid_mask = valid_mask.any(dim=1)  # [B]
            valid_indices = torch.where(sample_valid_mask)[0]
            
            if len(valid_indices) > 0:
                # 提取有效样本的特征
                valid_region_visual = region_visual_features[valid_indices]  # [N_valid, 29, 768]
                valid_text = text_features[valid_indices]  # [N_valid, 29, 768]
                
                # Cross-Attention: 文本特征做query，视觉特征做key&value
                enhanced_region_features, _ = self.text_to_visual_cross_attn(
                    query=valid_text,        # [N_valid, 29, 768]
                    key=valid_region_visual, # [N_valid, 29, 768]
                    value=valid_region_visual, # [N_valid, 29, 768]
                    need_weights=False
                )  # enhanced_region_features: [N_valid, 29, 768]
                
                # 固定权重融合：alpha * enhanced + (1-alpha) * original
                alpha = getattr(self.config, 'TEXT_ENHANCEMENT_FUSION_WEIGHT', 0.3)  # 固定权重
                fused_region_features = alpha * enhanced_region_features + (1 - alpha) * valid_region_visual
                
                # 将增强后的区域特征放回原始tensor中
                enhanced_visual_features = visual_features.clone()
                enhanced_visual_features[valid_indices, 1:, :] = fused_region_features
            else:
                enhanced_visual_features = visual_features
            
            return enhanced_visual_features
            
        except Exception as e:
            print(f"❌ Cross-Attention文本增强过程中出错: {e}")
            import traceback
            traceback.print_exc()
            # 出错时返回原始视觉特征
            return visual_features
