import torch
import torch.nn as nn
import torch.nn.functional as F

from models.negativa_sample_pool import NegativeSamplePool

class MOE(nn.Module):
    def __init__(
        self,
        args,
        object_detector=None,
        image_encoder=None,
        history_encoder=None,
        modality_fusion=None,
        findings_decoder=None,
        cxr_bert=None
    ):
        super(MOE, self).__init__()
        
        # 初始化各个组件
        self.object_detector = object_detector
        self.image_encoder = image_encoder
        self.history_encoder = history_encoder
        self.modality_fusion = modality_fusion
        self.findings_decoder = findings_decoder
        self.cxr_bert = cxr_bert
        self.negative_pool = NegativeSamplePool(num_diseases=args.num_diseases if hasattr(args, 'num_diseases') else 14)
        self.negative_pool.load(args.negative_pool_dir)

        self.visual_projection = nn.Linear(768, 768)
        self.text_projection = nn.Linear(768, 768)
        
        # 保存参数配置
        self.args = args

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
    ):
        # 在这里实现前向传播逻辑
        if phase == "TRAIN_DETECTION":
            return self.object_detector(image, bbox_targets)
        elif phase == "PRETRAIN_VIT":
            # 第一步：使用目标检测器提取区域特征
            with torch.no_grad():  # 在阶段2冻结目标检测器   
                detection_outputs = self.object_detector(image, bbox_targets, current_epoch=current_epoch, total_epochs=total_epochs)
                region_features = detection_outputs['region_features']
                region_detected = detection_outputs['region_detected']
            
            # 第二步：通过ViT处理区域特征
            image_encoder_outputs = self.image_encoder(
                region_features, 
                region_detected=region_detected,
                image_labels=label
            )
            
            # 获取ViT的输出
            cls_token = image_encoder_outputs['cls_output']  # [B, hidden_size]
            visual_tokens = image_encoder_outputs['final_region_features']  # [B, 29, hidden_size]
            
            if mode == "train":
                # 使用CXR-BERT编码findings，获取text_cls_token
                with torch.no_grad():  # 冻结CXR-BERT
                    text_cls_token = self.cxr_bert(findings)
                
                # 将文本和视觉特征映射到共享空间
                mapped_visual_cls = self.visual_projection(cls_token)
                mapped_text_cls = self.text_projection(text_cls_token)
                
                # 获取当前批次的疾病标签/预测
                # disease_labels = image_encoder_outputs['image_preds'] if 'image_preds' in image_encoder_outputs else None
                disease_labels = label
                
                # 计算全局对比损失(LTC)
                if disease_labels is not None and self.negative_pool is not None:
                    # 使用negative pool获取困难负样本
                    batch_size = mapped_visual_cls.size(0)
                    neg_samples_per_instance = batch_size - 1
                    
                    # 为每个样本获取对应的负样本
                    negative_samples = self.negative_pool.get_negative_samples_batch(
                        disease_labels,
                        k=neg_samples_per_instance
                    )
                    
                    # 将负样本也映射到共享空间
                    mapped_negative_samples = []
                    for sample_batch in negative_samples:
                        if sample_batch is not None:
                            mapped_negative_samples.append(self.text_projection(sample_batch))
                        else:
                            mapped_negative_samples.append(None)
                    
                    # 使用困难负样本计算对比损失
                    ltc_loss = self.compute_global_ltc_loss(
                        mapped_visual_cls,
                        mapped_text_cls,
                        mapped_negative_samples
                    )
                else:
                    # 如果没有负样本池，使用批内对比
                    ltc_loss = self.compute_batch_ltc_loss(mapped_visual_cls, mapped_text_cls)

                ltc_loss = torch.tensor(0.)
                
                # 返回包含LTC损失的结果
                results = {
                    'cls_token': cls_token,
                    'visual_tokens': visual_tokens,
                    'region_preds': image_encoder_outputs['region_preds'],
                    'image_preds': image_encoder_outputs['image_preds'],
                    'ltc_loss': ltc_loss,
                    'cls_loss': image_encoder_outputs['loss'],
                    'cls_global_loss': image_encoder_outputs['global_loss'],
                    'cls_region_loss': image_encoder_outputs['region_loss']

                }
                return results
            
            # 如果是评估/推理模式
            else:
                return {
                    'cls_token': cls_token,
                    'visual_tokens': visual_tokens,
                    'region_preds': image_encoder_outputs['region_preds'],
                    'image_preds': image_encoder_outputs['image_preds']
                }
        
        elif phase == "INFER_BERT":
            with torch.no_grad():
                # 获取文本的CLS token
                text_cls_token = self.cxr_bert(findings)
                
                # 如果有标签，更新负样本池
                if label is not None:
                    self.negative_pool.update(text_cls_token, label)
                
                # 返回文本特征
                return {'text_cls_token': text_cls_token}

    def compute_global_ltc_loss(self, visual_cls, text_cls, negative_samples):
        """
        计算使用全局负样本池的语言-视觉对比损失(LTC)，完全并行处理
        
        参数:
            visual_cls: 映射后的视觉特征 [B, hidden_size]
            text_cls: 映射后的文本特征 [B, hidden_size]
            negative_samples: 负样本列表，假设每个元素都是相同大小的tensor [K, hidden_size]
                
        返回:
            全局对比损失值
        """
        batch_size = visual_cls.size(0)
        temperature = 0.07  # 温度参数
        device = visual_cls.device
        
        # 归一化特征 (使用torch.nn.functional.normalize更快)
        visual_cls = F.normalize(visual_cls, p=2, dim=1)  # [B, hidden_size]
        text_cls = F.normalize(text_cls, p=2, dim=1)     # [B, hidden_size]
        
        # 检查并构建负样本张量
        if all(ns is not None and ns.size(0) > 0 for ns in negative_samples):
            # 获取负样本数量K
            K = negative_samples[0].size(0)
            
            # 一次性归一化所有负样本 [B, K, hidden_size]
            all_neg_samples = torch.stack([
                F.normalize(ns, p=2, dim=1) for ns in negative_samples
            ], dim=0)
            
            # 计算正样本相似度 [B]
            pos_similarities = torch.sum(visual_cls * text_cls, dim=1)
            
            # 批量计算负样本相似度 [B, K]
            neg_similarities = torch.bmm(
                visual_cls.unsqueeze(1),
                all_neg_samples.transpose(1, 2)
            ).squeeze(1)
            
            # 合并相似度并应用温度缩放 [B, K+1]
            all_similarities = torch.cat([
                pos_similarities.unsqueeze(1), 
                neg_similarities
            ], dim=1) / temperature
            
            # 使用交叉熵计算损失
            labels = torch.zeros(batch_size, dtype=torch.long, device=device)
            loss = F.cross_entropy(all_similarities, labels)
            
            return loss
            
        # 如果有无效的负样本，返回零损失
        return torch.tensor(0.0, device=device, requires_grad=True)

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
        temperature = self.args.temperature
        
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
    
