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
        elif phase == "PRETRIAN_VIT":
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
            
            # 如果处于训练模式且有CXR-BERT
            if mode == "train" and self.cxr_bert is not None and findings is not None:
                # 使用CXR-BERT编码findings，获取text_cls_token
                with torch.no_grad():  # 冻结CXR-BERT
                    text_cls_token = self.cxr_bert(findings)
                
                # 计算对比损失(LTC)
                ltc_loss = self.compute_ltc_loss(cls_token, text_cls_token, targets)
                
                # 返回包含LTC损失的结果
                results = {
                    'cls_token': cls_token,
                    'visual_tokens': visual_tokens,
                    'region_preds': image_encoder_outputs['region_preds'],
                    'image_preds': image_encoder_outputs['image_preds'],
                    'ltc_loss': ltc_loss,
                    'cls_loss': image_encoder_outputs['loss']
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

    def compute_ltc_loss(self, visual_cls, text_cls, targets):
        """
        计算语言-视觉对比损失(LTC)
        
        参数:
            visual_cls: 视觉编码器的CLS token [B, hidden_size]
            text_cls: 文本编码器的CLS token [B, hidden_size]
            targets: 包含疾病标签的字典
            
        返回:
            对比损失值
        """
        batch_size = visual_cls.size(0)
        
        # 归一化特征
        visual_cls = F.normalize(visual_cls, p=2, dim=1)
        text_cls = F.normalize(text_cls, p=2, dim=1)
        
        # 计算批内所有样本对之间的相似度
        sim_matrix = torch.matmul(visual_cls, text_cls.t())  # [B, B]
        
        # 对角线是正样本对的相似度，其余为负样本
        labels = torch.arange(batch_size, device=visual_cls.device)
        
        # 通过疾病向量找到最不相似的样本
        if targets and 'labels' in targets:
            disease_vectors = targets['labels']  # [B, 14]
            
            # 计算疾病向量间的相似度，选择最不相似的作为困难负样本
            disease_sim = torch.matmul(disease_vectors, disease_vectors.t())  # [B, B]
            
            # 创建掩码，降低相似疾病样本对的权重
            mask = (disease_sim < 0.3).float() * 2.0  # 对不相似的样本加权
            mask.fill_diagonal_(1.0)  # 对角线保持原始权重
            
            # 应用掩码到相似度矩阵
            sim_matrix = sim_matrix * mask
        
        # 计算对比损失 (InfoNCE)
        loss_v2t = F.cross_entropy(sim_matrix / 0.07, labels)  # 视觉到文本
        loss_t2v = F.cross_entropy(sim_matrix.t() / 0.07, labels)  # 文本到视觉
        
        return (loss_v2t + loss_t2v) / 2.0
    
