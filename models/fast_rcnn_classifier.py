import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import RoIAlign
import numpy as np

class DetectionOnlyFastRCNN(nn.Module):
    """
    第一阶段：仅用于目标检测训练的模型
    不包含特征映射层
    """
    def __init__(self, num_regions=29, pretrained=True):
        super(DetectionOnlyFastRCNN, self).__init__()
        
        # 使用预训练的Faster R-CNN模型
        self.detector = fasterrcnn_resnet50_fpn(pretrained=pretrained)
        
        # 修改检测器以适应医学CT图像
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_regions + 1)  # +1 for background
        
        # 保存区域数量
        self.num_regions = num_regions
        
    def forward(self, images, targets=None):
        """
        仅执行检测任务的前向传播
        """
        if self.training and targets is not None:
            loss_dict = self.detector(images, targets)
            return loss_dict
        else:
            return self.detector(images)
    
    def predict_regions(self, images, confidence_threshold=0.5):
        """
        仅执行解剖区域检测
        
        参数:
            images (List[torch.Tensor]): 输入CT图像
            confidence_threshold (float): 置信度阈值
            
        返回:
            List[Dict]: 每张图像的检测结果，包含boxes, labels, scores
        """
        self.eval()
        with torch.no_grad():
            detections = self.detector(images)
        
        filtered_results = []
        for detection in detections:
            keep = detection['scores'] > confidence_threshold
            filtered_results.append({
                'boxes': detection['boxes'][keep],
                'labels': detection['labels'][keep],
                'scores': detection['scores'][keep]
            })
        
        return filtered_results


class EnhancedFastRCNN(nn.Module):
    """
    第二阶段：带特征提取功能的完整模型
    使用已训练好的检测器，并添加特征映射层
    """
    def __init__(self, pretrained_detector=None, num_regions=29, feature_dim=768):
        super(EnhancedFastRCNN, self).__init__()
        
        # 加载预训练的检测器或创建新的
        if pretrained_detector is not None:
            if isinstance(pretrained_detector, DetectionOnlyFastRCNN):
                self.detector = pretrained_detector.detector
            else:
                self.detector = pretrained_detector
        else:
            # 如果没有提供预训练的检测器，创建一个新的
            detector = fasterrcnn_resnet50_fpn(pretrained=True)
            in_features = detector.roi_heads.box_predictor.cls_score.in_features
            detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_regions + 1)
            self.detector = detector
        
        # 冻结检测器参数
        for param in self.detector.parameters():
            param.requires_grad = False
        
        # 特征提取层，用于解剖区域特征
        self.roi_align = RoIAlign(output_size=(7, 7), spatial_scale=1/16, sampling_ratio=2)
        
        # 特征投影层，将ROI特征映射到特定维度
        self.feature_projector = nn.Sequential(
            nn.Linear(256 * 7 * 7, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(2048, feature_dim),
            nn.LayerNorm(feature_dim)  # 添加最终的LayerNorm确保特征空间一致性
        )
        
        # 特殊token用于未检测到的区域
        self.missing_region_token = nn.Parameter(torch.randn(feature_dim))
        
        # 保存区域数量
        self.num_regions = num_regions
    
    def unfreeze_detector(self, unfreeze_all=False):
        """
        选择性地解冻检测器参数
        
        参数:
            unfreeze_all (bool): 是否解冻所有参数，默认只解冻最后几层
        """
        if unfreeze_all:
            # 解冻所有参数
            for param in self.detector.parameters():
                param.requires_grad = True
        else:
            # 只解冻box predictor部分
            for param in self.detector.roi_heads.box_predictor.parameters():
                param.requires_grad = True
    
    def extract_features(self, images, boxes_list, labels_list, scores_list=None):
        """
        从指定的边界框中提取特征，每个解剖区域只保留一个置信度最高的边界框
        
        参数:
            images (List[torch.Tensor]): 输入CT图像列表
            boxes_list (List[torch.Tensor]): 每个图像对应的边界框列表
            labels_list (List[torch.Tensor]): 每个图像对应的标签列表
            scores_list (List[torch.Tensor], optional): 每个图像对应的置信度列表，用于选择最好的边界框
        
        返回:
            Tuple[torch.Tensor, torch.Tensor]: 
                - 区域特征矩阵 (batch_size, num_regions, feature_dim)
                - 区域检测标志 (batch_size, num_regions)
        """
        batch_size = len(images)
        device = images[0].device
        
        # 初始化区域特征和检测标志，确保29个解剖区域按固定顺序排列
        region_features = torch.zeros((batch_size, self.num_regions, self.feature_projector[-2].out_features), device=device)
        region_detected = torch.zeros((batch_size, self.num_regions), dtype=torch.bool, device=device)
        
        # 提取特征图
        feature_maps = []
        for img in images:
            feature_dict = self.detector.backbone(img.unsqueeze(0))
            # 特征层级与spatial_scale的对应关系
            # P2 -> spatial_scale=1/4, P3 -> spatial_scale=1/8, P4 -> spatial_scale=1/16, P5 -> spatial_scale=1/32
            feature = list(feature_dict.values())[2]  # 使用P4特征图
            feature_maps.append(feature)
        
        for i in range(batch_size):
            current_boxes = boxes_list[i]
            current_labels = labels_list[i]
            current_scores = scores_list[i] if scores_list is not None else None
            
            if len(current_boxes) > 0:
                # 预处理：为每个区域选择最佳边界框
                selected_boxes = []
                selected_labels = []
                region_best_scores = torch.full((self.num_regions,), -1.0, device=device)  # 初始化为负值
                region_best_indices = torch.full((self.num_regions,), -1, dtype=torch.long, device=device)
                
                # 遍历所有检测框，为每个解剖区域找到最佳的边界框
                for j, label in enumerate(current_labels):
                    region_idx = label.item() - 1  # 转为0-indexed
                    
                    if 0 <= region_idx < self.num_regions:  # 确保标签在有效范围内
                        score = current_scores[j].item() if current_scores is not None else 1.0
                        
                        # 如果是GT框且没有scores，或者当前框的置信度高于之前的最佳框
                        if (current_scores is None) or (score > region_best_scores[region_idx]):
                            region_best_scores[region_idx] = score
                            region_best_indices[region_idx] = j
                
                # 收集每个区域的最佳边界框
                for region_idx in range(self.num_regions):
                    best_idx = region_best_indices[region_idx].item()
                    if best_idx >= 0:  # 如果找到了这个区域的边界框
                        selected_boxes.append(current_boxes[best_idx])
                        selected_labels.append(torch.tensor([region_idx + 1], device=device))  # 转回1-indexed
                        region_detected[i, region_idx] = True
                
                # 如果有选中的边界框，则提取特征
                if len(selected_boxes) > 0:
                    selected_boxes = torch.stack(selected_boxes) if len(selected_boxes) > 0 else torch.zeros((0, 4), device=device)
                    selected_labels = torch.cat(selected_labels) if len(selected_labels) > 0 else torch.zeros((0,), dtype=torch.long, device=device)
                    
                    # 调整boxes以匹配feature map尺寸
                    feature_map = feature_maps[i]
                    image_size = images[i].shape[-2:]
                    feature_size = feature_map.shape[-2:]
                    scale_factor = min(feature_size[0] / image_size[0], feature_size[1] / image_size[1])
                    scaled_boxes = selected_boxes * scale_factor
                    
                    # 执行RoI Align获取特征
                    roi_features = self.roi_align(feature_map, [scaled_boxes])
                    
                    # 展平特征并投影到指定维度
                    flattened_features = roi_features.view(roi_features.size(0), -1)
                    projected_features = self.feature_projector(flattened_features)
                    
                    # 根据标签分配特征到对应区域
                    for j, label in enumerate(selected_labels):
                        region_idx = label.item() - 1  # 转为0-indexed
                        region_features[i, region_idx] = projected_features[j]
            
            # 对于未检测到的区域，使用特殊token填充
            for j in range(self.num_regions):
                if not region_detected[i, j]:
                    region_features[i, j] = self.missing_region_token
        
        return region_features, region_detected
    
    def forward(self, images, targets=None, teacher_forcing_ratio=None, current_epoch=0, total_epochs=10):
        """
        完整的前向传播过程：检测 + 特征提取
        
        参数:
            images (List[torch.Tensor]): 输入CT图像，形状为 [C, H, W]
            targets (List[Dict], optional): 训练时的目标标注
            teacher_forcing_ratio (float, optional): 使用ground truth框的概率，None表示自动根据训练进度调整
            current_epoch (int): 当前训练轮次，用于自动计算teacher_forcing_ratio
            total_epochs (int): 总训练轮次，用于自动计算teacher_forcing_ratio
            
        返回:
            Dict: 包含以下键：
                - 'detections': Faster R-CNN的检测结果 (如果使用GT框则为None)
                - 'region_features': 提取的区域特征，形状为 (batch_size, num_regions, feature_dim)
                - 'region_detected': 布尔掩码，表示哪些区域被检测到
                - 'using_gt': 布尔值，表示本次前向传播是否使用了ground truth框
        """
        batch_size = len(images)
        device = images[0].device
        
        # 决定是否使用ground truth边界框
        if teacher_forcing_ratio is None and self.training:
            # 根据训练进度自动调整teacher forcing概率
            # 从1.0线性降低到0.2，确保在训练后期仍有一定概率使用GT框
            teacher_forcing_ratio = max(0.2, 1.0 - 0.8 * current_epoch / total_epochs)
        
        # 是否使用ground truth边界框的条件
        use_gt = self.training and targets is not None and (
            teacher_forcing_ratio is not None and 
            random.random() < teacher_forcing_ratio
        )
        
        # 准备用于特征提取的框和标签
        boxes_list = []
        labels_list = []
        scores_list = []
        detections = None
        
        if use_gt:
            # 使用ground truth边界框
            for i in range(batch_size):
                boxes_list.append(targets[i]['boxes'])
                labels_list.append(targets[i]['labels'])
                scores_list.append(None)  # GT框没有置信度
            
        else:
            # 使用检测器预测的边界框
            self.detector.eval()
            with torch.no_grad():
                detections = self.detector(images)
            
            for i in range(batch_size):
                boxes = detections[i]['boxes']
                labels = detections[i]['labels']
                scores = detections[i]['scores']
                
                # 仅保留高置信度的检测结果
                threshold = 0.5
                keep = scores > threshold
                boxes_list.append(boxes[keep])
                labels_list.append(labels[keep])
                scores_list.append(scores[keep])
        
        # 提取区域特征，每个解剖区域只保留最佳边界框
        region_features, region_detected = self.extract_features(images, boxes_list, labels_list, scores_list)
        
        return {
            'detections': detections,  # 如果使用GT框则为None
            'region_features': region_features,
            'region_detected': region_detected,
            'losses': None,  # 第二阶段不计算检测损失
            'using_gt': use_gt  # 返回是否使用了ground truth框
        }


class RegionFeatureExtractor(nn.Module):
    """
    从训练好的EnhancedFastRCNN中提取特征的独立模块
    """
    def __init__(self, enhanced_rcnn):
        super(RegionFeatureExtractor, self).__init__()
        
        if not isinstance(enhanced_rcnn, EnhancedFastRCNN):
            raise TypeError("Input must be an EnhancedFastRCNN model")
            
        self.model = enhanced_rcnn
        
    def forward(self, images):
        """提取区域特征"""
        with torch.no_grad():
            results = self.model(images)
        return results['region_features'], results['region_detected']