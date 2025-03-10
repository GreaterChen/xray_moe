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
            nn.Linear(2048, feature_dim)
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
    
    def extract_features(self, images, boxes, labels):
        """
        从指定的边界框中提取特征
        """
        batch_size = len(images)
        device = images[0].device
        
        # 初始化区域特征和检测标志
        region_features = torch.zeros((batch_size, self.num_regions, self.feature_projector[-1].out_features), device=device)
        region_detected = torch.zeros((batch_size, self.num_regions), dtype=torch.bool, device=device)
        
        # 提取特征图
        feature_maps = []
        for img in images:
            feature_dict = self.detector.backbone(img.unsqueeze(0))
            # 使用适当的特征层级
            feature = list(feature_dict.values())[3]  # 使用较高级别的特征
            feature_maps.append(feature)
        
        for i in range(batch_size):
            current_boxes = boxes[i]
            current_labels = labels[i]
            
            if len(current_boxes) > 0:
                # 对于每个检测到的区域，提取特征
                feature_map = feature_maps[i]
                
                # 调整boxes以匹配feature map尺寸
                image_size = images[i].shape[-2:]
                feature_size = feature_map.shape[-2:]
                scale_factor = min(feature_size[0] / image_size[0], feature_size[1] / image_size[1])
                scaled_boxes = current_boxes * scale_factor
                
                # 执行RoI Align获取特征
                roi_features = self.roi_align(feature_map, [scaled_boxes])
                
                # 展平特征并投影到指定维度
                flattened_features = roi_features.view(roi_features.size(0), -1)
                projected_features = self.feature_projector(flattened_features)
                
                # 根据标签分配特征到对应区域
                for j, label in enumerate(current_labels):
                    region_idx = label.item() 
                    if 1 <= region_idx <= self.num_regions:  # 确保标签在有效范围内
                        region_features[i, region_idx-1] = projected_features[j]
                        region_detected[i, region_idx-1] = True
            
            # 对于未检测到的区域，使用特殊token
            for j in range(self.num_regions):
                if not region_detected[i, j]:
                    region_features[i, j] = self.missing_region_token
        
        return region_features, region_detected
        
    def forward(self, images, targets=None):
        """
        完整的前向传播过程：检测 + 特征提取
        
        参数:
            images (List[torch.Tensor]): 输入CT图像，形状为 [C, H, W]
            targets (List[Dict], optional): 训练时的目标标注
            
        返回:
            Dict: 包含以下键：
                - 'detections': Faster R-CNN的检测结果
                - 'region_features': 提取的区域特征，形状为 (batch_size, num_regions, feature_dim)
                - 'region_detected': 布尔掩码，表示哪些区域被检测到
        """
        # 使用检测器获取检测结果，但不计算损失（因为检测器已冻结）
        self.detector.eval()
        with torch.no_grad():
            detections = self.detector(images)
        
        batch_size = len(images)
        device = images[0].device
        
        # 准备用于特征提取的框和标签
        boxes_list = []
        labels_list = []
        
        for i in range(batch_size):
            if self.training and targets is not None:
                # 训练模式：使用ground truth边界框
                boxes_list.append(targets[i]['boxes'])
                labels_list.append(targets[i]['labels'])
            else:
                # 推理模式：使用检测到的边界框
                boxes = detections[i]['boxes']
                labels = detections[i]['labels']
                scores = detections[i]['scores']
                
                # 仅保留高置信度的检测结果
                keep = scores > 0.5
                boxes_list.append(boxes[keep])
                labels_list.append(labels[keep])
        
        # 提取区域特征
        region_features, region_detected = self.extract_features(images, boxes_list, labels_list)
        
        return {
            'detections': detections,
            'region_features': region_features,
            'region_detected': region_detected,
            'losses': None  # 第二阶段不计算检测损失
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


# 使用示例
# def training_phase1():
#     """第一阶段训练示例"""
#     # 初始化仅用于检测的模型
#     model = DetectionOnlyFastRCNN(num_regions=29)
    
#     # 设置为训练模式
#     model.train()
    
#     # 训练循环示例
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
#     for epoch in range(10):
#         for images, targets in train_loader:  # 假设有train_loader
#             # 前向传播
#             loss_dict = model(images, targets)
#             losses = sum(loss for loss in loss_dict.values())
            
#             # 反向传播
#             optimizer.zero_grad()
#             losses.backward()
#             optimizer.step()
    
#     # 保存训练好的模型
#     torch.save(model.state_dict(), "detection_model.pth")
#     return model


# def training_phase2(detection_model):
#     """第二阶段训练示例"""
#     # 初始化完整模型，加载预训练好的检测器
#     model = EnhancedFastRCNN(pretrained_detector=detection_model, feature_dim=768)
    
#     # 设置为训练模式，但检测器部分被冻结
#     model.train()
    
#     # 只训练特征提取器部分
#     optimizer = torch.optim.Adam([
#         {'params': model.feature_projector.parameters()},
#         {'params': model.missing_region_token}
#     ], lr=0.001)
    
#     # 训练循环示例
#     for epoch in range(5):
#         for images, targets in train_loader:  # 假设有train_loader
#             # 前向传播
#             results = model(images, targets)
#             region_features = results['region_features']
            
#             # 计算损失（这里是下游任务的损失，如报告生成）
#             downstream_loss = compute_downstream_loss(region_features, targets)
            
#             # 反向传播
#             optimizer.zero_grad()
#             downstream_loss.backward()
#             optimizer.step()
    
#     # 保存完整模型
#     torch.save(model.state_dict(), "enhanced_model.pth")
#     return model


# def inference_example():
#     """推理示例"""
#     # 加载训练好的完整模型
#     model = EnhancedFastRCNN(num_regions=29, feature_dim=768)
#     model.load_state_dict(torch.load("enhanced_model.pth"))
#     model.eval()
    
#     # 创建特征提取器
#     feature_extractor = RegionFeatureExtractor(model)
    
#     # 预处理输入图像
#     images = [preprocess_image("example.jpg")]  # 假设有预处理函数
    
#     # 提取特征
#     region_features, region_detected = feature_extractor(images)
    
#     # 使用提取的特征进行下游任务
#     print(f"提取了 {region_detected[0].sum().item()} 个区域的特征")
#     print(f"特征形状: {region_features.shape}")