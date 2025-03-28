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
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_regions + 1
        )  # +1 for background

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
            keep = detection["scores"] > confidence_threshold
            filtered_results.append(
                {
                    "boxes": detection["boxes"][keep],
                    "labels": detection["labels"][keep],
                    "scores": detection["scores"][keep],
                }
            )

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
            detector.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, num_regions + 1
            )
            self.detector = detector

        # 冻结检测器参数
        for param in self.detector.parameters():
            param.requires_grad = False

        # 特征提取层，用于解剖区域特征
        self.roi_align = RoIAlign(
            output_size=(7, 7), spatial_scale=1 / 16, sampling_ratio=2
        )

        # 特征投影层，将ROI特征映射到特定维度
        self.feature_projector = nn.Sequential(
            nn.Linear(256 * 7 * 7, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(2048, feature_dim),
            nn.LayerNorm(feature_dim),  # 添加最终的LayerNorm确保特征空间一致性
        )

        # 特殊token用于未检测到的区域
        self.missing_region_token = nn.Parameter(torch.randn(feature_dim))

        # 保存区域数量
        self.num_regions = num_regions

    def extract_features(self, images, boxes_list, labels_list, scores_list=None):
        """特征提取方法"""

        batch_size = images.size(0)
        device = images.device
        stacked_images = images

        feature_dim = self.feature_projector[-2].out_features

        # 初始化输出tensor
        region_features = torch.zeros(
            (batch_size, self.num_regions, feature_dim), device=device
        )
        region_detected = torch.zeros(
            (batch_size, self.num_regions), dtype=torch.bool, device=device
        )

        # 批量提取特征图
        with torch.no_grad(), torch.amp.autocast("cuda"):
            feature_dict = self.detector.backbone(stacked_images)
            feature_maps = list(feature_dict.values())[2]  # P4特征图

        # 计算图像和特征尺寸比例
        image_shapes = torch.tensor(
            [[images.shape[2], images.shape[3]]] * batch_size, device=device
        )

        feature_shape = torch.tensor(
            [feature_maps.shape[-2], feature_maps.shape[-1]], device=device
        )
        scale_factors = torch.min(feature_shape.float() / image_shapes, dim=1)[
            0
        ].unsqueeze(1)

        # 预分配数据结构
        all_rois = []
        roi_batch_indices = []
        roi_to_region_map = []  # 映射ROI索引到(batch_idx, region_idx)

        # 高效处理每个批次的框
        for batch_idx in range(batch_size):
            current_boxes = boxes_list[batch_idx]
            current_labels = labels_list[batch_idx]
            current_scores = scores_list[batch_idx] if scores_list is not None else None

            if len(current_boxes) == 0:
                continue

            # 找出每个区域的最佳边界框
            if current_scores is not None:
                # 转换标签从1-indexed到0-indexed
                region_indices = current_labels.clamp(1, self.num_regions) - 1

                # 为每个区域找到最高分数的边界框
                region_best_scores = torch.full(
                    (self.num_regions,), -1.0, device=device
                )
                region_best_indices = torch.full(
                    (self.num_regions,), -1, dtype=torch.long, device=device
                )

                # 更新最佳索引
                for j in range(len(current_scores)):
                    region_idx = region_indices[j].item()
                    score = current_scores[j].item()
                    if score > region_best_scores[region_idx]:
                        region_best_scores[region_idx] = score
                        region_best_indices[region_idx] = j
            else:
                # 对于GT框，每个区域最多只有一个框（无重复）
                region_indices = current_labels.clamp(1, self.num_regions) - 1

                # 创建一个区域到框索引的映射，默认为-1（表示该区域无框）
                region_best_indices = torch.full(
                    (self.num_regions,), -1, dtype=torch.long, device=device
                )

                # 框索引数组
                box_indices = torch.arange(len(region_indices), device=device)

                # 使用scatter直接分配（后面的值会覆盖前面的，但我们假设每个区域最多只有一个框）
                if len(region_indices) > 0:  # 确保有框存在
                    region_best_indices.scatter_(0, region_indices, box_indices)

            # 收集所有最佳边界框
            valid_regions = region_best_indices >= 0
            if valid_regions.any():
                region_indices = torch.nonzero(valid_regions, as_tuple=True)[0]
                box_indices = region_best_indices[valid_regions]

                # 标记检测到的区域
                region_detected[batch_idx, region_indices] = True

                # 缩放边界框到特征图尺寸
                scaled_boxes = current_boxes[box_indices] * scale_factors[batch_idx]

                # 添加到批处理列表
                all_rois.append(scaled_boxes)

                # 添加批次索引
                batch_indices = torch.full(
                    (len(scaled_boxes),), batch_idx, dtype=torch.long, device=device
                )
                roi_batch_indices.append(batch_indices)

                # 记录每个ROI对应的区域
                for roi_idx, region_idx in enumerate(region_indices):
                    roi_to_region_map.append((batch_idx, region_idx.item()))

        # 如果有有效的边界框，进行特征提取
        if all_rois:
            # 合并所有ROI和批次索引
            all_rois = torch.cat(all_rois)
            roi_batch_indices = torch.cat(roi_batch_indices)

            # 批量提取ROI特征
            with torch.no_grad(), torch.amp.autocast("cuda"):
                roi_features = torchvision.ops.roi_align(
                    feature_maps,
                    torch.cat([roi_batch_indices.unsqueeze(1), all_rois], dim=1),
                    output_size=(7, 7),
                    spatial_scale=1.0,
                    sampling_ratio=2,
                )

                # 展平特征并批量投影
                flat_features = roi_features.reshape(roi_features.size(0), -1)
                projected_features = self.feature_projector(flat_features)

            # 将特征分配到相应区域
            for i, (batch_idx, region_idx) in enumerate(roi_to_region_map):
                region_features[batch_idx, region_idx] = projected_features[i]

        # 处理未检测区域 - 使用向量化操作
        missing_regions = ~region_detected
        if missing_regions.any():
            region_features[missing_regions] = self.missing_region_token

        return region_features, region_detected

    def forward(
        self,
        images,
        targets=None,
        teacher_forcing_ratio=None,
        current_epoch=0,
        total_epochs=10,
    ):
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
        use_gt = (
            self.training
            and targets is not None
            and (
                teacher_forcing_ratio is not None
                and random.random() < teacher_forcing_ratio
            )
        )

        use_gt = True

        # 准备用于特征提取的框和标签
        boxes_list = []
        labels_list = []
        scores_list = []
        detections = None

        if use_gt:
            # 使用ground truth边界框 - 使用列表推导式
            boxes_list = [target["boxes"] for target in targets]
            labels_list = [target["labels"] for target in targets]
            scores_list = [None] * batch_size  # GT框没有置信度

        else:
            # 使用检测器预测的边界框
            self.detector.eval()
            with torch.no_grad():
                detections = self.detector(images)

            # 设置阈值
            threshold = 0.5

            # 使用列表推导式和向量化操作
            boxes_list = [det["boxes"][det["scores"] > threshold] for det in detections]
            labels_list = [
                det["labels"][det["scores"] > threshold] for det in detections
            ]
            scores_list = [
                det["scores"][det["scores"] > threshold] for det in detections
            ]

        # 提取区域特征，每个解剖区域只保留最佳边界框
        region_features, region_detected = self.extract_features(
            images, boxes_list, labels_list, scores_list
        )

        return {
            "detections": detections if not use_gt else None,
            "region_features": region_features,
            "region_detected": region_detected,
            "losses": None,  # 第二阶段不计算检测损失
            "using_gt": use_gt,  # 返回是否使用了ground truth框
        }
