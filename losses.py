import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLearningLoss(nn.Module):
    def __init__(self, feature_dim=768, projection_dim=128, hidden_dim=256):
        """
        对比学习损失模块，基于 SimSiam。
        Args:
            feature_dim: 输入特征维度 (F_F 和 F_I)。
            projection_dim: 投影空间维度。
            hidden_dim: 预测 MLP 的隐藏层维度。
        """
        super(ContrastiveLearningLoss, self).__init__()
        # Projection MLP
        self.projection_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )
        # Prediction MLP
        self.prediction_mlp = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, F_F, F_I):
        """
        Args:
            F_F: Findings 的特征，形状 (B, feature_dim)。
            F_I: Impression 的特征，形状 (B, feature_dim)。
        Returns:
            loss_sim: 对比学习损失 (标量)。
        """
        # Step 1: Projection
        Z_F = self.projection_mlp(F_F)  # (B, projection_dim)
        Z_I = self.projection_mlp(F_I)  # (B, projection_dim)

        # Step 2: Prediction
        P_F = self.prediction_mlp(Z_F)  # (B, projection_dim)
        P_I = self.prediction_mlp(Z_I)  # (B, projection_dim)

        # Step 3: Normalize features for cosine similarity
        Z_F = F.normalize(Z_F, dim=-1)
        Z_I = F.normalize(Z_I, dim=-1)
        P_F = F.normalize(P_F, dim=-1)
        P_I = F.normalize(P_I, dim=-1)

        # Step 4: Cosine Similarity Loss
        loss_sim = (
            -F.cosine_similarity(P_F, Z_I.detach(), dim=-1).mean()
            - F.cosine_similarity(P_I, Z_F.detach(), dim=-1).mean()
        ) / 2

        return loss_sim


class MultiLabelBCELoss(nn.Module):
    def __init__(self, num_classes=14, pos_weight=None):
        """
        计算多标签分类的二元交叉熵损失。
        Args:
            num_classes: 类别数量，默认14个疾病类别
            pos_weight: 正样本权重，用于处理类别不平衡，形状 (num_classes,)
        """
        super(MultiLabelBCELoss, self).__init__()
        self.num_classes = num_classes
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")

    def forward(self, predictions, targets):
        """
        Args:
            predictions: 模型输出的类别logits，形状 (B, num_classes)
            targets: 目标标签，形状 (B, num_classes)，每个位置是0或1
        Returns:
            loss: 归一化的二元交叉熵损失 (标量)
        """
        loss = self.criterion(predictions, targets)
        return loss


class CombinedLoss(nn.Module):
    def __init__(
        self,
        pad_id=0,
        feature_dim=768,
        projection_dim=128,
        hidden_dim=256,
        lambda_contrastive=0.1,
        lambda_class=0.1,
    ):
        """
        综合损失函数，包括：
        1. Findings 和 Impression 的交叉熵损失。
        2. Findings 和 Impression 特征的对比学习损失。
        3. 多标签分类损失。
        Args:
            pad_id: 填充值的索引，用于忽略填充标记的交叉熵损失。
            feature_dim: 输入特征维度 (F_F 和 F_I)。
            projection_dim: 投影空间维度，用于对比学习。
            hidden_dim: 预测 MLP 的隐藏层维度。
            lambda_contrastive: 对比损失的权重因子。
            lambda_class: 多标签分类损失的权重因子。
        """
        super(CombinedLoss, self).__init__()
        self.contrastive_loss = ContrastiveLearningLoss(
            feature_dim=feature_dim,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
        )
        self.multi_label_loss = MultiLabelBCELoss()
        self.lambda_contrastive = lambda_contrastive
        self.lambda_class = lambda_class

    def forward(self, output, targets):
        """
        Args:
            output: 模型输出，包含：
                - F_F: Findings 的特征 (B, feature_dim)。
                - F_I: Impression 的特征 (B, feature_dim)。
                - class_logits: 14个疾病类别的概率分布 (B, 14)。
            targets: 目标值，包含：
                - label_gt: 14个疾病类别的标签 (B, 14)。

        Returns:
            total_loss: 综合损失 (标量)。
            losses: 包含各项损失的字典。
        """
        F_F, F_I, class_logits = (
            output["F_F"],
            output["F_I"],
            output["class_logits"],
        )  # 解包 output
        class_logits_gt = targets["label"]  # 解包 targets

        if class_logits is None:
            class_loss = torch.tensor(0.0)
        else:
            # 计算分类损失
            class_loss = self.multi_label_loss(class_logits, class_logits_gt)

        # 计算对比学习损失
        if F_F is not None and F_I is not None:
            contrastive_loss = self.contrastive_loss(F_F, F_I)
        else:
            contrastive_loss = torch.tensor(0.0)

        # 综合损失
        total_loss = (
            self.lambda_contrastive * contrastive_loss + self.lambda_class * class_loss
        )

        # 返回总损失和各项损失
        losses = {
            "contrastive_loss": contrastive_loss,
            "class_loss": class_loss,
            "total_loss": total_loss,
        }
        return total_loss, losses
