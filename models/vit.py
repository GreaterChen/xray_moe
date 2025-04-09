import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig
from configs.region2dis import region_disease_table

# 创建解剖区域疾病关系掩码
def create_anatomy_disease_mask():
    """
    将region2dis中的掩码转换为二元掩码
    1和0 -> 1 (相关)
    -1 -> 0 (不相关)
    """
    mask = torch.tensor(region_disease_table, dtype=torch.float)
    # 将1和0转为1，将-1转为0
    mask = (mask >= 0).float()
    return mask


# 全局变量，避免重复计算
ANATOMY_DISEASE_MASK = create_anatomy_disease_mask()


class DiseaseClassifier(nn.Module):
    """
    基于解剖区域的疾病分类器
    每个疾病使用一个独立的二分类器，输入为与该疾病相关的区域特征
    """

    def __init__(self, hidden_size, num_diseases=14, dropout_rate=0.3):
        super(DiseaseClassifier, self).__init__()

        # 为每种疾病创建一个独立的分类器
        self.disease_classifiers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.LayerNorm(hidden_size // 2),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(hidden_size // 2, 1),
                )
                for _ in range(num_diseases)
            ]
        )

        # 记录维度
        self.num_diseases = num_diseases

        # 获取解剖区域掩码
        self.register_buffer("disease_region_mask", ANATOMY_DISEASE_MASK)

    def forward(self, region_features):
        """
        基于区域特征的疾病分类

        Args:
            region_features: [batch_size, num_regions, hidden_size] - 区域特征

        Returns:
            disease_preds: [batch_size, num_diseases] - 疾病预测结果
        """
        batch_size, num_regions, hidden_size = region_features.shape
        device = region_features.device

        # 存储所有疾病的预测结果
        disease_preds = []

        # 对每种疾病进行独立分类
        for disease_idx in range(self.num_diseases):
            # 获取当前疾病的区域掩码 [num_regions]
            mask = self.disease_region_mask[:, disease_idx]

            # 如果没有相关区域，则预测为0
            if mask.sum() == 0:
                disease_preds.append(torch.zeros(batch_size, 1, device=device))
                continue

            # 选择与当前疾病相关的区域特征
            relevant_indices = torch.where(mask > 0)[0]
            relevant_features = region_features[
                :, relevant_indices, :
            ]  # [batch_size, n_relevant, hidden_size]

            # 对相关区域特征进行平均池化
            pooled_features = relevant_features.mean(dim=1)  # [batch_size, hidden_size]

            # 使用当前疾病的分类器进行预测
            pred = self.disease_classifiers[disease_idx](
                pooled_features
            )  # [batch_size, 1]
            disease_preds.append(pred)

        # 拼接所有疾病的预测结果
        disease_preds = torch.cat(disease_preds, dim=1)  # [batch_size, num_diseases]

        return disease_preds

    def compute_loss(self, disease_preds, image_labels):
        """
        计算损失

        Args:
            disease_preds: [batch_size, num_diseases] - 疾病预测结果
            image_labels: [batch_size, num_diseases] - 图像标签
        """
        # 计算二分类交叉熵损失
        loss = F.binary_cross_entropy_with_logits(disease_preds, image_labels)

        return loss


class MedicalVisionTransformer(nn.Module):
    """
    基于解剖区域特征的医学视觉Transformer模型
    只在偶数层使用分类器进行预测
    """

    def __init__(
        self,
        pretrained_vit_name="google/vit-base-patch16-224",
        num_diseases=14,
        num_regions=29,
    ):
        super(MedicalVisionTransformer, self).__init__()

        # 加载预训练ViT配置
        self.config = ViTConfig.from_pretrained(pretrained_vit_name)
        self.hidden_size = self.config.hidden_size  # 通常是768 for ViT-B

        # 创建可学习的 [CLS] token（用于对比学习）
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))

        # 使用预训练ViT中的encoder部分
        pretrained_vit = ViTModel.from_pretrained(pretrained_vit_name)
        self.encoder = pretrained_vit.encoder

        # 初始化归一化层
        self.layernorm = nn.LayerNorm(self.hidden_size)

        # 只为偶数层创建疾病分类器
        self.num_layers = self.config.num_hidden_layers
        self.classifiers = nn.ModuleList(
            [
                DiseaseClassifier(self.hidden_size, num_diseases)
                if i % 2 == 0
                else None
                for i in range(self.num_layers)
            ]
        )

        # 初始化cls_token
        torch.nn.init.normal_(self.cls_token, std=0.02)

        # 保存区域数量和疾病数量
        self.num_regions = num_regions
        self.num_diseases = num_diseases

        # 获取区域疾病掩码
        self.register_buffer("anatomy_disease_mask", ANATOMY_DISEASE_MASK)

    def forward(
        self,
        region_features,
        region_detected=None,
        image_labels=None,
    ):
        """
        前向传播

        Args:
            region_features: [batch_size, num_regions, hidden_size] - 区域特征
            region_detected: [batch_size, num_regions] - 已检测区域的掩码 (未使用，保留兼容性)
            image_labels: [batch_size, num_diseases] - 图像标签
        """
        batch_size = region_features.shape[0]
        device = region_features.device

        # 扩展并添加CLS token到区域特征前面（用于对比学习）
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, region_features), dim=1)  # [B, 1+num_regions, 768]

        # 跟踪每层输出和预测
        all_hidden_states = []
        all_disease_preds = []

        # 通过Transformer层
        hidden_states = x
        for i, layer_module in enumerate(self.encoder.layer):
            layer_outputs = layer_module(hidden_states)
            hidden_states = layer_outputs[0]
            all_hidden_states.append(hidden_states)

            # 只在偶数层进行分类预测
            if i % 2 == 0 and self.classifiers[i] is not None:
                # 区域特征（不含CLS token）
                region_hidden_states = hidden_states[:, 1:, :]
                # 通过疾病分类器
                disease_preds = self.classifiers[i](region_hidden_states)
                all_disease_preds.append(disease_preds)

        # 最终输出
        final_hidden_states = self.layernorm(hidden_states)
        final_cls_output = final_hidden_states[:, 0]  # CLS token输出，用于对比学习
        final_region_features = final_hidden_states[:, 1:]  # 区域特征输出

        # 计算损失（如果提供了标签）
        loss = None
        if image_labels is not None and len(all_disease_preds) > 0:
            loss = 0
            num_classifier_layers = len(all_disease_preds)  # 实际使用分类器的层数
            for i, disease_preds in enumerate(all_disease_preds):
                layer_idx = i * 2  # 由于只在偶数层使用分类器
                layer_loss = self.classifiers[layer_idx].compute_loss(
                    disease_preds, image_labels
                )
                loss += layer_loss
            loss = loss / num_classifier_layers  # 平均每个分类器的损失

        # 最后一层的疾病预测结果，用于评估
        final_preds = all_disease_preds[-1] if all_disease_preds else None

        return {
            "loss": loss,
            # "hidden_states": all_hidden_states,
            # "disease_preds": all_disease_preds,
            "final_disease_preds": final_preds,
            "cls_output": final_cls_output, 
            "final_region_features": final_region_features,
        }


# 使用示例
def example_usage():
    batch_size = 4
    num_regions = 29
    hidden_size = 768
    num_diseases = 14

    # 模拟来自目标检测的区域特征
    region_features = torch.randn(batch_size, num_regions, hidden_size)

    # 模拟疾病标签
    image_labels = torch.randint(0, 2, (batch_size, num_diseases)).float()

    # 初始化模型
    model = MedicalVisionTransformer(num_diseases=num_diseases)

    # 展示掩码矩阵
    print(f"解剖区域疾病掩码形状: {model.anatomy_disease_mask.shape}")
    print(f"原始区域疾病掩码的一部分:\n{region_disease_table[:3][:5]}")
    print(f"转换后的掩码矩阵的一部分:\n{model.anatomy_disease_mask[:3, :5]}")

    # 前向传播
    outputs = model(region_features, image_labels=image_labels)

    # 输出结果
    print(f"Loss: {outputs['loss']}")
    print(f"最终疾病预测形状: {outputs['final_disease_preds'].shape}")
    print(f"最终区域特征形状: {outputs['final_region_features'].shape}")

    # 验证独立分类器的效果
    disease_classifier = DiseaseClassifier(hidden_size)
    disease_preds = disease_classifier(region_features)
    print(f"独立疾病分类器预测形状: {disease_preds.shape}")
    loss = disease_classifier.compute_loss(disease_preds, image_labels)
    print(f"独立疾病分类器损失: {loss}")


# 如果直接运行这个文件，可以测试示例用法
if __name__ == "__main__":
    example_usage()
