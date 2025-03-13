import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig

class RegionClassifier(nn.Module):
    """
    简化版区域分类器: 保持基本功能，移除复杂处理逻辑
    """
    def __init__(self, hidden_size, num_diseases, num_regions=29, dropout_rate=0.3):
        super(RegionClassifier, self).__init__()
        
        # 区域级分类器
        self.region_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_diseases)
        )
        
        # 疾病特异性权重
        self.disease_region_weights = nn.Parameter(torch.ones(num_diseases, num_regions))
        
        # 记录维度
        self.num_diseases = num_diseases
        self.num_regions = num_regions
        
    def forward(self, region_features, region_detected=None):
        """
        简化的前向传播
        
        Args:
            region_features: [batch_size, num_regions, hidden_size]
            region_detected: [batch_size, num_regions] - 未使用，保留参数兼容性
        """
        batch_size = region_features.shape[0]
        
        # 1. 区域级预测 [batch_size, num_regions, num_diseases]
        region_preds = self.region_classifier(region_features)
        
        # 2. 获取区域权重 [num_diseases, num_regions]
        # 使用sigmoid确保权重为正值
        weights = torch.sigmoid(self.disease_region_weights)
        
        # 3. 对每个疾病的权重归一化 [num_diseases, num_regions]
        normalized_weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # 4. 批量扩展权重 [batch_size, num_diseases, num_regions]
        batch_weights = normalized_weights.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 5. 简单直接的图像级预测计算
        # 重塑区域预测 [batch_size, num_regions, num_diseases] -> [batch_size, num_diseases, num_regions]
        region_preds_t = region_preds.transpose(1, 2)
        
        # 批量矩阵乘法 [batch_size, num_diseases, num_regions] @ [batch_size, num_regions, 1]
        # 创建一个全1向量用于求和 [batch_size, num_regions, 1]
        ones = torch.ones(batch_size, region_features.shape[1], 1, device=region_features.device)
        
        # 加权求和: [batch_size, num_diseases, 1]
        image_preds = torch.bmm(region_preds_t * batch_weights, ones).squeeze(-1)
        
        return region_preds, image_preds, batch_weights
    
    def compute_loss(self, region_preds, image_preds, region_labels, image_labels, region_detected=None, attention_weights=None):
        """
        简化的损失计算
        
        Args:
            region_detected: 未使用，保留参数兼容性
            attention_weights: 未使用，保留参数兼容性
        """
        # 处理可能的NaN值（安全措施）
        if torch.isnan(image_preds).any():
            image_preds = torch.nan_to_num(image_preds, nan=0.0)
        
        # 图像级损失
        image_loss = F.binary_cross_entropy_with_logits(image_preds, image_labels)
        
        # 区域级损失（如果提供了区域标签）
        region_loss = 0
        
        # 如果没有提供区域标签，则使用图像标签作为每个区域的标签
        if region_labels is None and image_labels is not None:
            batch_size, num_regions = region_preds.shape[0], region_preds.shape[1]
            # 将图像标签扩展到每个区域
            region_labels = image_labels.unsqueeze(1).expand(-1, num_regions, -1)
        
        # 计算区域级损失（如果有标签）
        if region_labels is not None:
            if torch.isnan(region_preds).any():
                region_preds = torch.nan_to_num(region_preds, nan=0.0)
            
            region_loss = F.binary_cross_entropy_with_logits(region_preds, region_labels)
        
        # 总损失
        region_weight = 0.3 if region_labels is not None else 0.5
        total_loss = image_loss + region_weight * region_loss
        
        return total_loss


class MedicalVisionTransformer(nn.Module):
    """
    基于解剖区域特征的医学视觉Transformer模型
    """
    def __init__(self, pretrained_vit_name="google/vit-base-patch16-224", num_diseases=14, num_regions=29):
        super(MedicalVisionTransformer, self).__init__()
        
        # 加载预训练ViT配置
        self.config = ViTConfig.from_pretrained(pretrained_vit_name)
        self.hidden_size = self.config.hidden_size  # 通常是768 for ViT-B
        
        # 创建可学习的 [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        
        # 使用预训练ViT中的encoder部分
        pretrained_vit = ViTModel.from_pretrained(pretrained_vit_name)
        self.encoder = pretrained_vit.encoder
        
        # 初始化归一化层
        self.layernorm = nn.LayerNorm(self.hidden_size)
        
        # 为每个Transformer层创建分类器
        self.num_layers = self.config.num_hidden_layers
        self.classifiers = nn.ModuleList([
            RegionClassifier(self.hidden_size, num_diseases) 
            for _ in range(self.num_layers)
        ])
        
        # 初始化cls_token
        torch.nn.init.normal_(self.cls_token, std=0.02)
        
        # 保存区域数量
        self.num_regions = num_regions
        
    def forward(self, region_features, region_detected=None, region_labels=None, image_labels=None):
        """
        Args:
            region_features: [batch_size, num_regions=29, hidden_size=768] - 从EnhancedFastRCNN提取的特征
            region_detected: [batch_size, num_regions] - 布尔掩码，表示哪些区域实际被检测到
            region_labels: [batch_size, num_regions, num_diseases] 或 None - 区域级疾病标签
            image_labels: [batch_size, num_diseases] 或 None - 图像级疾病标签
            
        Returns:
            all_hidden_states: 所有层的隐藏状态
            all_region_preds: 每层的区域级预测
            all_image_preds: 每层的图像级预测
            final_cls_output: 最终的CLS token输出，用于下游任务
        """
        batch_size = region_features.shape[0]
        device = region_features.device
        
        # 扩展并添加CLS token到区域特征前面
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, region_features), dim=1)  # [B, 1+num_regions, 768]
        
        # 创建注意力掩码，考虑未检测到的区域
        # 默认所有区域都参与注意力计算
        if region_detected is None:
            attention_mask = torch.ones(batch_size, x.size(1), device=device)
        else:
            # CLS token总是参与注意力计算(值为1)，对于区域特征使用region_detected
            attention_mask = torch.ones(batch_size, 1, device=device)
            region_mask = region_detected.float()  # 将布尔掩码转换为浮点数
            attention_mask = torch.cat([attention_mask, region_mask], dim=1)
        
        # 准备ViT encoder的输入
        extended_attention_mask = self.get_extended_attention_mask(attention_mask)
        
        # 跟踪每层输出
        all_hidden_states = []
        all_region_preds = []
        all_image_preds = []
        
        # 通过Transformer层
        hidden_states = x
        for i, layer_module in enumerate(self.encoder.layer):
            # 通过Transformer层
            layer_outputs = layer_module(hidden_states, extended_attention_mask)
            hidden_states = layer_outputs[0]
            all_hidden_states.append(hidden_states)
            
            # 提取区域特征（排除CLS token）
            region_features_layer = hidden_states[:, 1:, :]
            
            # 通过分类器
            region_preds, image_preds, normalized_weights = self.classifiers[i](region_features_layer, region_detected)
            all_region_preds.append(region_preds)
            all_image_preds.append(image_preds)
        
        # 最终输出
        final_hidden_states = self.layernorm(hidden_states)
        final_cls_output = final_hidden_states[:, 0]  # CLS token输出
        final_region_features = final_hidden_states[:, 1:]  # 区域特征输出
        
        # 计算损失（如果提供了标签）
        loss = None
        if image_labels is not None:
            loss = 0
            for i in range(self.num_layers):
                layer_loss = self.classifiers[i].compute_loss(  # TODO loss 有 nan 问题
                    all_region_preds[i], all_image_preds[i], region_labels, image_labels, region_detected
                )
                layer_loss = self.classifiers[i].compute_loss(
                    all_region_preds[i], all_image_preds[i], region_labels, image_labels, region_detected
                )

                loss += layer_loss
            loss = loss / self.num_layers  # 平均每层的损失
            
        return {
            'loss': loss,
            'hidden_states': all_hidden_states,
            'region_preds': all_region_preds,
            'image_preds': all_image_preds,
            'cls_output': final_cls_output,
            'final_region_features': final_region_features
        }
        
    def get_extended_attention_mask(self, attention_mask):
        """
        将注意力掩码转换为扩展格式
        """
        # 创建扩展的注意力掩码 [batch_size, 1, 1, seq_length]
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        # 将0转换为大的负值，1保持不变
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        return extended_attention_mask


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
    region_labels = torch.randint(0, 2, (batch_size, num_regions, num_diseases)).float()
    
    # 初始化模型
    model = MedicalVisionTransformer(num_diseases=num_diseases)
    
    # 前向传播
    outputs = model(region_features, region_labels, image_labels)
    
    # 输出结果
    print(f"Loss: {outputs['loss']}")
    print(f"CLS output shape: {outputs['cls_output'].shape}")
    print(f"Final region predictions shape: {outputs['region_preds'][-1].shape}")
    print(f"Final image predictions shape: {outputs['image_preds'][-1].shape}")

# 如果直接运行这个文件，可以测试示例用法
if __name__ == "__main__":
    example_usage()