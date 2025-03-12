import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig

class RegionClassifier(nn.Module):
    """
    二阶段分类器：先处理region-level分类，再聚合为image-level
    """
    def __init__(self, input_dim, num_diseases=14):
        super(RegionClassifier, self).__init__()
        self.num_diseases = num_diseases
        
        # 区域级分类器
        self.region_classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, num_diseases),
        )
        
    def forward(self, region_features):
        """
        Args:
            region_features: 形状为 [batch_size, num_regions, hidden_dim]
                其中num_regions=29，对应29个解剖区域
                
        Returns:
            region_preds: 每个区域的疾病预测 [batch_size, num_regions, num_diseases]
            image_preds: 整体图像的疾病预测 [batch_size, num_diseases]
        """
        batch_size, num_regions, _ = region_features.shape
        
        # 区域级预测
        region_preds = self.region_classifier(region_features)  # [B, 29, 14]
        
        # 使用平均池化聚合为图像级预测
        image_preds = region_preds.mean(dim=1)  # [B, 14]
        
        return region_preds, image_preds
    
    def compute_loss(self, region_preds, image_preds, region_labels, image_labels):
        """
        计算两阶段分类的损失
        
        Args:
            region_preds: [B, 29, 14]
            image_preds: [B, 14]
            region_labels: [B, 29, 14] 或 None (如果区域标签有争议)
            image_labels: [B, 14]
            
        Returns:
            loss: 总损失
        """
        # 图像级BCE损失
        image_loss = F.binary_cross_entropy_with_logits(image_preds, image_labels)
        
        # 如果有区域标签，计算区域级损失
        if region_labels is not None:
            region_loss = F.binary_cross_entropy_with_logits(
                region_preds.view(-1, self.num_diseases), 
                region_labels.view(-1, self.num_diseases)
            )
            return image_loss + region_loss
        
        return image_loss


class MedicalVisionTransformer(nn.Module):
    """
    基于解剖区域特征的医学视觉Transformer模型
    """
    def __init__(self, pretrained_vit_name="google/vit-base-patch16-224", num_diseases=14):
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
        
    def forward(self, region_features, region_labels=None, image_labels=None):
        """
        Args:
            region_features: [batch_size, num_regions=29, hidden_size=768]
            region_labels: [batch_size, num_regions, num_diseases] 或 None
            image_labels: [batch_size, num_diseases] 或 None
            
        Returns:
            all_hidden_states: 所有层的隐藏状态
            all_region_preds: 每层的区域级预测
            all_image_preds: 每层的图像级预测
            final_cls_output: 最终的CLS token输出，用于下游任务
        """
        batch_size = region_features.shape[0]
        
        # 扩展并添加CLS token到区域特征前面
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, region_features), dim=1)  # [B, 30, 768]
        
        # 准备ViT encoder的输入
        extended_attention_mask = torch.ones(batch_size, x.size(1), device=x.device)
        extended_attention_mask = self.get_extended_attention_mask(extended_attention_mask)
        
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
            region_preds, image_preds = self.classifiers[i](region_features_layer)
            all_region_preds.append(region_preds)
            all_image_preds.append(image_preds)
        
        # 最终输出
        final_hidden_states = self.layernorm(hidden_states)
        final_cls_output = final_hidden_states[:, 0]  # CLS token输出
        
        # 计算损失（如果提供了标签）
        loss = None
        if image_labels is not None:
            loss = 0
            for i in range(self.num_layers):
                layer_loss = self.classifiers[i].compute_loss(
                    all_region_preds[i], all_image_preds[i], region_labels, image_labels
                )
                loss += layer_loss
            loss = loss / self.num_layers  # 平均每层的损失
            
        return {
            'loss': loss,
            'hidden_states': all_hidden_states,
            'region_preds': all_region_preds,
            'image_preds': all_image_preds,
            'cls_output': final_cls_output
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