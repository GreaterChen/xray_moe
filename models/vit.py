import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig

class RegionClassifier(nn.Module):
    """
    分离的全局和区域分类器
    """
    def __init__(self, hidden_size, num_diseases, num_regions=29, dropout_rate=0.3):
        super(RegionClassifier, self).__init__()
        
        # 全局分类器 (用于CLS token)
        self.global_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_diseases)
        )
        
        # 区域分类器
        self.region_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_diseases)
        )
        
        # 记录维度
        self.num_diseases = num_diseases
        self.num_regions = num_regions
        
    def forward(self, hidden_states, region_detected=None):
        """
        分离的前向传播
        
        Args:
            hidden_states: [batch_size, 1+num_regions, hidden_size] - 包含CLS token
            region_detected: [batch_size, num_regions] - 未使用，保留参数兼容性
        """
        batch_size = hidden_states.shape[0]
        
        # 1. 全局预测 (使用CLS token)
        cls_output = hidden_states[:, 0, :]  # [batch_size, hidden_size]
        global_preds = self.global_classifier(cls_output)  # [batch_size, num_diseases]
        
        # 2. 区域预测 (使用其他tokens)
        region_features = hidden_states[:, 1:, :]  # [batch_size, num_regions, hidden_size]
        region_preds = self.region_classifier(region_features)  # [batch_size, num_regions, num_diseases]
        
        return region_preds, global_preds
    
    def compute_loss(self, region_preds, global_preds, region_labels, image_labels, region_detected=None, attention_weights=None):
        """
        分离的损失计算
        
        Args:
            region_detected: 未使用，保留参数兼容性
            attention_weights: 未使用，保留参数兼容性
        """
        # 全局损失
        global_loss = F.binary_cross_entropy_with_logits(global_preds, image_labels)
        
        # 区域级损失
        region_loss = 0
        
        # 如果没有提供区域标签，则使用图像标签作为每个区域的标签
        if region_labels is None and image_labels is not None:
            batch_size, num_regions = region_preds.shape[0], region_preds.shape[1]
            region_labels = image_labels.unsqueeze(1).expand(-1, num_regions, -1)
        
        # 计算区域级损失（如果有标签）
        if region_labels is not None:
            if torch.isnan(region_preds).any():
                region_preds = torch.nan_to_num(region_preds, nan=0.0)
            region_loss = F.binary_cross_entropy_with_logits(region_preds, region_labels)
        
        # 总损失 - 可以调整权重
        total_loss = global_loss + 0.3 * region_loss
        
        return total_loss, global_loss, region_loss


class MedicalVisionTransformer(nn.Module):  
    """
    基于解剖区域特征的医学视觉Transformer模型
    只在偶数层使用分类器进行预测
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
        
        # 只为偶数层创建分类器
        self.num_layers = self.config.num_hidden_layers
        self.classifiers = nn.ModuleList([
            RegionClassifier(self.hidden_size, num_diseases) 
            if i % 2 == 0 else None  # 只在偶数层创建分类器
            for i in range(self.num_layers)
        ])
        
        # 初始化cls_token
        torch.nn.init.normal_(self.cls_token, std=0.02)
        
        # 保存区域数量
        self.num_regions = num_regions
        
    def forward(self, region_features, region_detected=None, region_labels=None, image_labels=None):
        batch_size = region_features.shape[0]
        device = region_features.device
        
        # 扩展并添加CLS token到区域特征前面
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, region_features), dim=1)  # [B, 1+num_regions, 768]
        
        # 跟踪每层输出
        all_hidden_states = []
        all_region_preds = []
        all_image_preds = []
        
        # 通过Transformer层 
        hidden_states = x
        for i, layer_module in enumerate(self.encoder.layer):
            layer_outputs = layer_module(hidden_states)
            hidden_states = layer_outputs[0]
            all_hidden_states.append(hidden_states)
            
            # 只在偶数层进行分类预测
            if i % 2 == 0:
                # 通过分类器
                region_preds, global_preds = self.classifiers[i](hidden_states, region_detected)
                all_region_preds.append(region_preds)
                all_image_preds.append(global_preds)
        
        # 最终输出
        final_hidden_states = self.layernorm(hidden_states)
        final_cls_output = final_hidden_states[:, 0]  # CLS token输出
        final_region_features = final_hidden_states[:, 1:]  # 区域特征输出
        
        # 计算损失（如果提供了标签）
        loss = None
        if image_labels is not None:
            loss = 0
            num_classifier_layers = len(all_region_preds)  # 实际使用分类器的层数
            for i, (region_preds, image_preds) in enumerate(zip(all_region_preds, all_image_preds)):
                layer_loss, global_loss, region_loss = self.classifiers[i*2].compute_loss(  # 注意这里使用i*2因为是偶数层
                    region_preds, image_preds, region_labels, image_labels, region_detected
                )
                loss += layer_loss
            loss = loss / num_classifier_layers  # 平均每个分类器的损失
            
        return {
            'loss': loss,
            'global_loss': global_loss,
            'region_loss': region_loss,
            'hidden_states': all_hidden_states,
            'region_preds': all_region_preds,
            'image_preds': all_image_preds,
            'cls_output': final_cls_output,
            'final_region_features': final_region_features
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