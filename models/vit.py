import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig


class MedicalVisionTransformer(nn.Module):
    """
    标准医学视觉Transformer模型
    用于提取解剖区域的视觉特征,只负责特征编码
    """

    def __init__(
        self,
        pretrained_vit_name="google/vit-base-patch16-224",
        num_regions=29,
    ):
        super(MedicalVisionTransformer, self).__init__()

        # 加载预训练ViT配置
        self.config = ViTConfig.from_pretrained(pretrained_vit_name)
        self.hidden_size = self.config.hidden_size

        # 创建可学习的 [CLS] token（用于对比学习）
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))

        # 使用预训练ViT中的encoder部分
        pretrained_vit = ViTModel.from_pretrained(pretrained_vit_name)
        self.encoder = pretrained_vit.encoder

        # 初始化最终输出的归一化层 - 与原始ViT保持一致
        self.layernorm = nn.LayerNorm(self.hidden_size)

        # 初始化cls_token
        torch.nn.init.normal_(self.cls_token, std=0.02)

        # 保存区域数量
        self.num_regions = num_regions

    def forward(self, region_features):
        """
        标准ViT前向传播
        
        Args:
            region_features: [batch_size, num_regions, hidden_size] - 区域特征
            
        Returns:
            visual_features: [batch_size, 1+num_regions, hidden_size] - 包含CLS token的完整视觉特征
        """
        batch_size = region_features.shape[0]
        
        # 扩展并添加CLS token到区域特征前面
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, region_features), dim=1)  # [B, 1+num_regions, 768]
        
        # 通过Transformer层
        hidden_states = x
        for layer_module in self.encoder.layer:
            layer_outputs = layer_module(hidden_states)
            hidden_states = layer_outputs[0]
        
        # 使用全局LayerNorm处理最终输出
        final_hidden_states = self.layernorm(hidden_states)
        
        return final_hidden_states  # [B, 1+num_regions, hidden_size]