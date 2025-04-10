import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig
from configs.region2dis import region_disease_table
from peft import LoraConfig, get_peft_model
import math

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

class CompleteFeedForwardExpert(nn.Module):
    """
    完整的前馈网络专家模块，直接使用原始模型的FFN，只添加LoRA部分
    """
    def __init__(self, ffn_intermediate, ffn_output, lora_r=8, lora_alpha=16, lora_dropout=0.1):
        super(CompleteFeedForwardExpert, self).__init__()
        
        # 保存原始FFN组件
        self.intermediate = ffn_intermediate  # 上投影+激活函数
        self.output = ffn_output  # 下投影+LayerNorm+残差连接
        
        # 获取维度信息
        dense = self.intermediate.dense
        input_dim = dense.weight.shape[1]  # 输入维度
        hidden_dim = dense.weight.shape[0]  # 隐藏维度
        output_dim = self.output.dense.weight.shape[0]  # 输出维度

        # LoRA dropout
        self.lora_dropout = nn.Dropout(lora_dropout)
        
        # 为上投影创建LoRA权重
        self.lora_A_up = nn.Parameter(torch.zeros(lora_r, input_dim))
        self.lora_B_up = nn.Parameter(torch.zeros(hidden_dim, lora_r))
        
        # 为下投影创建LoRA权重
        self.lora_A_down = nn.Parameter(torch.zeros(lora_r, hidden_dim))
        self.lora_B_down = nn.Parameter(torch.zeros(output_dim, lora_r))
        
        # 缩放因子
        self.scaling = lora_alpha / lora_r
        
        # 初始化LoRA权重
        nn.init.kaiming_uniform_(self.lora_A_up, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_up)
        nn.init.kaiming_uniform_(self.lora_A_down, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_down)
        
    
    def forward(self, x):
        # 获取输入的形状
        orig_shape = x.shape
        batch_size = orig_shape[0]
        
        # 如果是二维输入(batch_size已经被展平)，则调整为三维
        if len(orig_shape) == 2:
            x = x.unsqueeze(0)
        
        # 1. 运行原始intermediate（上投影+激活）
        hidden = self.intermediate(x)
        
        # 2. 处理LoRA部分 - 确保正确处理批次维度
        # 将输入展平为2D，以便于矩阵乘法
        x_2d = x.view(-1, x.size(-1))
        
        # 应用LoRA上投影，但降低缩放因子以控制数值范围
        lora_scaling = self.scaling  # 限制缩放因子大小
        lora_a_out = self.lora_dropout(x_2d) @ self.lora_A_up.T  # [batch_size*seq_len, lora_r]
        lora_up = lora_a_out @ self.lora_B_up.T  # [batch_size*seq_len, hidden_dim]
        lora_up = lora_up.view(hidden.shape) * lora_scaling  # 恢复原始形状，使用较小的缩放因子
        
        # 添加LoRA结果到中间层输出
        hidden = hidden + lora_up
        
        # 3. 处理下投影 (dense层)
        output = self.output.dense(hidden)
        
        # 4. 应用LoRA下投影
        hidden_2d = hidden.view(-1, hidden.size(-1))
        lora_a_down_out = self.lora_dropout(hidden_2d) @ self.lora_A_down.T  # [batch_size*seq_len, lora_r]
        lora_down = lora_a_down_out @ self.lora_B_down.T  # [batch_size*seq_len, output_dim]
        lora_down = lora_down.view(output.shape) * lora_scaling  # 使用较小的缩放因子
        
        # 添加LoRA结果到输出
        output = output + lora_down
        
        # 应用LayerNorm确保输出在合理范围内
        output = F.layer_norm(output, normalized_shape=output.shape[-1:])
        
        # 如果原始输入是展平的，恢复为展平的形状
        if len(orig_shape) == 2:
            output = output.squeeze(0)
            
        return output


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
        
        # 为偶数层创建MOE专家
        self.num_experts = num_diseases + 1  # 14个疾病专家 + 1个通用专家
        self.experts = nn.ModuleDict()
        
        # 只在偶数层创建专家
        for i in range(self.num_layers):
            if i % 2 == 0:
                layer_experts = nn.ModuleList()
                # 为每个疾病创建一个专家 + 一个通用专家
                for j in range(self.num_experts):
                    # 获取当前层的上投影和下投影模块
                    ffn_intermediate = self.encoder.layer[i].intermediate
                    ffn_output = self.encoder.layer[i].output

                    # 创建完整的专家模块（同时包含上投影和下投影）
                    expert = CompleteFeedForwardExpert(
                        ffn_intermediate=ffn_intermediate,
                        ffn_output=ffn_output,
                        lora_r=8,
                        lora_alpha=16,
                        lora_dropout=0.1
                    )
                    layer_experts.append(expert)
                
                if len(layer_experts) > 0:
                    self.experts[f"layer_{i}"] = layer_experts
                else:
                    print(f"警告：第{i}层没有成功创建任何专家")

        # 初始化cls_token
        torch.nn.init.normal_(self.cls_token, std=0.02)

        # 保存区域数量和疾病数量
        self.num_regions = num_regions
        self.num_diseases = num_diseases

        # 获取区域疾病掩码
        self.register_buffer("anatomy_disease_mask", ANATOMY_DISEASE_MASK)
        
        # 当前阶段
        self.current_phase = "PRETRAIN_VIT"

    def forward(
        self,
        region_features,
        region_detected=None,
        image_labels=None,
        phase="PRETRAIN_VIT",
    ):
        """
        前向传播

        Args:
            region_features: [batch_size, num_regions, hidden_size] - 区域特征
            region_detected: [batch_size, num_regions] - 已检测区域的掩码 (未使用，保留兼容性)
            image_labels: [batch_size, num_diseases] - 图像标签
            phase: 当前训练阶段
        """
        batch_size = region_features.shape[0]
        device = region_features.device
        
        # 更新当前阶段
        self.current_phase = phase

        # 扩展并添加CLS token到区域特征前面（用于对比学习）
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, region_features), dim=1)  # [B, 1+num_regions, 768]

        # 跟踪每层输出和预测
        all_hidden_states = []
        all_disease_preds = []

        # 通过Transformer层
        hidden_states = x
        for i, layer_module in enumerate(self.encoder.layer):
            # 在FINETUNE阶段，使用MOE
            if phase in ["FINETUNE_MISTRAL", "FINETUNE_LLAMA"] and i % 2 == 0:
                # 先应用layernorm_before，再使用原始注意力机制
                normalized_hidden_states = layer_module.layernorm_before(hidden_states)
                attention_output = layer_module.attention(normalized_hidden_states)[0]
                
                # 获取区域特征（不含CLS token）
                region_hidden_states = attention_output[:, 1:, :]
                
                # 通过疾病分类器获取路由权重
                with torch.no_grad():
                    disease_preds = self.classifiers[i](region_hidden_states)  # [B, num_diseases]
                    disease_probs = torch.sigmoid(disease_preds)  # [B, num_diseases]
                
                # 路由到专家
                expert_outputs = []
                
                # 处理每个样本
                for b in range(batch_size):
                    sample_hidden = attention_output[b]  # [1+num_regions, hidden_size]
                    sample_output = torch.zeros_like(sample_hidden)  # 初始化输出
                    
                    # 处理cls token (使用通用专家)
                    cls_token_hidden = sample_hidden[0].unsqueeze(0)  # [1, hidden_size]
                    general_expert = self.experts[f"layer_{i}"][-1]  # 通用专家是最后一个
                    # 直接输入和输出处理
                    cls_output = general_expert(cls_token_hidden)
                    sample_output[0] = cls_output.squeeze(0)
                    
                    # 处理每个区域token
                    for r in range(1, sample_hidden.size(0)):  # 跳过cls token
                        region_idx = r - 1  # 实际区域索引
                        
                        # 初始化加权输出
                        weighted_output = torch.zeros_like(sample_hidden[r])
                        
                        # 获取当前区域关联的疾病列表
                        region_disease_mask = self.anatomy_disease_mask[region_idx]  # [num_diseases]
                        
                        # 找出与该区域相关且被诊断为有病(>0.5)的疾病
                        active_disease_indices = []
                        for d in range(self.num_diseases):
                            if region_disease_mask[d] > 0 and disease_probs[b, d] > 0.5:
                                active_disease_indices.append(d)
                        
                        # 如果没有相关疾病或没有疾病被诊断为有病，则使用通用专家
                        if len(active_disease_indices) == 0:
                            general_expert = self.experts[f"layer_{i}"][-1]
                            general_input = sample_hidden[r].unsqueeze(0)  # [1, hidden_size]
                            general_output = general_expert(general_input).squeeze(0)  # [hidden_size]
                            weighted_output = general_output
                        else:
                            # 统计需要处理的专家数量（有病的相关疾病 + 通用专家）
                            num_experts = len(active_disease_indices) + 1  # +1是通用专家
                            
                            # 每个专家的权重相同
                            expert_weight = 1.0 / num_experts
                            
                            # 累加每个激活疾病专家的输出
                            for d_idx in active_disease_indices:
                                disease_expert = self.experts[f"layer_{i}"][d_idx]
                                expert_input = sample_hidden[r].unsqueeze(0)  # [1, hidden_size]
                                expert_output = disease_expert(expert_input).squeeze(0)  # [hidden_size]
                                weighted_output += expert_weight * expert_output
                            
                            # 添加通用专家的输出
                            general_expert = self.experts[f"layer_{i}"][-1]
                            general_input = sample_hidden[r].unsqueeze(0)
                            general_output = general_expert(general_input).squeeze(0)
                            weighted_output += expert_weight * general_output
                        
                        # 更新样本输出
                        sample_output[r] = weighted_output
                    
                    expert_outputs.append(sample_output)
                
                # 组合批次输出
                hidden_states = torch.stack(expert_outputs)
                
                # 使用ViT正确的层归一化
                hidden_states = layer_module.layernorm_after(hidden_states + attention_output)
                
                # 模拟ViT的intermediate层处理(FFN的上半部分)
                # 注意：在我们的MOE中，这部分已经被专家处理，所以这里不需要再应用intermediate
            else:
                # 正常的Transformer前向传播
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
            "disease_preds": final_preds,  # 用于路由的疾病预测
            "final_disease_preds": final_preds,
            "cls_output": final_cls_output, 
            "final_region_features": final_region_features,
        }