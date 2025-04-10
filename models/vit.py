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
    完整的前馈网络专家模块，优化批处理效率
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
    
    def compute_lora_output(self, x, base_output):
        """
        计算LoRA部分的输出
        x: [batch_size, seq_len, input_dim] 原始输入
        base_output: [batch_size, seq_len, output_dim] FFN的基础输出
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. 计算上投影LoRA
        x_2d = x.reshape(-1, x.size(-1))
        lora_a_out = self.lora_dropout(x_2d) @ self.lora_A_up.T  # [batch_size*seq_len, lora_r]
        lora_up = lora_a_out @ self.lora_B_up.T  # [batch_size*seq_len, hidden_dim]
        lora_up = lora_up.view(batch_size, seq_len, -1) * self.scaling
        
        # 2. 运行原始intermediate（上投影+激活）获取隐藏状态
        hidden = self.intermediate(x) + lora_up
        
        # 3. 计算下投影LoRA
        hidden_2d = hidden.reshape(-1, hidden.size(-1))
        lora_a_down_out = self.lora_dropout(hidden_2d) @ self.lora_A_down.T  # [batch_size*seq_len, lora_r]
        lora_down = lora_a_down_out @ self.lora_B_down.T  # [batch_size*seq_len, output_dim]
        lora_down = lora_down.view(batch_size, seq_len, -1) * self.scaling
        
        # 4. 应用下投影并添加LoRA结果
        output = base_output + lora_down
        
        # 5. 归一化
        output = F.layer_norm(output, normalized_shape=output.shape[-1:])
        
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
                
                # 应用第一个残差连接 (attention_output + 原始输入)
                first_residual = attention_output + hidden_states
                
                # 应用第二个layer norm
                normalized_residual = layer_module.layernorm_after(first_residual)
                
                # 获取区域特征（不含CLS token）用于疾病分类
                region_hidden_states = attention_output[:, 1:, :]
                
                # 通过疾病分类器获取路由权重
                with torch.no_grad():
                    disease_preds = self.classifiers[i](region_hidden_states)  # [B, num_diseases]
                    disease_probs = torch.sigmoid(disease_preds)  # [B, num_diseases]
                
                # 创建输出张量，初始化为0
                batch_size, seq_len, hidden_dim = normalized_residual.shape
                expert_output = torch.zeros_like(normalized_residual)
                
                # 首先使用原始FFN批量处理所有token (包括CLS和区域token)
                base_output = layer_module.output.dense(layer_module.intermediate(normalized_residual))
                
                # 创建专家权重矩阵 [batch_size, seq_len, num_experts]
                # seq_len包括cls token和所有区域token
                # 初始值为1，后续会归一化
                expert_weights = torch.ones(batch_size, seq_len, self.num_experts, device=device)
                
                # 创建掩码，标记哪些token-专家对需要处理 [batch_size, seq_len, num_experts]
                # 通用专家(最后一个)默认对所有token有效
                active_experts = torch.zeros(batch_size, seq_len, self.num_experts, device=device)
                active_experts[:, :, -1] = 1.0  # 通用专家对所有token生效，包括CLS
                
                # 处理疾病专家 (不包括通用专家)
                for d in range(self.num_diseases):
                    # 创建该疾病的掩码，标记哪些样本-区域需要处理
                    # [batch_size, seq_len] (seq_len包括CLS和所有区域)
                    disease_mask = torch.zeros(batch_size, seq_len, device=device)
                    
                    # 遍历每个批次样本
                    for b in range(batch_size):
                        # 检查该疾病是否被诊断为有病 (>0.5)
                        if disease_probs[b, d] > 0.5:
                            # 找出与该疾病相关的所有区域 (跳过CLS token，即索引0)
                            for r in range(1, seq_len):
                                region_idx = r - 1  # 实际区域索引
                                if region_idx < self.num_regions and self.anatomy_disease_mask[region_idx, d] > 0:
                                    # 标记该区域为需要处理
                                    disease_mask[b, r] = 1.0
                                    # 记录激活的专家
                                    active_experts[b, r, d] = 1.0
                
                # 归一化专家权重
                expert_sum = active_experts * expert_weights  # 只考虑激活的专家
                expert_sum = expert_sum.sum(dim=-1, keepdim=True)  # [batch_size, seq_len, 1]
                expert_weights = expert_weights / torch.clamp(expert_sum, min=1.0)  # 归一化，避免除零
                
                # 获取通用专家并应用到所有token
                general_expert = self.experts[f"layer_{i}"][-1]
                general_output = general_expert.compute_lora_output(normalized_residual, base_output)
                
                # 初始化最终输出为通用专家的加权输出
                final_output = general_output * expert_weights[:, :, -1].unsqueeze(-1)
                
                # 为每个疾病专家添加输出
                for d in range(self.num_diseases):
                    disease_expert = self.experts[f"layer_{i}"][d]
                    
                    # 创建掩码，标记该专家激活的token [batch_size, seq_len]
                    disease_active = active_experts[:, :, d]
                    
                    if disease_active.sum() > 0:
                        # 获取激活token的索引
                        batch_indices, token_indices = torch.where(disease_active > 0)
                        
                        if len(batch_indices) > 0:
                            # 收集需要处理的token
                            selected_tokens = normalized_residual[batch_indices, token_indices].unsqueeze(0)
                            selected_base_output = base_output[batch_indices, token_indices].unsqueeze(0)
                            
                            # 应用LoRA增强
                            selected_expert_output = disease_expert.compute_lora_output(selected_tokens, selected_base_output)
                            selected_expert_output = selected_expert_output.squeeze(0)  # [num_selected, hidden_dim]
                            
                            # 加权添加到输出
                            for idx, (b, t) in enumerate(zip(batch_indices, token_indices)):
                                weight = expert_weights[b, t, d]
                                final_output[b, t] += selected_expert_output[idx] * weight
                
                # 应用第二个残差连接 (FFN输出 + 第一个残差连接的结果)
                hidden_states = final_output + first_residual
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