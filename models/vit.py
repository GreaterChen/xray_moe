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
    专注于LoRA部分的前馈网络专家模块，避免重复计算
    """
    def __init__(self, input_dim, hidden_dim, output_dim, lora_r=8, lora_alpha=16, lora_dropout=0.1):
        super(CompleteFeedForwardExpert, self).__init__()
        
        # 不再保存原始FFN组件，只记录维度信息
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

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
    
    def compute_lora_output(self, x, intermediate_output, base_output):
        """
        计算LoRA部分的输出
        x: [batch_size, seq_len, input_dim] 原始输入
        intermediate_output: [batch_size, seq_len, hidden_dim] 上投影(中间层)的输出
        base_output: [batch_size, seq_len, output_dim] FFN的基础输出
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. 计算上投影LoRA
        x_2d = x.reshape(-1, x.size(-1))
        lora_a_out = self.lora_dropout(x_2d) @ self.lora_A_up.T  # [batch_size*seq_len, lora_r]
        lora_up = lora_a_out @ self.lora_B_up.T  # [batch_size*seq_len, hidden_dim]
        lora_up = lora_up.view(batch_size, seq_len, -1) * self.scaling
        
        # 2. 添加LoRA到中间层输出
        hidden = intermediate_output + lora_up
        
        # 3. 计算下投影LoRA
        hidden_2d = hidden.reshape(-1, hidden.size(-1))
        lora_a_down_out = self.lora_dropout(hidden_2d) @ self.lora_A_down.T  # [batch_size*seq_len, lora_r]
        lora_down = lora_a_down_out @ self.lora_B_down.T  # [batch_size*seq_len, output_dim]
        lora_down = lora_down.view(batch_size, seq_len, -1) * self.scaling
        
        # 4. 添加LoRA到基础输出
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
                        input_dim=ffn_intermediate.dense.weight.shape[1],
                        hidden_dim=ffn_intermediate.dense.weight.shape[0],
                        output_dim=ffn_output.dense.weight.shape[0],
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
        use_moe=True,
    ):
        """
        前向传播

        Args:
            region_features: [batch_size, num_regions, hidden_size] - 区域特征
            region_detected: [batch_size, num_regions] - 已检测区域的掩码 (未使用，保留兼容性)
            image_labels: [batch_size, num_diseases] - 图像标签
            phase: 当前训练阶段
            use_moe: 是否启用MOE，如果为False则使用普通FFN
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
        
        # 保存各层的分类结果，避免重复计算
        layer_disease_predictions = {}
        
        # 通过Transformer层
        hidden_states = x
        for i, layer_module in enumerate(self.encoder.layer):
            # 在FINETUNE阶段，使用MOE（如果启用）
            if phase in ["FINETUNE_MISTRAL", "FINETUNE_LLAMA", "FINETUNE_BERT"] and i % 2 == 0 and use_moe:
                # 先应用layernorm_before，再使用原始注意力机制
                normalized_hidden_states = layer_module.layernorm_before(hidden_states)
                attention_output = layer_module.attention(normalized_hidden_states)[0]
                
                # 应用第一个残差连接 (attention_output + 原始输入)
                first_residual = attention_output + hidden_states
                
                # 应用第二个layer norm
                normalized_residual = layer_module.layernorm_after(first_residual)
                
                # 获取区域特征（不含CLS token）用于疾病分类
                region_hidden_states = attention_output[:, 1:, :]
                
                # 进行分类预测（带梯度）
                disease_preds = self.classifiers[i](region_hidden_states)  # [B, num_diseases]
                disease_probs = torch.sigmoid(disease_preds)  # [B, num_diseases]
                
                # 保存分类结果到字典中，以便后续复用
                layer_disease_predictions[i] = disease_preds
                all_disease_preds.append(disease_preds)
                
                # 创建输出张量，初始化为0
                batch_size, seq_len, hidden_dim = normalized_residual.shape
                
                # 首先使用原始FFN批量处理所有token (包括CLS和区域token)
                intermediate_output = layer_module.intermediate(normalized_residual)
                base_output = layer_module.output.dense(intermediate_output)
                
                # 创建专家权重和激活掩码
                # 使用矩阵运算批量构建激活掩码，避免嵌套循环
                
                # 1. 创建通用专家的激活掩码 - 对所有token都激活
                active_experts = torch.zeros(batch_size, seq_len, self.num_experts, device=device)
                active_experts[:, :, -1] = 1.0  # 通用专家对所有token激活
                
                # 2. 为疾病专家创建激活掩码 - 完全向量化处理
                if seq_len > 1 and self.num_diseases > 0:  # 确保有区域token和疾病专家
                    # 获取图像级别的疾病激活状态 [batch_size, num_diseases]
                    disease_active = (disease_probs > 0.5).float()  # [B, num_diseases]
                    
                    # 使用解剖区域-疾病关系掩码 [num_regions, num_diseases]
                    region_disease_mask = self.anatomy_disease_mask  # [num_regions, num_diseases]
                    
                    # 使用广播计算哪些[区域, 疾病]对在每个batch样本中是激活的
                    # disease_active.unsqueeze(1): [B, 1, num_diseases]
                    # region_disease_mask.unsqueeze(0): [1, num_regions, num_diseases]
                    # 结果: [B, num_regions, num_diseases]，表示批次b的区域r与激活的疾病d相关
                    region_expert_active_mask = disease_active.unsqueeze(1) * region_disease_mask.unsqueeze(0)
                    
                    # 将这个[B, num_regions, num_diseases]的掩码填充到active_experts的对应位置
                    # active_experts的维度是[B, seq_len, num_experts] = [B, 1+num_regions, num_diseases+1]
                    # 我们需要填充的部分是[:, 1:1+num_regions, :num_diseases]
                    valid_regions = self.num_regions
                    active_experts[:, 1:1+valid_regions, :self.num_diseases] = region_expert_active_mask[:, :valid_regions, :].float()
                
                # 3. 创建专家权重矩阵 - 所有激活的专家权重相等
                expert_weights = active_experts.clone()
                
                # 4. 归一化专家权重
                expert_sum = expert_weights.sum(dim=-1, keepdim=True)  # [batch_size, seq_len, 1]
                expert_weights = expert_weights / torch.clamp(expert_sum, min=1.0)  # 归一化权重
                
                # 5. 应用通用专家 - 一次性处理所有token
                general_expert = self.experts[f"layer_{i}"][-1]
                general_output = general_expert.compute_lora_output(normalized_residual, intermediate_output, base_output)
                
                # 初始化最终输出为通用专家的加权输出
                final_output = general_output * expert_weights[:, :, -1].unsqueeze(-1)
                
                # 6. 疾病专家处理 - 批量处理有效的专家和token
                # 找出所有激活的疾病专家
                active_disease_experts = []
                for d in range(self.num_diseases):
                    if active_experts[:, :, d].sum() > 0:  # 只处理有激活的专家
                        active_disease_experts.append(d)
                
                # 批量处理激活的专家
                for d in active_disease_experts:
                    # 获取当前疾病专家的激活掩码
                    disease_mask = active_experts[:, :, d]  # [batch_size, seq_len]
                    
                    # 获取需要处理的token索引
                    batch_indices, token_indices = torch.where(disease_mask > 0)
                    
                    if len(batch_indices) > 0:
                        # 收集所有需要处理的token
                        selected_tokens = normalized_residual[batch_indices, token_indices]
                        selected_intermediate = intermediate_output[batch_indices, token_indices]
                        selected_base_output = base_output[batch_indices, token_indices]
                        
                        # 获取当前疾病专家
                        disease_expert = self.experts[f"layer_{i}"][d]
                        
                        # 批量应用专家 - 去除不必要的维度操作
                        selected_expert_output = disease_expert.compute_lora_output(
                            selected_tokens.unsqueeze(0), 
                            selected_intermediate.unsqueeze(0), 
                            selected_base_output.unsqueeze(0)
                        ).squeeze(0)
                        
                        # 收集权重
                        weights = expert_weights[batch_indices, token_indices, d]
                        
                        # 计算加权专家输出
                        weighted_expert_output = selected_expert_output * weights.unsqueeze(-1)
                        
                        # 将加权输出添加到最终输出
                        indices_tuple = (batch_indices, token_indices)
                        final_output.index_put_(indices_tuple, weighted_expert_output, accumulate=True)
                
                # 应用第二个残差连接 (FFN输出 + 第一个残差连接的结果)
                hidden_states = final_output + first_residual
            else:
                # 正常的Transformer前向传播
                layer_outputs = layer_module(hidden_states)
                hidden_states = layer_outputs[0]
            
            all_hidden_states.append(hidden_states)

            # 只在偶数层进行分类预测（如果尚未在MOE中处理过）
            if i % 2 == 0 and self.classifiers[i] is not None and i not in layer_disease_predictions:
                # 区域特征（不含CLS token）
                region_hidden_states = hidden_states[:, 1:, :]
                # 通过疾病分类器
                disease_preds = self.classifiers[i](region_hidden_states)
                # 保存结果
                layer_disease_predictions[i] = disease_preds
                all_disease_preds.append(disease_preds)

        # 最终输出
        final_hidden_states = self.layernorm(hidden_states)
        
        # 计算损失（如果提供了标签）- 直接使用之前的分类结果
        loss = None
        if image_labels is not None and len(all_disease_preds) > 0:
            loss = 0
            num_classifier_layers = len(all_disease_preds)  # 实际使用分类器的层数
            for i, disease_preds in enumerate(all_disease_preds):
                # 直接使用已计算的分类结果计算损失
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
            "visual_features": final_hidden_states,  # 完整的视觉特征 [B, 1+num_regions, hidden_size]
        }