# Region ITC (Image-Text Contrastive) 损失计算文档

## 1. 概述

Region ITC是一种区域级别的图像-文本对比学习损失，用于对齐医学影像中的解剖区域视觉特征与对应的文本描述特征。该损失函数在模型预训练阶段使用，帮助模型学习更好的视觉-语义对齐。

**核心思想**：将检测到的解剖区域的视觉特征与其对应的文本描述进行对比学习，使相同语义的视觉-文本对更接近，不同语义的对更远离。

---

## 2. 整体流程

Region ITC损失计算分为三个主要步骤：

```
输入数据
  ↓
步骤1: 收集有效的视觉-文本对 (compute_region_itc_loss)
  ↓
步骤2: 计算相似度并构建正负样本掩码 (_compute_region_itc_direct)
  ↓
步骤3: 计算监督对比学习损失 (_supervised_contrastive_loss)
  ↓
输出损失值
```

---

## 3. 详细实现

### 3.1 主函数：`compute_region_itc_loss`

**函数签名**：
```python
def compute_region_itc_loss(
    self, 
    visual_features,                   # [B, 1+num_regions, hidden_size]
    region_detected,                   # [B, num_regions]
    anatomical_embeddings_batch,       # List[Dict[int, Tensor]]
    anatomical_nlp_status_batch,       # List[Dict[int, str]]
    same_text_region_groups_batch,     # List[List[List[int]]]
    image_ids                          # List[str] (可选)
) -> Optional[torch.Tensor]
```

**输入参数**：
- `visual_features`: ViT编码后的视觉特征，形状为 `[B, 1+num_regions, hidden_size]`，其中第一个token是CLS，后面29个是解剖区域特征
- `region_detected`: 区域检测掩码，形状为 `[B, 29]`，表示每个区域是否被检测到
- `anatomical_embeddings_batch`: 每个样本的解剖区域文本嵌入字典，格式为 `{region_id: text_embedding}`
- `anatomical_nlp_status_batch`: 每个样本的解剖区域NLP状态字典，格式为 `{region_id: "normal"/"abnormal"}`
- `same_text_region_groups_batch`: 每个样本的同文本区域分组列表，格式为 `[[1, 3], [2], [5, 7, 9]]`（表示区域1和3具有相同文本，区域2单独，区域5、7、9相同文本）
- `image_ids`: 图像ID列表（用于调试）

**主要逻辑**：

1. **有效性检查**：
   ```python
   if not anatomical_embeddings_batch:
       return None
   ```

2. **收集有效的视觉-文本对**：
   - 遍历批次中的每个样本
   - 对于每个样本，遍历其所有解剖区域
   - 只保留被检测到的区域（`region_detected > 0.5`）
   - 收集四个列表：
     - `valid_pairs`: `[(batch_idx, region_idx), ...]` 有效的样本-区域索引对
     - `text_embeds_list`: 对应的文本嵌入列表
     - `nlp_status_list`: 对应的NLP状态列表（"normal"/"abnormal"/None）
     - `same_text_group_ids`: 对应的同文本分组ID列表 `[(batch_idx, group_id), ...]`

3. **样本数量检查**：
   ```python
   if total_valid < 2:
       return None
   ```
   - 至少需要2个有效样本才能计算对比损失

---

### 3.2 核心计算：`_compute_region_itc_direct`

这是损失计算的核心函数，实现了基于NLP状态的监督对比学习。

#### 3.2.1 特征提取与投影

1. **提取区域视觉特征**：
   ```python
   region_visual = visual_features[:, 1:30, :]  # 去除CLS token
   visual_feats = region_visual[batch_indices, region_indices]  # [N, hidden_size]
   ```

2. **数值稳定性检查**：
   ```python
   if torch.isnan(visual_feats).any() or torch.isinf(visual_feats).any():
       visual_feats = torch.nan_to_num(visual_feats, nan=0.0, posinf=1.0, neginf=-1.0)
   ```

3. **特征投影与归一化**：
   ```python
   mapped_visual = F.normalize(self.region_visual_projection(visual_feats), p=2, dim=1, eps=1e-8)
   mapped_text = F.normalize(self.region_text_projection(text_embeds), p=2, dim=1, eps=1e-8)
   ```
   - 使用独立的线性投影层：`region_visual_projection` 和 `region_text_projection`
   - L2归一化确保特征在单位球面上

4. **计算相似度矩阵**：
   ```python
   temperature = getattr(self.config, 'REGION_ITC_TEMPERATURE', 0.07)
   logits = torch.matmul(mapped_visual, mapped_text.t()) / temperature  # [N, N]
   logits = torch.clamp(logits, min=-10.0, max=10.0)  # 数值稳定性
   ```

#### 3.2.2 正负样本掩码构建（核心创新）

这是Region ITC的关键创新点，基于**解剖区域**和**NLP状态**双重约束定义正负样本。

**1. 基本掩码**：
```python
same_region_mask = (region_indices.unsqueeze(1) == region_indices.unsqueeze(0))  # [N, N]
diff_region_mask = (region_indices.unsqueeze(1) != region_indices.unsqueeze(0))  # [N, N]
diff_sample_mask = (batch_indices.unsqueeze(1) != batch_indices.unsqueeze(0))   # [N, N]
```

**2. NLP状态编码**：
```python
# 将状态转换为数值：normal=0, abnormal=1, None=-1
status_codes = []
for status in nlp_status_list:
    if status == "normal":
        status_codes.append(0)
    elif status == "abnormal":
        status_codes.append(1)
    else:
        status_codes.append(-1)  # 未知状态
```

**3. 状态掩码**：
```python
same_status_mask = (status_tensor.unsqueeze(1) == status_tensor.unsqueeze(0))  # [N, N]
diff_status_mask = (status_tensor.unsqueeze(1) != status_tensor.unsqueeze(0))  # [N, N]
valid_pair_mask = valid_status_mask.unsqueeze(1) & valid_status_mask.unsqueeze(0)  # [N, N]
```

**3.5. 同文本分组掩码（新增）**：
```python
# 构建同文本区域掩码
group_batch_indices = torch.tensor([gid[0] for gid in same_text_group_ids], device=device)  # [N]
group_ids = torch.tensor([gid[1] for gid in same_text_group_ids], device=device)  # [N]

# 同一分组掩码：同一样本 且 同一分组ID 且 分组ID有效（>= 0）
same_sample_mask = (group_batch_indices.unsqueeze(1) == group_batch_indices.unsqueeze(0))  # [N, N]
same_group_mask = (group_ids.unsqueeze(1) == group_ids.unsqueeze(0))  # [N, N]
valid_group_mask = (group_ids >= 0)  # [N]
valid_group_pair_mask = valid_group_mask.unsqueeze(1) & valid_group_mask.unsqueeze(0)  # [N, N]

# 同文本区域掩码：同一样本 且 同一分组 且 分组有效
same_text_mask = same_sample_mask & same_group_mask & valid_group_pair_mask  # [N, N]
```

**4. 正样本定义**：
```python
positive_mask = same_region_mask & same_status_mask & valid_pair_mask
```
✅ **正样本条件**：
- 同一解剖区域（如都是"左肺"）
- **且** 相同NLP状态（都是normal或都是abnormal）
- **且** 状态都有效（不是None）

**示例**：
- ✅ 图像A的左肺(normal) ←→ 图像B的左肺(normal)
- ✅ 图像A的心脏(abnormal) ←→ 图像B的心脏(abnormal)

**5. 负样本定义（含同文本分组过滤）**：
```python
# 情况1：同一解剖区域 但 不同NLP状态（必然是不同样本，不需要考虑同文本）
negative_mask_same_region_diff_status = same_region_mask & diff_status_mask & valid_pair_mask

# 情况2：不同解剖区域
# 2a: 同样本的不同区域 且 不是同文本区域
negative_mask_same_sample_diff_region = (~diff_sample_mask) & diff_region_mask & (~same_text_mask)
# 2b: 不同样本的不同区域（都是负样本）
negative_mask_diff_sample_diff_region = diff_sample_mask & diff_region_mask

# 合并所有负样本
negative_mask = negative_mask_same_region_diff_status | negative_mask_same_sample_diff_region | negative_mask_diff_sample_diff_region
```

❌ **负样本条件**（三种情况）：

**情况1：同一解剖区域 但 不同NLP状态**
- 同一解剖区域必然来自不同样本
- 不需要考虑同文本区域（因为是不同样本）
- 示例：图像A的左肺(normal) ←→ 图像B的左肺(abnormal)

**情况2a：同样本的不同解剖区域 且 非同文本区域**
- 同一图像内的不同解剖区域
- 但必须不在同一文本分组中
- 示例：图像A的左肺 ←→ 图像A的心脏（非同文本分组）

**情况2b：不同样本的不同解剖区域**
- 来自不同图像的不同解剖区域
- 全都是负样本（不需要考虑同文本区域）
- 示例：图像A的左肺 ←→ 图像B的心脏

**6. 忽略样本**：
```python
eye_mask = torch.eye(N, device=device, dtype=torch.bool)
positive_mask = positive_mask & (~eye_mask)
```
⚠️ **忽略条件**：
- 对角线样本（自己与自己）
- **同一图像中具有相同文本描述的不同区域（同一分组内的区域）** 【新增】
  - 例如：图像A的区域1 和 区域3 在同一文本分组中，则不作为负样本

---

### 3.3 监督对比损失：`_supervised_contrastive_loss`

**损失函数类型**：InfoNCE (Information Noise Contrastive Estimation)

**数学公式**：
```
Loss_i = -log( Σ exp(sim(v_i, t_pos)) / (Σ exp(sim(v_i, t_pos)) + Σ exp(sim(v_i, t_neg))) )
```

其中：
- `v_i`: 第i个视觉特征
- `t_pos`: 正样本文本特征集合
- `t_neg`: 负样本文本特征集合
- `sim(·,·)`: 余弦相似度 / temperature

**实现逻辑**：

1. **检查有效性**：
   ```python
   num_positives = positive_mask.sum(dim=1)  # [N]
   has_positive = num_positives > 0  # [N]
   if not has_positive.any():
       return torch.tensor(0.0, device=device, requires_grad=True)
   ```

2. **逐样本计算损失**：
   ```python
   for i in valid_indices:
       pos_mask_i = positive_mask[i]  # 第i个样本的正样本掩码
       neg_mask_i = negative_mask[i]  # 第i个样本的负样本掩码
       
       pos_logits = logits[i][pos_mask_i]  # 正样本相似度
       neg_logits = logits[i][neg_mask_i]  # 负样本相似度
       
       pos_exp = torch.exp(pos_logits)
       neg_exp = torch.exp(neg_logits)
       
       pos_sum = pos_exp.sum()
       neg_sum = neg_exp.sum()
       
       loss_i = -torch.log(pos_sum / (pos_sum + neg_sum + 1e-8) + 1e-8)
       losses.append(loss_i)
   ```

3. **平均所有损失**：
   ```python
   loss = torch.stack(losses).mean()
   ```

---

## 4. 关键配置参数

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `ENABLE_REGION_ITC` | `True` | 是否启用Region ITC损失 |
| `REGION_ITC_TEMPERATURE` | `0.07` | 温度参数（控制相似度分布的尖锐度） |

---

## 5. 使用场景

Region ITC损失在以下阶段使用：

### 5.1 预训练阶段 (`PRETRAIN_VIT`)

```python
if mode == "train":
    region_itc_loss = self.compute_region_itc_loss(
        visual_features, 
        region_detected, 
        anatomical_embeddings_batch, 
        anatomical_nlp_status_batch, 
        image_ids
    )
    results = {
        "region_itc_loss": region_itc_loss,
        "visual_features": visual_features,
    }
```

在ViT预训练阶段，通过对比学习使视觉编码器学习到：
- 相同解剖区域的视觉特征应该相似
- 但只有在NLP状态相同时才应该拉近（正样本）
- 相同区域但状态不同的应该推远（负样本）

这种设计使模型能够区分正常和异常的解剖区域。

---

## 6. 创新点与优势

### 6.1 创新点

1. **基于NLP状态的细粒度对比**：
   - 传统对比学习：只考虑区域类别（左肺 vs 右肺）
   - Region ITC：同时考虑区域类别和健康状态（正常左肺 vs 异常左肺）

2. **三层次正负样本定义**：
   - 区域级别：相同/不同解剖区域
   - 状态级别：正常/异常
   - **文本级别：相同/不同文本描述（新增）**

3. **同文本区域过滤机制（新增）**：
   - 自动识别同一图像中具有相同文本描述的不同区域
   - 避免将文本相同的区域作为负样本进行对比
   - 提升训练的合理性和稳定性

4. **计算优化**：
   - 批量操作减少GPU查询次数
   - 使用所有可用的视觉-文本对进行对比学习

### 6.2 优势

1. **更精确的语义对齐**：使模型学习到病变相关的视觉特征
2. **提升下游任务性能**：在报告生成和疾病分类任务上表现更好
3. **稳健性**：数值稳定性检查和异常处理

---

## 7. 数值稳定性保障

代码中采用多种措施确保数值稳定：

```python
# 1. NaN/Inf检查与修复
visual_feats = torch.nan_to_num(visual_feats, nan=0.0, posinf=1.0, neginf=-1.0)

# 2. 归一化时添加epsilon
mapped_visual = F.normalize(visual_feats, p=2, dim=1, eps=1e-8)

# 3. Logits范围限制
logits = torch.clamp(logits, min=-10.0, max=10.0)

# 4. 分母添加epsilon避免除零
denominator = pos_sum + neg_sum + 1e-8
loss_i = -torch.log(pos_sum / denominator + 1e-8)
```

---

## 8. 完整示例（含同文本区域分组）

假设一个批次有2张图像，每张检测到3个区域：

```python
# 输入
visual_features: [2, 30, 768]  # 2个样本，每个30个token（1个CLS + 29个区域）
region_detected: [2, 29]       # 检测掩码

anatomical_embeddings_batch: [
    {1: tensor(...), 5: tensor(...), 10: tensor(...)},  # 图像0的区域1,5,10
    {1: tensor(...), 5: tensor(...), 15: tensor(...)},  # 图像1的区域1,5,15
]

anatomical_nlp_status_batch: [
    {1: "normal", 5: "abnormal", 10: "normal"},
    {1: "normal", 5: "normal", 15: "abnormal"},
]

# 新增：同文本区域分组
same_text_region_groups_batch: [
    [[1, 10], [5]],      # 图像0：区域1和10具有相同文本，区域5单独
    [[1], [5, 15]],      # 图像1：区域1单独，区域5和15具有相同文本
]

# 收集到的有效对（假设都检测到）
valid_pairs = [
    (0, 0),  # 图像0, 区域1 (normal)
    (0, 4),  # 图像0, 区域5 (abnormal)
    (0, 9),  # 图像0, 区域10 (normal)
    (1, 0),  # 图像1, 区域1 (normal)
    (1, 4),  # 图像1, 区域5 (normal)
    (1, 14), # 图像1, 区域15 (abnormal)
]

same_text_group_ids = [
    (0, 0),  # 图像0, 分组0 (区域1)
    (0, 1),  # 图像0, 分组1 (区域5)
    (0, 0),  # 图像0, 分组0 (区域10，与区域1同组)
    (1, 0),  # 图像1, 分组0 (区域1)
    (1, 1),  # 图像1, 分组1 (区域5)
    (1, 1),  # 图像1, 分组1 (区域15，与区域5同组)
]

# 正负样本关系分析
# ==========================================
# (0,0) - 图像0区域1(normal) 的对比关系：
# ------------------------------------------
# 与 (0,4) - 图像0区域5(abnormal):
#   → 同样本、不同区域、非同文本分组 → 负样本 ❌ (情况2a)
#
# 与 (0,9) - 图像0区域10(normal):
#   → 同样本、不同区域、同文本分组 → 忽略 ⚠️ (同文本分组过滤)
#
# 与 (1,0) - 图像1区域1(normal):
#   → 不同样本、相同区域、相同状态 → 正样本 ✅
#
# 与 (1,4) - 图像1区域5(normal):
#   → 不同样本、不同区域 → 负样本 ❌ (情况2b)
#
# 与 (1,14) - 图像1区域15(abnormal):
#   → 不同样本、不同区域 → 负样本 ❌ (情况2b)
# ==========================================

# 关键点总结：
# 1. 同一解剖区域 + 不同NLP状态 → 负样本（情况1，必然是不同样本）
# 2. 同样本 + 不同区域 + 非同文本分组 → 负样本（情况2a）
# 3. 同样本 + 不同区域 + 同文本分组 → 忽略（同文本区域过滤）
# 4. 不同样本 + 不同区域 → 负样本（情况2b，不需要考虑同文本）
```

---

## 9. 数据格式说明

### 9.1 解剖区域数据库格式

数据库使用 pickle 格式存储，包含以下字段：

```python
{
    'image_region_embeddings': {
        'image_id1_1': np.array([...]),   # 图像1的区域1的文本嵌入
        'image_id1_5': np.array([...]),   # 图像1的区域5的文本嵌入
        ...
    },
    'image_region_nlp_status': {
        'image_id1_1': 'normal',          # 图像1的区域1的NLP状态
        'image_id1_5': 'abnormal',        # 图像1的区域5的NLP状态
        ...
    },
    'image_same_text_region_groups': {   # 新增字段
        'image_id1': [[1, 3], [2], [5, 7, 9]],  # 图像1的同文本区域分组
        'image_id2': [[4, 6], [8], [10]],       # 图像2的同文本区域分组
        ...
    },
    'metadata': {
        'total_keys': 12345,
        'embedding_dim': 768,
        'model_name': 'dmis-lab/biobert-v1.1',
        ...
    }
}
```

**同文本区域分组说明**：
- 每个图像对应一个分组列表
- 每个分组是一个区域索引列表
- 同一分组内的区域具有相同的文本描述
- 例如 `[[1, 3], [2], [5, 7, 9]]` 表示：
  - 区域1和3的文本叙述相同
  - 区域2是单独的文本
  - 区域5、7、9的文本叙述相同

---

## 10. 总结

Region ITC是一个精心设计的对比学习损失，核心特点：

1. **区域级别**：针对29个解剖区域分别学习
2. **状态感知**：区分正常和异常状态
3. **文本分组感知（新增）**：识别并排除同文本区域作为负样本
4. **监督对比**：使用NLP提取的状态标签构建正负样本
5. **计算高效**：批量操作优化计算效率
6. **数值稳定**：多层保障确保训练稳定性

通过这种方式，模型能够学习到更细粒度、更具判别性的视觉特征，同时避免不合理的对比（如将相同文本的区域推远），为下游的报告生成和疾病分类任务奠定坚实基础。

