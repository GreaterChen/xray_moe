# RGAT架构改动总结

## 概述

本次改动实现了基于双层异构图的三阶段临床推理网络(RGAT),替代原有的MoE模块。主要改动包括:

1. **创建RGAT模块** - 实现三阶段异构图注意力网络
2. **简化ViT** - 去除疾病分类器和MoE,变为标准ViT
3. **修改预训练阶段** - 删除负样本采样对齐(LTC),只保留patch-sentence对齐(Region ITC)
4. **修改微调阶段** - 集成RGAT,将14个疾病特征和ViT特征一起传入LLM
5. **更新配置和训练器** - 适配新架构

---

## 一、新增文件

### 1. `models/rgat.py` - RGAT网络模块

实现了完整的三阶段关系感知图注意力网络:

#### 核心组件:

- **`RelationalGraphAttentionLayer`**: 同构图注意力层,用于A→A和D→D
- **`HeterogeneousGraphAttentionLayer`**: 异构图注意力层,用于A→D
- **`DiseaseClassifierHead`**: 14个疾病的二分类器头
- **`ThreeStageRGAT`**: 主模块,协调三阶段推理

#### 三阶段推理流程:

1. **阶段1 - 解剖区域上下文感知 (A→A)**
   - 输入: 29个解剖区域特征
   - 输出: 上下文感知的解剖特征
   - 图结构: AA邻接矩阵 (K近邻)

2. **阶段2 - 疾病特异性表征聚合 (A→D)**
   - 输入: 上下文解剖特征 + 疾病嵌入
   - 输出: 初步疾病特征
   - 图结构: DA邻接矩阵 (解剖-疾病关联)

3. **阶段3 - 疾病间关系推理 (D→D)**
   - 输入: 初步疾病特征
   - 输出: 最终14个疾病特征
   - 图结构: DD邻接矩阵 (Jaccard权重)

#### 图矩阵加载:

- **AA矩阵**: `/mnt/chenlb/MIMIC/extra/anatomy_distance_matrix/anatomy_distance_matrix_adjacency_k5.csv`
- **DD矩阵**: `/mnt/chenlb/MIMIC/extra/disease_graph/disease_graph_adj_jaccard.csv`
- **DA矩阵**: `/mnt/chenlb/MIMIC/extra/anatomy_disease_matrix/anatomy_hierarchy_matrix.csv`

---

## 二、修改的核心文件

### 1. `models/vit.py` - 简化为标准ViT

#### 删除的组件:
- `DiseaseClassifier` 类
- `CompleteFeedForwardExpert` 类 (MoE专家)
- 所有疾病分类相关代码
- 所有MoE路由和专家选择逻辑

#### 保留的组件:
- 标准ViT encoder
- CLS token
- LayerNorm

#### 新的forward函数:
```python
def forward(self, region_features):
    # 简单的标准ViT前向传播
    # 输入: [B, 29, 768]
    # 输出: [B, 30, 768] (1个CLS + 29个区域)
```

---

### 2. `models/moe_model.py` - 集成RGAT模块

#### 主要改动:

1. **删除负样本采样**:
   - 移除 `NegativeSamplePool` 导入和初始化
   - 删除 `compute_global_ltc_loss` 函数
   - 删除 `compute_batch_ltc_loss` 函数

2. **添加RGAT模块**:
   ```python
   if config.PHASE == "FINETUNE_BERT":
       self.rgat = ThreeStageRGAT(...)
   ```

3. **修改PRETRAIN_VIT阶段**:
   ```python
   # 只保留Region-ITC损失
   visual_features = self.image_encoder(region_features)
   region_itc_loss = self.compute_region_itc_loss(...)
   ```

4. **修改FINETUNE阶段**:
   ```python
   # 1. ViT提取视觉特征 (冻结)
   visual_features = self.image_encoder(region_features)
   
   # 2. RGAT推理 (可训练)
   anatomy_features = visual_features[:, 1:, :]
   disease_features, disease_preds = self.rgat(anatomy_features)
   
   # 3. 拼接特征
   combined_features = torch.cat([visual_features, disease_features], dim=1)
   # [B, 30, 768] + [B, 14, 768] = [B, 44, 768]
   
   # 4. 输入到LLM
   outputs = self.findings_decoder(visual_features=combined_features, ...)
   ```

---

### 3. `configs/` - 配置文件更新

#### `default_config.py` 新增参数:
```python
# RGAT配置
AA_ADJ_PATH = None  # Anatomy-Anatomy邻接矩阵路径
DD_ADJ_PATH = None  # Disease-Disease邻接矩阵路径
DA_ADJ_PATH = None  # Disease-Anatomy邻接矩阵路径
RGAT_DROPOUT = 0.1  # RGAT dropout率
RGAT_LOSS_WEIGHT = 1.0  # RGAT损失权重
```

#### `local_config.py` 新增配置:
```python
# 图矩阵路径
AA_ADJ_PATH = "/mnt/chenlb/MIMIC/extra/anatomy_distance_matrix/anatomy_distance_matrix_adjacency_k5.csv"
DD_ADJ_PATH = "/mnt/chenlb/MIMIC/extra/disease_graph/disease_graph_adj_jaccard.csv"
DA_ADJ_PATH = "/mnt/chenlb/MIMIC/extra/anatomy_disease_matrix/anatomy_hierarchy_matrix.csv"

# RGAT配置
RGAT_DROPOUT = 0.1
RGAT_LOSS_WEIGHT = 1.0
```

---

### 4. `utils/train_utils.py` - 训练逻辑更新

#### 修改的函数:

**`_compute_loss`**:
```python
# 预训练阶段: 只有Region-ITC损失
if phase == "PRETRAIN_VIT":
    loss = region_itc_weight * output["region_itc_loss"]

# 微调阶段: 生成损失 + RGAT损失
if phase == "FINETUNE_BERT":
    loss = output.loss
    if hasattr(output, 'rgat_loss'):
        loss += rgat_weight * output.rgat_loss
```

**`_log_training_metrics`**:
```python
# 预训练阶段: 只记录Region-ITC损失
elif phase == "PRETRAIN_VIT":
    writer.add_scalar("Train/ViT/Region_ITC_Loss", ...)

# 微调阶段: 记录生成损失和RGAT损失
elif phase in ["FINETUNE_BERT", ...]:
    writer.add_scalar("Train/Finetune/Generation_Loss", ...)
    writer.add_scalar("Train/Finetune/RGAT_Loss", ...)
```

---

### 5. `utils/eval_utils.py` - 评估逻辑更新

#### 修改的函数:

**`test_vit`**: 大幅简化,只评估Region-ITC损失

之前:
- 评估疾病分类性能
- 计算准确率、精确率、召回率、F1、AUC等
- 保存详细的分类报告

现在:
- 只计算Region-ITC损失
- 简化返回结果

```python
def test_vit(...):
    # 只收集region_itc_loss
    for batch in data_loader:
        outputs = model(**source)
        if "region_itc_loss" in outputs:
            running_loss += outputs["region_itc_loss"].item()
    
    # 返回简化结果
    result = {
        "overall_metrics": {
            "ce_f1": avg_region_itc_loss,  # 用作主要指标
        },
        "loss": avg_loss,
        "region_itc_loss": avg_region_itc_loss,
    }
    return avg_loss, result
```

---

## 三、训练流程变化

### 预训练阶段 (PRETRAIN_VIT)

**之前**:
1. 检测器提取区域特征 (冻结)
2. ViT处理 + MoE + 疾病分类器 (可训练)
3. CXR-BERT编码文本 (冻结)
4. 计算三种损失:
   - LTC损失 (全局对比,使用负样本池)
   - Region-ITC损失 (patch-sentence对齐)
   - 疾病分类损失

**现在**:
1. 检测器提取区域特征 (冻结)
2. 标准ViT处理 (可训练)
3. **只计算Region-ITC损失** (patch-sentence对齐)

---

### 微调阶段 (FINETUNE_BERT)

**之前**:
1. 检测器提取区域特征 (冻结)
2. ViT处理 + MoE路由 (可训练)
3. 生成报告 (可训练)
4. 计算生成损失

**现在**:
1. 检测器提取区域特征 (冻结)
2. **标准ViT处理 (冻结)**
3. **RGAT三阶段推理 (可训练)**
   - 输入: 29个解剖区域特征
   - 输出: 14个疾病特征 + 疾病分类预测
4. **拼接ViT特征和疾病特征**
   - [B, 30, 768] (ViT) + [B, 14, 768] (RGAT) = [B, 44, 768]
5. 生成报告 (可训练)
6. 计算两种损失:
   - 生成损失
   - **RGAT疾病分类损失**

---

## 四、可训练参数变化

### 预训练阶段:
- **之前**: 检测器(冻结) + ViT(可训练) + 分类器(可训练) + MoE(可训练)
- **现在**: 检测器(冻结) + **标准ViT(可训练)**

### 微调阶段:
- **之前**: 检测器(冻结) + ViT(冻结) + MoE(可训练) + Decoder(可训练)
- **现在**: 检测器(冻结) + **ViT(冻结)** + **RGAT(可训练)** + Decoder(可训练)

---

## 五、特征维度变化

### 预训练阶段输出:
- **之前**: `visual_features` [B, 30, 768] + 疾病预测 [B, 14]
- **现在**: `visual_features` [B, 30, 768]

### 微调阶段输入到LLM:
- **之前**: `visual_features` [B, 30, 768]
- **现在**: `combined_features` [B, 44, 768]
  - 30个token: 1个CLS + 29个解剖区域
  - 14个token: 14个疾病特征

---

## 六、关键优势

1. **更强的临床可解释性**:
   - 三阶段明确对应临床诊断流程
   - 图结构可视化注意力权重

2. **显式知识建模**:
   - 解剖空间关系 (AA图)
   - 疾病共现模式 (DD图,Jaccard权重)
   - 解剖-疾病关联 (DA图)

3. **更丰富的特征表示**:
   - 44个token vs 之前的30个token
   - 疾病特征显式建模

4. **模块化设计**:
   - ViT和RGAT解耦
   - 易于单独调试和优化

---

## 七、使用指南

### 1. 预训练阶段

```bash
# 修改配置
config.PHASE = "PRETRAIN_VIT"

# 启用Region-ITC
config.ENABLE_REGION_ITC = True
config.REGION_ITC_WEIGHT = 1.0

# 运行训练
python train.py
```

### 2. 微调阶段

```bash
# 修改配置
config.PHASE = "FINETUNE_BERT"

# 设置图矩阵路径 (已在local_config.py中配置)
# AA_ADJ_PATH, DD_ADJ_PATH, DA_ADJ_PATH

# 设置RGAT参数
config.RGAT_DROPOUT = 0.1
config.RGAT_LOSS_WEIGHT = 1.0

# 加载ViT预训练权重
config.VIT_CHECKPOINT_PATH_FROM = "path/to/pretrained_vit.pth"

# 运行训练
python train.py
```

---

## 八、注意事项

1. **图矩阵文件格式**:
   - CSV格式,无表头
   - AA矩阵: [29, 29] - 0/1二值矩阵
   - DD矩阵: [14, 14] - 连续权重矩阵 (Jaccard系数)
   - DA矩阵: [29, 14] - 0/1二值矩阵

2. **检查点兼容性**:
   - 旧的包含MoE的检查点不兼容
   - 需要重新训练或手动转换

3. **内存使用**:
   - RGAT增加了额外的参数和计算
   - 微调阶段输入LLM的token数增加 (30→44)

4. **评估指标**:
   - 预训练阶段主要指标变为`region_itc_loss`
   - 微调阶段增加`rgat_loss`

---

## 九、后续优化方向

1. **图结构优化**:
   - 尝试不同的K值 (KNN)
   - 探索其他边权重计算方法

2. **RGAT架构优化**:
   - 调整隐藏层维度
   - 增加更多图注意力层
   - 尝试多头注意力

3. **损失权重调优**:
   - 平衡生成损失和RGAT损失
   - 动态调整权重

4. **知识蒸馏**:
   - 从旧的MoE模型蒸馏知识到RGAT

---

**实现日期**: 2025年10月29日
**文档版本**: v1.0

