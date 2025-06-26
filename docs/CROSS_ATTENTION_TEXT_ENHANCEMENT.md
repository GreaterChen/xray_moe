# Cross-Attention 文本增强功能

这个模块提供了一种基于Cross-Attention的文本增强方法，通过检索相似的文本描述并使用Cross-Attention机制来增强视觉特征。

## 功能特性

- **基于视觉特征检索**：使用区域视觉特征检索相似的文本描述
- **Cross-Attention增强**：文本特征与视觉特征进行Cross-Attention交互
- **门控融合机制**：智能融合原始视觉特征和增强特征
- **批量并行处理**：优化的GPU计算，支持大批量处理
- **向后兼容**：保持对传统文本拼接方式的支持

## 配置参数

在 `local_config.py` 中添加以下配置：

```python
# 启用文本增强功能
ENABLE_TEXT_ENHANCEMENT = True

# 文本增强数据库路径
TEXT_ENHANCEMENT_DB_PATH = "/path/to/visual_text_knowledge_base.pkl"

# 使用Cross-Attention方式（推荐）
TEXT_ENHANCEMENT_USE_CROSS_ATTENTION = True

# 文本增强参数
TEXT_ENHANCEMENT_SIMILARITY_THRESHOLD = 0.5  # 相似度阈值
TEXT_ENHANCEMENT_TOP_K = 1  # 每个区域检索的样本数
TEXT_ENHANCEMENT_TOP_SENTENCES = 5  # 选择的句子数

# Cross-Attention参数
TEXT_ENHANCEMENT_CROSS_ATTN_HEADS = 12  # 注意力头数
TEXT_ENHANCEMENT_CROSS_ATTN_DROPOUT = 0.1  # Dropout率
TEXT_ENHANCEMENT_FUSION_WEIGHT = 0.3  # 固定融合权重（α值）
```

## 工作流程

1. **特征提取**：从ViT输出中提取区域视觉特征 `[B, 29, 768]`
2. **文本检索**：基于视觉相似度检索相关文本描述 - **每个解剖区域选择top1**
3. **文本编码**：将检索到的文本编码为特征 `[B, 29, 768]`（与视觉特征维度完全匹配）
4. **Cross-Attention**：
   - 文本特征做query，视觉特征做key&value
   - 输出增强的区域特征 `[B, 29, 768]`
5. **固定权重融合**：`α * enhanced_visual + (1-α) * original_visual`
6. **模型输入**：将增强后的视觉特征输入到生成模型（原有线性层已存在）

## 对比传统方法

| 方法 | 文本处理 | 视觉增强 | 计算效率 |
|------|----------|----------|----------|
| 传统拼接 | 文本直接拼接到history | 无 | 中等 |
| Cross-Attention | 文本编码为特征 | 通过注意力机制增强 | 高 |

## 代码结构

### 主要文件修改

1. **models/text_enhancement.py**
   - 添加 `extract_text_features()` 方法
   - 修改 `forward()` 方法支持返回特征

2. **models/moe_model.py**
   - 添加Cross-Attention模块初始化
   - 添加 `apply_text_enhancement()` 方法
   - 修改forward流程集成文本增强

3. **configs/default_config.py**
   - 添加文本增强相关配置参数

### 关键类和方法

```python
# 文本特征提取 - 每个区域top1
text_features, valid_mask = text_enhancer.extract_text_features(
    visual_features=region_features,  # [B, 29, 768]
    query_image_ids=image_ids,
    similarity_threshold=0.5
)  # 返回: text_features [B, 29, 768], valid_mask [B, 29]

# Cross-Attention增强
enhanced_visual_features = model.apply_text_enhancement(
    visual_features=visual_features,  # [B, 30, 768]
    region_features=region_features,  # [B, 29, 768]
    image_ids=image_ids
)  # 返回: enhanced_visual_features [B, 30, 768]
```

## 注意事项

- 需要预先构建文本增强数据库 (`visual_text_knowledge_base.pkl`)
- 建议在FINETUNE阶段使用，避免影响预训练
- 可以通过配置参数灵活调整增强强度
- 第一次运行时会自动初始化嵌入层（如果tokenizer不支持）

## 性能优化

- 使用GPU缓存减少重复计算
- 批量tokenization提高编码效率
- 智能采样控制内存使用
- 只对有效样本进行Cross-Attention计算 