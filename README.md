### 运行tensorboard
在终端执行
```
tensorboard --logdir=runs --port=6006 --bind_all
```
然后访问
```
192.168.7.232:6006
```

rsync -avz --progress --remove-source-files /path/to/source/ /path/to/destination/

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
```

## 工作流程

1. **特征提取**：从ViT输出中提取区域视觉特征 `[B, 29, 768]`
2. **文本检索**：基于视觉相似度检索相关文本描述
3. **文本编码**：将检索到的文本编码为特征 `[B, 5, 768]`
4. **Cross-Attention**：视觉特征attend to文本特征
5. **特征融合**：通过门控机制融合原始和增强特征
6. **模型输入**：将增强后的视觉特征输入到生成模型

## 对比传统方法

| 方法 | 文本处理 | 视觉增强 | 计算效率 |
|------|----------|----------|----------|
| 传统拼接 | 文本直接拼接到history | 无 | 中等 |
| Cross-Attention | 文本编码为特征 | 通过注意力机制增强 | 高 |

## 注意事项

- 需要预先构建文本增强数据库 (`visual_text_knowledge_base.pkl`)
- 建议在FINETUNE阶段使用，避免影响预训练
- 可以通过配置参数灵活调整增强强度