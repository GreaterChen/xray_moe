# 解剖区域特征数据库构建指南

## 概述

BUILD_DATABASE阶段用于构建解剖区域特征数据库，该数据库以图像ID为键，存储每个图像检测到的解剖区域特征。这个数据库可用于相似性搜索、检索增强生成(RAG)、区域特征分析等任务。

## 核心功能

- **特征提取**: 使用预训练的目标检测器和ViT模型提取29个解剖区域的768维特征
- **稀疏存储**: 只保存检测到的区域特征，节省存储空间
- **灵活格式**: 支持NPZ（压缩）和Pickle（元数据丰富）两种存储格式
- **图像索引**: 以图像ID为主键，便于按图像查询和分析

## 配置说明

在 `configs/config.py` 中添加以下配置：

```python
# BUILD_DATABASE 阶段配置
PHASE = "BUILD_DATABASE"
DATABASE_OUTPUT_DIR = "results/anatomical_database"  # 数据库输出目录
SAVE_FEATURES_FORMAT = "pkl"  # 保存格式: "npz" 或 "pkl"

# 检查点路径
DETECTION_CHECKPOINT_PATH_FROM = "results/detection/best_model.pth"
VIT_CHECKPOINT_PATH_FROM = "results/vit/best_model.pth"
```

## 数据库结构

### 主要结构（image_id为键）

```python
image_database = {
    "image_id_1": {
        "regions": {
            region_idx: {
                "region_name": str,    # 区域名称
                "feature": np.array    # 768维特征向量
            },
            # ... 其他检测到的区域
        },
        "sample_idx": int,        # 样本索引
        "detected_count": int     # 该图像检测到的区域数量
    },
    # ... 其他图像
}
```

### 存储格式

#### NPZ格式
```python
# 直接存储image_database字典
np.savez_compressed("anatomical_database.npz", image_database=image_database)
```

#### Pickle格式
```python
# 直接存储image_database字典
pickle.dump(image_database, file)
```

## 使用方法

### 1. 运行数据库构建

```bash
cd /path/to/your/project
python train_full.py
```

确保配置文件中设置了正确的：
- `PHASE = "BUILD_DATABASE"`
- 检测器和ViT模型的检查点路径
- 输出目录和保存格式

### 2. 加载和使用数据库

```python
from examples.use_anatomical_database import AnatomicalDatabase

# 初始化数据库
db = AnatomicalDatabase("path/to/anatomical_database.pkl")

# 获取特定图像的数据
image_data = db.get_image_data("your_image_id")
print(f"检测到的区域数: {image_data['detected_count']}")

# 获取特定图像的特定区域特征
feature = db.get_image_region_feature("your_image_id", region_idx=5)
if feature is not None:
    print(f"区域5的特征维度: {feature.shape}")

# 基于特定区域查找相似图像
similar_images = db.find_similar_images_by_region(
    query_image_id="your_image_id", 
    region_idx=5, 
    top_k=10
)
for img_id, similarity in similar_images:
    print(f"相似图像: {img_id}, 相似度: {similarity:.4f}")

# 基于所有区域查找相似图像
similar_images = db.find_similar_images_by_all_regions(
    query_image_id="your_image_id", 
    top_k=10
)
for img_id, similarity, shared_regions in similar_images:
    print(f"相似图像: {img_id}, 相似度: {similarity:.4f}, 共同区域: {shared_regions}")
```

### 3. 数据库分析

```python
# 获取数据库统计信息
info = db.get_database_info()
print(f"总图像数: {info['total_images']}")

# 获取各区域检测统计
region_stats = db.get_region_statistics()
for region_idx, stats in region_stats.items():
    print(f"区域 {region_idx}: {stats['region_name']}, "
          f"检测率: {stats['detection_rate']:.1%}")
```

## 文件结构

构建完成后，输出目录包含：

```
results/anatomical_database/
├── anatomical_database.pkl        # 主数据库文件（如果选择pkl格式）
└── anatomical_database.npz        # 主数据库文件（如果选择npz格式）
```

## 应用场景

### 1. 相似性搜索
- 基于解剖区域特征查找相似的医学图像
- 支持单区域或多区域组合搜索

### 2. 检索增强生成(RAG)
- 为报告生成提供相似病例参考
- 增强临床决策支持

### 3. 数据分析
- 分析不同解剖区域的检测分布
- 研究区域间的特征相关性

### 4. 质量控制
- 识别异常或低质量的图像
- 数据集清洗和标注验证

## 性能优化

### 存储优化
- **稀疏存储**: 只保存检测到的区域，大幅减少存储空间
- **格式选择**: NPZ适合大规模部署，Pickle适合开发分析
- **压缩**: NPZ格式自动压缩，进一步减少文件大小

### 查询优化
- **向量化计算**: 使用numpy和sklearn进行批量相似度计算
- **内存映射**: 大文件支持内存映射加载
- **索引结构**: 按图像ID快速定位

## 注意事项

1. **内存使用**: 大规模数据集建议使用NPZ格式并启用内存映射
2. **特征质量**: 确保检测器和ViT模型已充分训练
3. **数据一致性**: 保持与训练数据相同的预处理流程
4. **版本兼容**: 记录模型版本和配置，确保可重现性

## 故障排除

### 常见问题

**问题**: 内存不足
**解决**: 减少批次大小或使用NPZ格式

**问题**: 特征质量差
**解决**: 检查预训练模型质量，确保检测器和ViT充分训练

**问题**: 加载速度慢
**解决**: 使用NPZ格式并启用内存映射

### 日志分析

构建过程中关注以下日志信息：
- 平均每样本检测区域数（正常范围：5-15个）
- 各区域检测率分布
- 内存使用情况和构建速度

## 扩展功能

数据库构建完成后，可以进一步扩展：

1. **特征降维**: 使用PCA或t-SNE进行可视化分析
2. **聚类分析**: 基于区域特征进行无监督聚类
3. **异常检测**: 识别特征分布异常的图像
4. **跨模态检索**: 结合文本特征进行多模态搜索 