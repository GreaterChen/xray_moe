# 单机四卡分布式训练指南

## 配置说明

### 1. 设备配置（已配置）
在 `configs/local_config.py` 中：
```python
USE_CUDA = True  # 启用 GPU 训练
CUDA_VISIBLE_DEVICES = "0,1,2,3"  # 使用四张 GPU
USE_DISTRIBUTED = True  # 启用分布式训练
```

### 2. Batch Size 配置（重要）
- **单卡 Batch Size**: 16
- **总有效 Batch Size**: 16 × 4 = 64
- **验证 Batch Size**: 8（单卡）× 4 = 32（总）

如果您的显存充足，可以适当增加单卡的 batch size，例如：
- TRAIN_BATCH_SIZE = 32 → 总batch size = 128
- TRAIN_BATCH_SIZE = 64 → 总batch size = 256

### 3. 数据加载配置
```python
NUM_WORKERS = 16  # 建议设置为 CPU核心数 / GPU数量，避免worker过多
PREFETCH_FACTOR = 4  # 预加载批次数
PERSISTENT_WORKERS = True  # 保持worker存活
PIN_MEMORY = True  # GPU训练时启用
```

## 启动方法

### 方法1：使用启动脚本（推荐）
```bash
./train_distributed.sh
```

### 方法2：直接使用 torchrun
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 --master_port=29500 train.py
```

### 方法3：使用 torch.distributed.launch（旧版本）
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29500 train.py
```

## 常见问题

### 1. 如果只想使用部分GPU（如2张卡）
修改 `configs/local_config.py`：
```python
CUDA_VISIBLE_DEVICES = "0,1"  # 只使用前两张卡
```
然后启动：
```bash
torchrun --nproc_per_node=2 --master_port=29500 train.py
```

### 2. 如果想使用 DataParallel 而不是 DistributedDataParallel
修改配置：
```python
USE_DISTRIBUTED = False  # 使用 DataParallel
```
然后直接运行：
```bash
python train.py  # 不需要 torchrun
```

注意：DistributedDataParallel 通常比 DataParallel 效率更高，推荐使用。

### 3. 端口冲突
如果 29500 端口被占用，修改启动命令中的端口号：
```bash
torchrun --nproc_per_node=4 --master_port=29501 train.py
```

### 4. 显存不足 (OOM)
- 减少单卡 batch size（例如从 16 改为 8）
- 减少 NUM_WORKERS
- 启用梯度累积（需要在代码中实现）

### 5. 检查GPU使用情况
```bash
# 实时监控GPU使用
nvidia-smi -l 1

# 或使用 watch
watch -n 1 nvidia-smi
```

## 性能优化建议

### 1. 学习率调整
分布式训练时，有效 batch size 增大了4倍，建议相应调整学习率：
- 线性缩放：`lr_new = lr_base × 4`
- 或使用 warmup 策略

代码中 `DeviceManager.adjust_learning_rate()` 已实现自动调整。

### 2. 数据加载优化
- `NUM_WORKERS`: 建议设置为 (CPU核心数) / (GPU数量)，避免竞争
- `PREFETCH_FACTOR`: 增加可以减少数据等待时间
- `PIN_MEMORY`: GPU训练时设为 True

### 3. 混合精度训练
代码已支持混合精度训练（AMP），可以节省显存并加速训练。

## 训练监控

### 1. 日志输出
只有主进程（rank 0）会输出日志，避免重复信息。

### 2. 检查分布式是否生效
训练开始时会显示：
```
分布式训练初始化成功 - Rank: 0, Local Rank: 0, World Size: 4
模型已包装为DistributedDataParallel
```

### 3. 验证训练速度
- 单卡训练时间 vs 四卡训练时间
- 理想情况：四卡训练速度 ≈ 单卡的 3-3.5倍（考虑通信开销）

## 保存和加载模型

分布式训练中，只有主进程（rank 0）需要保存模型。代码已经处理：
```python
if device_manager.is_main_process():
    torch.save(model.state_dict(), checkpoint_path)
```

加载模型时，所有进程都会加载相同的权重。

## 故障排查

### 1. 查看分布式初始化是否成功
检查是否有 "分布式训练初始化成功" 的输出。

### 2. 检查环境变量
```bash
echo $CUDA_VISIBLE_DEVICES
echo $WORLD_SIZE
echo $RANK
echo $LOCAL_RANK
```

### 3. 测试GPU通信
```bash
# 确保所有GPU可见
python -c "import torch; print(f'GPU数量: {torch.cuda.device_count()}')"
```

### 4. 如果遇到 NCCL 错误
- 检查GPU之间是否可以通信
- 尝试设置环境变量：`export NCCL_DEBUG=INFO`
- 检查防火墙设置

## 与单GPU训练的差异

| 项目 | 单GPU | 四GPU分布式 |
|------|-------|-------------|
| 启动方式 | `python train.py` | `torchrun --nproc_per_node=4 train.py` |
| 单卡batch size | 64 | 16 |
| 总batch size | 64 | 64 (16×4) |
| 训练速度 | 1x | ~3-3.5x |
| 显存占用 | 单卡 | 分散到4卡 |
| 模型包装 | 无/DataParallel | DistributedDataParallel |

## 参考资料
- [PyTorch分布式训练文档](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [torchrun 使用指南](https://pytorch.org/docs/stable/elastic/run.html)

