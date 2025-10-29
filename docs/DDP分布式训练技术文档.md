# DistributedDataParallel (DDP) 分布式训练技术文档

## 目录
1. [DDP 核心原理](#1-ddp-核心原理)
2. [单卡 vs DDP 代码对比](#2-单卡-vs-ddp-代码对比)
3. [必须的代码改动及原理](#3-必须的代码改动及原理)
4. [本项目的实现细节](#4-本项目的实现细节)
5. [性能优化原理](#5-性能优化原理)
6. [常见问题与陷阱](#6-常见问题与陷阱)

---

## 1. DDP 核心原理

### 1.1 什么是 DDP？

**DistributedDataParallel (DDP)** 是 PyTorch 的分布式数据并行训练方案：
- 每个 GPU 对应一个**独立的进程**
- 每个进程拥有**完整的模型副本**
- 每个进程处理**不同的数据批次**
- 通过 **NCCL** (NVIDIA Collective Communications Library) 高效同步梯度

### 1.2 DDP vs DataParallel

| 特性 | DataParallel (DP) | DistributedDataParallel (DDP) |
|------|-------------------|-------------------------------|
| **进程模型** | 单进程多线程 | 多进程 |
| **通信方式** | Python GIL 限制 | NCCL，无 GIL |
| **GPU 负载** | GPU 0 负载重 | 所有 GPU 均衡 |
| **通信效率** | 低（CPU 瓶颈） | 高（GPU 直连） |
| **扩展性** | 仅单机 | 单机多卡 + 多机多卡 |
| **速度** | 慢 | **快 20-30%+** |

### 1.3 DDP 工作流程

```
初始化阶段:
┌─────────────────────────────────────────────────┐
│ 1. torchrun 启动 N 个进程 (N = GPU数量)          │
│    - 每个进程获得唯一的 RANK 和 LOCAL_RANK       │
│    - 设置环境变量: WORLD_SIZE, RANK, LOCAL_RANK  │
└─────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────┐
│ 2. 每个进程初始化 process group                  │
│    - dist.init_process_group(backend='nccl')   │
│    - 建立进程间通信通道                          │
└─────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────┐
│ 3. 每个进程加载完整模型到对应的 GPU              │
│    - Process 0 → GPU 0                         │
│    - Process 1 → GPU 1                         │
│    - ...                                       │
└─────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────┐
│ 4. 用 DDP 包装模型                               │
│    - model = DDP(model, device_ids=[local_rank])│
│    - 注册梯度同步钩子                            │
└─────────────────────────────────────────────────┘

训练阶段 (每个iteration):
┌─────────────────────────────────────────────────┐
│ 1. 数据加载 (DistributedSampler)                │
│    - 每个进程加载不同的数据子集                  │
│    - Process 0: batch [0, 4, 8, ...]           │
│    - Process 1: batch [1, 5, 9, ...]           │
│    - 确保没有数据重复或遗漏                      │
└─────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────┐
│ 2. 前向传播 (各进程独立)                        │
│    - 每个进程在自己的数据上计算 loss             │
│    - 完全并行，无通信                            │
└─────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────┐
│ 3. 反向传播 (自动梯度同步)                       │
│    - 每个进程计算局部梯度                        │
│    - DDP 自动触发 AllReduce 操作                │
│    - 通过 NCCL 高效同步所有进程的梯度            │
│    - 梯度平均: grad_avg = sum(grads) / N        │
└─────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────┐
│ 4. 参数更新 (各进程独立但同步)                   │
│    - 每个进程用相同的平均梯度更新参数            │
│    - 由于初始参数相同 + 梯度相同                 │
│      → 更新后的参数也相同                        │
│    - 保证所有进程的模型状态一致                  │
└─────────────────────────────────────────────────┘
```

### 1.4 梯度同步原理 (AllReduce)

DDP 使用 **Ring-AllReduce** 算法高效同步梯度：

```
假设有 4 个 GPU，每个有梯度 g0, g1, g2, g3

传统方式 (参数服务器):
GPU 0 ─┐
GPU 1 ─┤→ Server → 计算平均 → 广播回所有GPU
GPU 2 ─┤
GPU 3 ─┘
问题: Server 是瓶颈

Ring-AllReduce (DDP 使用):
┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐
│GPU 0│ → │GPU 1│ → │GPU 2│ → │GPU 3│
└─────┘ ← └─────┘ ← └─────┘ ← └─────┘
   ↑                              ↓
   └──────────────────────────────┘

优势:
- 每个 GPU 只与相邻 GPU 通信
- 通信和计算可以重叠
- 带宽利用率高
- 无中心节点瓶颈
```

---

## 2. 单卡 vs DDP 代码对比

### 2.1 启动方式

**单卡训练:**
```bash
python train.py
```

**DDP 训练:**
```bash
torchrun --nproc_per_node=4 train.py
# 或
python -m torch.distributed.launch --nproc_per_node=4 train.py
```

**原理**: `torchrun` 会：
1. 启动 4 个独立的 Python 进程
2. 为每个进程设置环境变量:
   ```
   RANK=0, LOCAL_RANK=0, WORLD_SIZE=4  # 进程 0
   RANK=1, LOCAL_RANK=1, WORLD_SIZE=4  # 进程 1
   ...
   ```
3. 每个进程执行相同的 `train.py` 代码

### 2.2 设备初始化

**单卡训练:**
```python
device = torch.device("cuda:0")
```

**DDP 训练:**
```python
# 从环境变量获取进程信息
local_rank = int(os.environ.get('LOCAL_RANK', 0))
rank = int(os.environ.get('RANK', 0))
world_size = int(os.environ.get('WORLD_SIZE', 1))

# 初始化进程组 (建立进程间通信)
dist.init_process_group(
    backend='nccl',        # GPU 通信使用 NCCL
    init_method='env://',  # 从环境变量读取配置
    world_size=world_size,
    rank=rank
)

# 每个进程绑定到对应的 GPU
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")
```

**原理**:
- `init_process_group`: 建立进程间的通信通道（通过共享内存或网络）
- `backend='nccl'`: 使用 NVIDIA 的 NCCL 库，针对 GPU 优化的集合通信
- `set_device`: 确保每个进程只使用分配给它的 GPU

### 2.3 模型创建

**单卡训练:**
```python
model = MyModel()
model = model.to(device)
```

**DDP 训练:**
```python
model = MyModel()
model = model.to(device)

# 用 DDP 包装模型
model = DistributedDataParallel(
    model,
    device_ids=[local_rank],
    output_device=local_rank,
    find_unused_parameters=True  # 如果有未使用的参数
)
```

**原理**:
- DDP 包装会在模型的每个参数上注册**梯度同步钩子**
- 反向传播时，梯度计算完成后自动触发 AllReduce
- 同步是**异步**的，与计算重叠，提高效率

### 2.4 数据加载器

**单卡训练:**
```python
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,  # 随机打乱
    num_workers=4
)
```

**DDP 训练:**
```python
from torch.utils.data.distributed import DistributedSampler

# 创建分布式采样器
train_sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,  # 总进程数
    rank=rank,                # 当前进程 ID
    shuffle=True              # 每个 epoch 打乱
)

train_loader = DataLoader(
    dataset,
    batch_size=32,            # 注意：这是每个进程的 batch size
    sampler=train_sampler,    # 使用分布式采样器
    shuffle=False,            # sampler 和 shuffle 互斥
    num_workers=4
)
```

**原理**:
- `DistributedSampler` 会将数据集**分片**给不同的进程
- 例如: 1000 个样本，4 个进程
  ```
  Process 0: indices [0, 4, 8, 12, ...]    (250 个)
  Process 1: indices [1, 5, 9, 13, ...]    (250 个)
  Process 2: indices [2, 6, 10, 14, ...]   (250 个)
  Process 3: indices [3, 7, 11, 15, ...]   (250 个)
  ```
- 保证:
  - ✅ 每个样本只被一个进程处理（无重复）
  - ✅ 所有样本都被处理（无遗漏）
  - ✅ 每个 epoch 自动打乱

**重要**: 总的有效 batch size = `batch_size × world_size`

### 2.5 训练循环

**单卡训练:**
```python
for epoch in range(num_epochs):
    for batch in train_loader:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**DDP 训练:**
```python
for epoch in range(num_epochs):
    # 每个 epoch 开始时设置 epoch（用于打乱）
    train_sampler.set_epoch(epoch)
    
    for batch in train_loader:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()  # DDP 在这里自动同步梯度！
        optimizer.step()
```

**原理**:
- `sampler.set_epoch(epoch)`: 确保每个 epoch 的打乱方式不同
- `loss.backward()`: DDP 的钩子会在梯度计算完成后自动触发 AllReduce

### 2.6 模型保存和加载

**单卡训练:**
```python
# 保存
torch.save(model.state_dict(), 'checkpoint.pth')

# 加载
model.load_state_dict(torch.load('checkpoint.pth'))
```

**DDP 训练:**
```python
# 保存 (只在主进程保存，避免冲突)
if rank == 0:
    # 注意: DDP 模型的参数在 model.module 中
    torch.save(model.module.state_dict(), 'checkpoint.pth')

# 加载 (所有进程都加载)
state_dict = torch.load('checkpoint.pth', map_location=device)
model.module.load_state_dict(state_dict)

# 同步所有进程（确保都加载完成）
dist.barrier()
```

**原理**:
- DDP 包装后，原始模型在 `model.module` 中
- 只需主进程保存（避免多个进程同时写同一文件）
- 所有进程都要加载（保持模型一致）
- `barrier()`: 同步屏障，确保所有进程都到达这一点

### 2.7 日志和打印

**单卡训练:**
```python
print(f"Epoch {epoch}, Loss: {loss.item()}")
logger.info(f"Epoch {epoch}, Loss: {loss.item()}")
```

**DDP 训练:**
```python
# 方法1: 只在主进程打印
if rank == 0:
    print(f"Epoch {epoch}, Loss: {loss.item()}")
    logger.info(f"Epoch {epoch}, Loss: {loss.item()}")

# 方法2: 重写 print 函数（本项目使用）
def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print
    
    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    
    __builtin__.print = print

setup_for_distributed(rank == 0)

# 之后 print 会自动只在主进程输出
print(f"Epoch {epoch}, Loss: {loss.item()}")  # 自动只在 rank 0 输出
```

**原理**:
- 避免 N 个进程同时打印，造成输出混乱
- 主进程（rank 0）负责输出和记录
- 其他进程静默运行

---

## 3. 必须的代码改动及原理

### 3.1 改动点总结

| 改动项 | 单卡 | DDP | 原因 |
|--------|------|-----|------|
| **启动方式** | `python` | `torchrun` | 需要启动多个进程 |
| **进程初始化** | 无 | `init_process_group()` | 建立进程通信 |
| **设备绑定** | `cuda:0` | `cuda:{local_rank}` | 每个进程绑定不同 GPU |
| **模型包装** | 直接使用 | `DDP(model)` | 注册梯度同步钩子 |
| **数据采样** | 普通 Sampler | `DistributedSampler` | 数据分片，避免重复 |
| **Batch Size** | 实际值 | 单卡值×GPU数 | 每个进程独立 batch |
| **Epoch 设置** | 无需 | `sampler.set_epoch()` | 每个 epoch 不同打乱 |
| **模型保存** | 直接保存 | 只主进程保存 `module` | 避免冲突 |
| **日志输出** | 直接打印 | 只主进程打印 | 避免重复 |
| **同步点** | 无需 | `dist.barrier()` | 关键点同步 |

### 3.2 关键原理详解

#### 3.2.1 为什么需要 DistributedSampler？

**问题**: 如果不用 DistributedSampler 会怎样？

```python
# 错误示例：所有进程使用相同的 DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 结果：
# - 每个进程都会遍历完整数据集
# - 每个样本被处理 N 次（N = GPU数量）
# - 训练速度没有提升
# - 浪费计算资源
```

**解决**: DistributedSampler 将数据集分片

```python
# 正确示例
sampler = DistributedSampler(dataset, num_replicas=4, rank=0)
loader = DataLoader(dataset, batch_size=32, sampler=sampler)

# 结果：
# - 每个进程只处理 1/N 的数据
# - 每个样本只被处理 1 次
# - 训练速度提升 N 倍（理论上）
```

#### 3.2.2 为什么需要包装成 DDP？

**DDP 做了什么**:

```python
class DistributedDataParallel(nn.Module):
    def __init__(self, module, device_ids, ...):
        super().__init__()
        self.module = module
        
        # 1. 注册梯度同步钩子
        for param in self.module.parameters():
            param.register_hook(self._gradient_sync_hook)
        
        # 2. 初始化通信 bucket（提高效率）
        self._setup_buckets()
    
    def _gradient_sync_hook(self, grad):
        """梯度计算完成后自动触发"""
        # 将梯度放入 bucket
        self._add_to_bucket(grad)
        
        # 如果 bucket 满了，触发 AllReduce
        if self._bucket_is_full():
            self._all_reduce_bucket()
    
    def forward(self, *args, **kwargs):
        # 前向传播与单卡完全一样
        return self.module(*args, **kwargs)
```

**关键优化**:
- **梯度分桶**: 不是每个梯度单独同步，而是攒一批一起同步，减少通信次数
- **通信计算重叠**: 梯度计算和通信可以并行，不需要等待
- **自动化**: 用户无需手动调用同步，DDP 自动处理

#### 3.2.3 为什么 Batch Size 的理解很重要？

**常见误区**:
```python
# 配置
batch_size = 64
num_gpus = 4

# 误以为：总 batch size = 64
# 实际上：总 batch size = 64 × 4 = 256
```

**正确理解**:
```python
# DDP 训练
batch_size_per_gpu = 16  # 每个 GPU 处理 16 个样本
num_gpus = 4
total_batch_size = 16 × 4 = 64  # 总共处理 64 个样本

# 等效于单卡训练
batch_size_single_gpu = 64
```

**影响**:
- 学习率调整: `lr_ddp = lr_single × num_gpus`（线性缩放规则）
- 显存占用: 每个 GPU 只需要支持单卡 batch size
- 收敛速度: 大 batch size 可能影响收敛

#### 3.2.4 为什么需要 set_epoch？

**问题**:
```python
# 如果不调用 set_epoch
for epoch in range(10):
    for batch in loader:
        train(batch)

# 结果：
# - 每个 epoch 的数据顺序完全一样
# - 失去了 shuffle 的效果
# - 影响模型收敛
```

**原理**:
```python
class DistributedSampler:
    def __iter__(self):
        # 使用 epoch 作为随机种子
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)
        
        # 基于种子打乱数据
        indices = torch.randperm(len(self.dataset), generator=g)
        
        # 分配给当前进程
        indices = indices[self.rank::self.num_replicas]
        return iter(indices)
```

所以需要:
```python
for epoch in range(10):
    sampler.set_epoch(epoch)  # 改变随机种子
    for batch in loader:
        train(batch)
```

---

## 4. 本项目的实现细节

### 4.1 DeviceManager 类

本项目封装了 `DeviceManager` 类来统一处理单卡/多卡逻辑：

```python
class DeviceManager:
    def __init__(self, config):
        self.config = config
        self._setup_device()
    
    def _setup_device(self):
        """自动检测并配置设备"""
        # 1. 检查配置
        use_cuda = getattr(self.config, 'USE_CUDA', True)
        visible_devices = getattr(self.config, 'CUDA_VISIBLE_DEVICES', "0")
        use_distributed = getattr(self.config, 'USE_DISTRIBUTED', False)
        
        # 2. 设置环境变量
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
        
        # 3. 获取 GPU 数量
        num_gpus = torch.cuda.device_count()
        
        # 4. 根据配置选择模式
        if num_gpus > 1 and use_distributed:
            self._setup_distributed()  # DDP 模式
        elif num_gpus > 1:
            self.multi_gpu = True      # DataParallel 模式
        else:
            self.device = torch.device("cuda:0")  # 单卡
    
    def _setup_distributed(self):
        """初始化 DDP"""
        # 从环境变量读取（torchrun 设置）
        self.local_rank = int(os.environ.get('LOCAL_RANK', -1))
        self.rank = int(os.environ.get('RANK', -1))
        self.world_size = int(os.environ.get('WORLD_SIZE', -1))
        
        if self.local_rank == -1:
            print("警告：未检测到 torchrun，回退到 DataParallel")
            self.distributed = False
            return
        
        # 初始化进程组
        dist.init_process_group(backend='nccl')
        
        # 绑定 GPU
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f"cuda:{self.local_rank}")
        self.distributed = True
    
    def wrap_model(self, model):
        """包装模型"""
        model = model.to(self.device)
        
        if self.distributed:
            model = DDP(model, device_ids=[self.local_rank])
        elif self.multi_gpu:
            model = DataParallel(model)
        
        return model
    
    def get_sampler(self, dataset, shuffle=True):
        """获取合适的采样器"""
        if self.distributed:
            return DistributedSampler(dataset, shuffle=shuffle)
        else:
            return None  # 使用默认采样器
```

**优势**:
- 统一接口，代码简洁
- 自动适配单卡/多卡
- 支持 DDP 和 DataParallel
- 错误处理和回退机制

### 4.2 日志系统改造

```python
def setup_logger(log_dir="logs", is_main_process=True):
    """
    创建 logger
    
    Args:
        is_main_process: 是否为主进程（DDP 中只有主进程输出到控制台）
    """
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    # 文件处理器（所有进程都写）
    file_handler = logging.FileHandler(log_file)
    logger.addHandler(file_handler)
    
    # 控制台处理器（只有主进程输出）
    if is_main_process:
        console_handler = logging.StreamHandler()
        logger.addHandler(console_handler)
    
    return logger
```

配合 `setup_for_distributed`:
```python
def setup_for_distributed(is_master):
    """重写 print 函数，只在主进程打印"""
    import builtins as __builtin__
    builtin_print = __builtin__.print
    
    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    
    __builtin__.print = print
```

**效果**:
- ✅ `logger.info()`: 只在主进程输出到控制台
- ✅ `print()`: 只在主进程输出
- ✅ 避免日志重复

### 4.3 训练流程

```python
def main():
    # 1. 初始化设备（必须最先执行）
    device_manager = DeviceManager(config)
    
    # 2. 设置打印（只有主进程打印）
    setup_for_distributed(device_manager.is_main_process())
    
    # 3. 创建 logger（只有主进程输出到控制台）
    logger = setup_logger(is_main_process=device_manager.is_main_process())
    
    # 4. 创建数据集
    train_dataset = MyDataset(...)
    
    # 5. 创建采样器（自动适配 DDP）
    train_sampler = device_manager.get_sampler(train_dataset, shuffle=True)
    
    # 6. 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,  # 单卡 batch size
        sampler=train_sampler,
        shuffle=(train_sampler is None)  # 有 sampler 时不能 shuffle
    )
    
    # 7. 创建模型
    model = MyModel()
    
    # 8. 包装模型（自动适配 DDP）
    model = device_manager.wrap_model(model)
    
    # 9. 训练循环
    for epoch in range(num_epochs):
        # 设置 epoch（DDP 需要）
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # 训练
        for batch in train_loader:
            # ... 标准训练流程
            pass
        
        # 保存模型（只主进程保存）
        if device_manager.is_main_process():
            save_checkpoint(model, epoch)
```

---

## 5. 性能优化原理

### 5.1 为什么 DDP 比 DataParallel 快？

#### DataParallel 的瓶颈:

```
Step 1: 将数据分发到各 GPU
CPU ─┬─> GPU 0
     ├─> GPU 1
     ├─> GPU 2
     └─> GPU 3

Step 2: 前向传播（并行）
GPU 0: forward()
GPU 1: forward()
GPU 2: forward()
GPU 3: forward()

Step 3: 将输出收集到 GPU 0
GPU 0 ←┬─ GPU 1
       ├─ GPU 2
       └─ GPU 3

Step 4: 在 GPU 0 计算 loss 和梯度
GPU 0: loss.backward()  # 只在 GPU 0！

Step 5: 将梯度分发到各 GPU
GPU 0 ─┬─> GPU 1
       ├─> GPU 2
       └─> GPU 3

Step 6: 各 GPU 更新参数
GPU 0: optimizer.step()
GPU 1: optimizer.step()
GPU 2: optimizer.step()
GPU 3: optimizer.step()
```

**问题**:
- ❌ GPU 0 负载重（loss计算、梯度计算都在 GPU 0）
- ❌ 频繁的 CPU-GPU 通信
- ❌ Python GIL 限制多线程性能
- ❌ 通信串行化

#### DDP 的优势:

```
Step 1: 每个进程独立加载数据到自己的 GPU
Process 0 → GPU 0: batch_0
Process 1 → GPU 1: batch_1
Process 2 → GPU 2: batch_2
Process 3 → GPU 3: batch_3

Step 2: 各 GPU 独立前向传播
GPU 0: forward(batch_0) → loss_0
GPU 1: forward(batch_1) → loss_1
GPU 2: forward(batch_2) → loss_2
GPU 3: forward(batch_3) → loss_3

Step 3: 各 GPU 独立反向传播
GPU 0: loss_0.backward() → grads_0
GPU 1: loss_1.backward() → grads_1
GPU 2: loss_2.backward() → grads_2
GPU 3: loss_3.backward() → grads_3

Step 4: GPU 间直接同步梯度（NCCL, Ring-AllReduce）
GPU 0 ←→ GPU 1 ←→ GPU 2 ←→ GPU 3
  ↑                              ↓
  └──────────────────────────────┘

Step 5: 各 GPU 独立更新参数
GPU 0: optimizer.step()
GPU 1: optimizer.step()
GPU 2: optimizer.step()
GPU 3: optimizer.step()
```

**优势**:
- ✅ 所有 GPU 负载均衡
- ✅ GPU 间直接通信（NVLink），无需经过 CPU
- ✅ 多进程，无 GIL 限制
- ✅ 通信和计算重叠

### 5.2 梯度同步的优化技巧

#### 技巧 1: Gradient Bucketing

```python
# 不优化：每个参数的梯度单独同步
for param in model.parameters():
    dist.all_reduce(param.grad)  # 1000 次通信！

# DDP 优化：梯度分桶
bucket_size = 25MB
buckets = []
current_bucket = []
current_size = 0

for param in model.parameters():
    current_bucket.append(param.grad)
    current_size += param.grad.numel() * param.grad.element_size()
    
    if current_size >= bucket_size:
        # 合并bucket中的梯度，一次性同步
        dist.all_reduce(torch.cat(current_bucket))
        buckets.append(current_bucket)
        current_bucket = []
        current_size = 0

# 结果：通信次数从 1000 次降到约 40 次
```

#### 技巧 2: 通信与计算重叠

```python
# 反向传播过程
def backward():
    # 从输出层往输入层计算梯度
    for layer in reversed(layers):
        layer.backward()  # 计算梯度
        
        # 一旦梯度计算完成，立即启动同步（不等待）
        if layer.grad_ready:
            async_all_reduce(layer.grad)  # 异步通信
    
    # 等待所有通信完成
    wait_all()

# 效果：梯度计算和通信并行，节省时间
```

#### 技巧 3: 混合精度训练 (AMP)

```python
scaler = torch.cuda.amp.GradScaler()

for batch in loader:
    with torch.cuda.amp.autocast():  # 自动使用 FP16
        output = model(input)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()  # 缩放梯度，避免下溢
    scaler.step(optimizer)
    scaler.update()

# 优势：
# - 减少显存占用 50%
# - 加速计算 2-3 倍（Tensor Core）
# - 减少通信量（梯度更小）
```

---

## 6. 常见问题与陷阱

### 6.1 问题：模型参数不一致

**症状**: 各 GPU 的模型参数逐渐偏离

**原因**:
```python
# 错误：不同进程加载了不同的初始权重
if rank == 0:
    model.load_state_dict(torch.load('checkpoint_0.pth'))
else:
    model.load_state_dict(torch.load('checkpoint_1.pth'))

# 结果：初始状态不同 → 即使梯度同步，最终参数也不同
```

**解决**:
```python
# 所有进程加载相同的权重
checkpoint = torch.load('checkpoint.pth', map_location=device)
model.load_state_dict(checkpoint)

# 确保所有进程都加载完成
dist.barrier()
```

### 6.2 问题：随机数不一致

**症状**: 使用了随机数的操作（如 Dropout）导致各 GPU 不一致

**原因**: 每个进程的随机种子不同

**解决**:
```python
# 为每个进程设置不同但确定的种子
seed = 42
torch.manual_seed(seed + rank)  # 每个进程的种子不同
torch.cuda.manual_seed(seed + rank)

# 这样：
# - 随机性得以保留（不同进程的 Dropout mask 不同）
# - 可重复性得以保证（固定种子）
```

### 6.3 问题：数据集长度不能整除 GPU 数量

**症状**: `DistributedSampler` 报错或数据丢失

**原因**: 
```python
# 例如：1000 个样本，3 个 GPU
# 无法平均分配：1000 / 3 = 333.33...
```

**解决**: DistributedSampler 自动处理（填充或截断）
```python
sampler = DistributedSampler(dataset, drop_last=False)
# drop_last=False: 自动填充，确保每个进程样本数相同
# drop_last=True:  丢弃多余样本
```

### 6.4 问题：find_unused_parameters 警告

**症状**:
```
UserWarning: find_unused_parameters=True was specified but...
```

**原因**: 模型中有些参数在某些forward中不使用

**解决**:
```python
# 方法1: 如果确实有未使用参数
model = DDP(model, find_unused_parameters=True)

# 方法2: 如果所有参数都使用（性能更好）
model = DDP(model, find_unused_parameters=False)

# 方法3: 屏蔽警告
import warnings
warnings.filterwarnings("ignore", message=".*find_unused_parameters.*")
```

### 6.5 问题：hang 住不动

**症状**: 程序卡住，没有任何输出

**原因**: 各进程不同步，某些进程在等待集合通信

**常见场景**:
```python
# 错误：只有主进程执行了集合通信
if rank == 0:
    dist.all_reduce(tensor)  # 其他进程在等待！

# 正确：所有进程都要执行
dist.all_reduce(tensor)  # 所有进程
```

**调试技巧**:
```bash
# 启用 NCCL 调试
export NCCL_DEBUG=INFO
torchrun --nproc_per_node=4 train.py

# 查看每个进程的状态
ps aux | grep python
```

### 6.6 问题：显存不均衡

**症状**: GPU 0 显存占用远高于其他 GPU

**原因**: 可能还在使用 DataParallel 或配置错误

**检查**:
```bash
# 运行时查看 GPU 使用
nvidia-smi -l 1

# 应该看到所有 GPU 显存相近
# 如果 GPU 0 特别高 → 可能是 DataParallel
```

---

## 总结

### DDP 的核心要点

1. **多进程模型**: 每个 GPU 一个独立进程
2. **数据分片**: DistributedSampler 确保数据不重不漏
3. **梯度同步**: 反向传播时自动 AllReduce
4. **参数一致**: 初始状态相同 + 梯度相同 → 参数一致
5. **高效通信**: NCCL + Ring-AllReduce + Bucketing

### 必须的代码改动

| 改动 | 目的 |
|------|------|
| `torchrun` | 启动多进程 |
| `init_process_group` | 建立通信 |
| `DDP` 包装 | 自动梯度同步 |
| `DistributedSampler` | 数据分片 |
| `set_epoch` | 每轮不同打乱 |
| 只主进程保存/打印 | 避免冲突和重复 |

### 性能提升原理

- ✅ GPU 负载均衡（vs DP 的 GPU 0 瓶颈）
- ✅ GPU 直接通信（vs DP 的 CPU 中转）
- ✅ 多进程无 GIL（vs DP 的多线程 GIL）
- ✅ 通信计算重叠（vs DP 的串行）

**结果**: 单机多卡 20-30% 加速，多机多卡线性扩展

