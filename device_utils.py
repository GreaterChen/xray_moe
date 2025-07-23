"""
设备管理工具，支持单卡和多卡的自动切换
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel as DP


class DeviceManager:
    """设备管理器，自动处理单卡和多卡配置"""
    
    def __init__(self, config):
        self.config = config
        self.world_size = None
        self.rank = None
        self.local_rank = None
        self.device = None
        self.distributed = False
        self.multi_gpu = False
        
        self._setup_device()
    
    def _setup_device(self):
        """设置设备配置"""
        # 检查可用GPU
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
            print("CUDA不可用，使用CPU")
            return
        
        # 解析CUDA_VISIBLE_DEVICES
        visible_devices = self.config.CUDA_VISIBLE_DEVICES
        if isinstance(visible_devices, str):
            if ',' in visible_devices:
                gpu_ids = [int(x.strip()) for x in visible_devices.split(',')]
            else:
                gpu_ids = [int(visible_devices)]
        else:
            gpu_ids = [visible_devices] if isinstance(visible_devices, int) else visible_devices
        
        # 设置环境变量
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config.CUDA_VISIBLE_DEVICES)
        
        # 获取实际可用的GPU数量
        num_gpus = torch.cuda.device_count()
        print(f"检测到 {num_gpus} 个可用GPU")
        
        if num_gpus == 0:
            self.device = torch.device("cpu")
            print("没有可用的GPU，使用CPU")
        elif num_gpus == 1:
            self.device = torch.device("cuda:0")
            print(f"使用单GPU: {self.device}")
        else:
            # 多GPU情况
            self.multi_gpu = True
            self.device = torch.device("cuda:0")  # 主设备
            
            # 检查是否启用分布式训练
            if getattr(self.config, 'USE_DISTRIBUTED', False):
                self._setup_distributed()
            else:
                print(f"使用DataParallel进行多GPU训练，GPU数量: {num_gpus}")
    
    def _setup_distributed(self):
        """设置分布式训练"""
        try:
            # 从环境变量或配置中获取分布式参数
            self.world_size = int(os.environ.get('WORLD_SIZE', torch.cuda.device_count()))
            self.rank = int(os.environ.get('RANK', 0))
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            
            # 初始化分布式进程组
            if not dist.is_initialized():
                dist.init_process_group(
                    backend='nccl',
                    init_method='env://',
                    world_size=self.world_size,
                    rank=self.rank
                )
            
            # 设置当前进程的GPU
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
            self.distributed = True
            
            print(f"分布式训练初始化成功 - Rank: {self.rank}, Local Rank: {self.local_rank}, World Size: {self.world_size}")
            
        except Exception as e:
            print(f"分布式训练初始化失败: {e}")
            print("回退到DataParallel模式")
            self.distributed = False
    
    def wrap_model(self, model):
        """为模型添加并行包装"""
        model = model.to(self.device)
        
        if self.distributed:
            # 使用DistributedDataParallel
            model = DDP(model, device_ids=[self.local_rank], find_unused_parameters=True)
            print("模型已包装为DistributedDataParallel")
        elif self.multi_gpu:
            # 使用DataParallel
            if torch.cuda.device_count() > 1:
                model = DP(model)
                print(f"模型已包装为DataParallel，使用 {torch.cuda.device_count()} 个GPU")
        
        return model
    
    def get_sampler(self, dataset, shuffle=True):
        """获取适当的数据采样器"""
        if self.distributed:
            from torch.utils.data.distributed import DistributedSampler
            return DistributedSampler(
                dataset, 
                num_replicas=self.world_size, 
                rank=self.rank, 
                shuffle=shuffle
            )
        else:
            return None
    
    def reduce_tensor(self, tensor):
        """减少张量（用于分布式训练中的平均）"""
        if not self.distributed:
            return tensor
        
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.world_size
        return rt
    
    def is_main_process(self):
        """检查是否为主进程"""
        return not self.distributed or self.rank == 0
    
    def barrier(self):
        """同步所有进程"""
        if self.distributed:
            dist.barrier()
    
    def cleanup(self):
        """清理分布式环境"""
        if self.distributed:
            dist.destroy_process_group()
    
    def get_effective_batch_size(self, batch_size):
        """获取有效批次大小"""
        if self.distributed:
            return batch_size * self.world_size
        elif self.multi_gpu:
            return batch_size * torch.cuda.device_count()
        else:
            return batch_size
    
    def adjust_learning_rate(self, lr):
        """根据GPU数量调整学习率"""
        if self.distributed:
            # 线性缩放规则
            return lr * self.world_size
        elif self.multi_gpu:
            return lr * torch.cuda.device_count()
        else:
            return lr
    
    def print_info(self):
        """打印设备信息"""
        if self.is_main_process():
            print("\n=== 设备配置信息 ===")
            print(f"设备: {self.device}")
            print(f"多GPU: {self.multi_gpu}")
            print(f"分布式: {self.distributed}")
            if self.distributed:
                print(f"World Size: {self.world_size}")
                print(f"Rank: {self.rank}")
                print(f"Local Rank: {self.local_rank}")
            elif self.multi_gpu:
                print(f"GPU数量: {torch.cuda.device_count()}")
            print("==================\n")


def to_device(data, device):
    """将数据移动到指定设备，支持嵌套结构"""
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(to_device(item, device) for item in data)
    else:
        return data


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print 