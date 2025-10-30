"""
设备管理工具，支持单卡和多卡的自动切换
"""

import os
import logging
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel as DP
from datetime import timedelta

# 获取logger
device_logger = logging.getLogger("train_logger")


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
        # 检查配置中是否强制使用CPU
        use_cuda = getattr(self.config, 'USE_CUDA', True)
        
        if not use_cuda:
            self.device = torch.device("cpu")
            device_logger.warning("⚠️  配置设置为不使用GPU，强制使用CPU")
            return
        
        # 检查CUDA是否可用
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
            device_logger.error("❌ CUDA不可用，使用CPU")
            return
        
        # 解析CUDA_VISIBLE_DEVICES（在分布式环境下不要覆盖启动器已设置的映射）
        launcher_set_ddp = (
            os.environ.get('LOCAL_RANK') is not None or
            os.environ.get('RANK') is not None or
            os.environ.get('WORLD_SIZE') is not None or
            os.environ.get('SLURM_PROCID') is not None
        )

        visible_devices = getattr(self.config, 'CUDA_VISIBLE_DEVICES', "0")
        if not launcher_set_ddp:
            # 仅在非分布式/未通过launcher启动时，根据配置设置可见GPU
            if isinstance(visible_devices, str):
                if ',' in visible_devices:
                    gpu_ids = [int(x.strip()) for x in visible_devices.split(',')]
                else:
                    gpu_ids = [int(visible_devices)]
            else:
                gpu_ids = [visible_devices] if isinstance(visible_devices, int) else visible_devices
            os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_devices)
            device_logger.info(f"已设置 CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
        else:
            device_logger.info("检测到分布式启动器环境变量（torchrun/SLURM），跳过覆盖 CUDA_VISIBLE_DEVICES")
        
        # 获取实际可用的GPU数量
        num_gpus = torch.cuda.device_count()
        
        if num_gpus == 0:
            self.device = torch.device("cpu")
            device_logger.error("❌ 没有可用的GPU，使用CPU")
        elif num_gpus == 1:
            self.device = torch.device("cuda:0")
            device_logger.info(f"✅ 使用单GPU: {self.device}")
        else:
            # 多GPU情况
            self.multi_gpu = True
            self.device = torch.device("cuda:0")  # 主设备
            
            # 检查是否启用分布式训练
            if getattr(self.config, 'USE_DISTRIBUTED', False):
                self._setup_distributed()
            else:
                device_logger.info(f"✅ 使用DataParallel进行多GPU训练，GPU数量: {num_gpus}")
    
    def _setup_distributed(self):
        """设置分布式训练"""
        try:
            # 从环境变量获取分布式参数
            # torchrun 会自动设置这些环境变量
            self.local_rank = int(os.environ.get('LOCAL_RANK', -1))
            self.rank = int(os.environ.get('RANK', -1))
            self.world_size = int(os.environ.get('WORLD_SIZE', -1))
            
            # 如果环境变量未设置，说明没有使用 torchrun 启动
            if self.local_rank == -1:
                device_logger.warning("⚠️  警告：未检测到分布式环境变量（RANK, LOCAL_RANK, WORLD_SIZE）")
                device_logger.warning("   请使用 torchrun 启动：torchrun --nproc_per_node=4 train.py")
                device_logger.warning("   回退到 DataParallel 模式")
                self.distributed = False
                self.device = torch.device("cuda:0")
                return
            
            # 初始化分布式进程组
            if not dist.is_initialized():
                # 允许通过环境变量调整超时，默认30分钟，避免在网络异常时无限等待
                timeout_seconds = int(os.environ.get('DIST_TIMEOUT', '1800'))
                device_logger.info(
                    f"初始化进程组: backend=nccl, world_size={self.world_size}, rank={self.rank}, "
                    f"master_addr={os.environ.get('MASTER_ADDR')}, master_port={os.environ.get('MASTER_PORT')}, "
                    f"timeout={timeout_seconds}s"
                )
                dist.init_process_group(
                    backend='nccl',
                    init_method='env://',
                    world_size=self.world_size,
                    rank=self.rank,
                    timeout=timedelta(seconds=timeout_seconds)
                )
            
            # 设置当前进程的GPU
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
            self.distributed = True
            
            device_logger.info(
                f"✅ 分布式训练初始化成功 - Rank: {self.rank}/{self.world_size}, Local Rank: {self.local_rank}, Device: {self.device}"
            )
            # 打印关键 NCCL/网络相关环境变量，便于定位网络问题
            if self.is_main_process():
                nccl_vars = {k: os.environ.get(k) for k in [
                    'NCCL_DEBUG', 'NCCL_IB_DISABLE', 'NCCL_SOCKET_IFNAME',
                    'NCCL_P2P_DISABLE', 'NCCL_NET_GDR_LEVEL', 'TORCH_DISTRIBUTED_DEBUG'
                ]}
                device_logger.info(f"NCCL/分布式环境变量: {nccl_vars}")
            
        except Exception as e:
            device_logger.error(f"❌ 分布式训练初始化失败: {e}")
            device_logger.warning("   回退到DataParallel模式")
            self.distributed = False
            self.device = torch.device("cuda:0")
    
    def wrap_model(self, model):
        """为模型添加并行包装"""
        model = model.to(self.device)
        
        if self.distributed:
            # 使用DistributedDataParallel
            # 设置 find_unused_parameters=True 并静态图模式减少开销
            import warnings
            warnings.filterwarnings("ignore", message=".*find_unused_parameters.*")
            model = DDP(
                model, 
                device_ids=[self.local_rank], 
                find_unused_parameters=True,
                # broadcast_buffers=False  # 如果不需要同步buffer可以关闭以提升性能
            )
            device_logger.info("✅ 模型已包装为DistributedDataParallel")
        elif self.multi_gpu:
            # 使用DataParallel
            if torch.cuda.device_count() > 1:
                model = DP(model)
                device_logger.info(f"模型已包装为DataParallel，使用 {torch.cuda.device_count()} 个GPU")
        
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
            device_logger.info("\n=== 设备配置信息 ===")
            device_logger.info(f"USE_CUDA配置: {getattr(self.config, 'USE_CUDA', True)}")
            device_logger.info(f"设备: {self.device}")
            if self.device.type == 'cuda':
                device_logger.info(f"CUDA_VISIBLE_DEVICES: {getattr(self.config, 'CUDA_VISIBLE_DEVICES', '0')}")
                device_logger.info(f"多GPU: {self.multi_gpu}")
                device_logger.info(f"分布式: {self.distributed}")
                if self.distributed:
                    device_logger.info(f"World Size: {self.world_size}")
                    device_logger.info(f"Rank: {self.rank}")
                    device_logger.info(f"Local Rank: {self.local_rank}")
                elif self.multi_gpu:
                    device_logger.info(f"GPU数量: {torch.cuda.device_count()}")
            device_logger.info("==================\n")


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