"""内存管理和分析工具"""
import gc
import logging
import torch

# 获取logger
memory_logger = logging.getLogger("train_logger")


def analyze_gpu_memory():
    """分析并打印GPU内存使用情况"""
    if not torch.cuda.is_available():
        memory_logger.warning("CUDA不可用")
        return
    
    for i in range(torch.cuda.device_count()):
        memory_logger.info(f"\n=== GPU {i} 内存分析 ===")
        
        # 总内存
        total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        memory_logger.info(f"总内存: {total_memory:.2f} GB")
        
        # 已分配内存
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        memory_logger.info(f"已分配: {allocated:.2f} GB ({allocated/total_memory*100:.1f}%)")
        
        # 缓存内存
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        memory_logger.info(f"已缓存: {reserved:.2f} GB ({reserved/total_memory*100:.1f}%)")
        
        # 可用内存
        free = total_memory - reserved
        memory_logger.info(f"可用: {free:.2f} GB ({free/total_memory*100:.1f}%)")


def clear_gpu_memory():
    """清理GPU内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_logger.info("GPU内存已清理")


def get_memory_profiler(enable_profile=False, log_path=None):
    """
    获取内存分析器的上下文管理器
    
    Args:
        enable_profile: 是否启用分析
        log_path: 日志保存路径
        
    Returns:
        上下文管理器
    """
    if not enable_profile:
        from contextlib import nullcontext
        return nullcontext()
    
    from torch.profiler import profile, ProfilerActivity
    
    return profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    )


def print_memory_summary():
    """打印详细的内存摘要"""
    if not torch.cuda.is_available():
        return
    
    memory_logger.info("\n" + "=" * 60)
    memory_logger.info("GPU内存摘要")
    memory_logger.info("=" * 60)


def release_process_memory():
    """尝试将空闲内存归还给操作系统（glibc malloc_trim）。

    说明:
        Python/NumPy/Pandas 大量创建/释放对象后，进程可能保留已释放的内存块而不归还给 OS，
        导致工作集持续走高。调用 malloc_trim(0) 可在 glibc 环境下主动归还空闲内存。
    """
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        # 兼容非 glibc 或受限环境，忽略错误
        pass
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i) / 1024**2
        reserved = torch.cuda.memory_reserved(i) / 1024**2
        