"""内存管理和分析工具"""
import gc
import torch


def analyze_gpu_memory():
    """分析并打印GPU内存使用情况"""
    if not torch.cuda.is_available():
        print("CUDA不可用")
        return
    
    for i in range(torch.cuda.device_count()):
        print(f"\n=== GPU {i} 内存分析 ===")
        
        # 总内存
        total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"总内存: {total_memory:.2f} GB")
        
        # 已分配内存
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        print(f"已分配: {allocated:.2f} GB ({allocated/total_memory*100:.1f}%)")
        
        # 缓存内存
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"已缓存: {reserved:.2f} GB ({reserved/total_memory*100:.1f}%)")
        
        # 可用内存
        free = total_memory - reserved
        print(f"可用: {free:.2f} GB ({free/total_memory*100:.1f}%)")


def clear_gpu_memory():
    """清理GPU内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU内存已清理")


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
    
    print("\n" + "=" * 60)
    print("GPU内存摘要")
    print("=" * 60)
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i) / 1024**2
        reserved = torch.cuda.memory_reserved(i) / 1024**2
        
        print(f"\nGPU {i}: {props.name}")
        print(f"  已分配内存: {allocated:.1f} MB")
        print(f"  保留内存: {reserved:.1f} MB")
        print(f"  总内存: {props.total_memory / 1024**2:.1f} MB")
    
    print("=" * 60)

