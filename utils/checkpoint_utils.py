"""检查点管理工具"""
import os
import torch


def save(path, model, optimizer=None, scheduler=None, epoch=-1, stats=None):
    """
    保存模型检查点
    
    Args:
        path: 保存路径
        model: 模型
        optimizer: 优化器（可选）
        scheduler: 学习率调度器（可选）
        epoch: 当前epoch
        stats: 统计信息
    """
    # 确保目录存在
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    
    # 获取scheduler状态
    scheduler_state = None
    if scheduler is not None:
        try:
            scheduler_state = scheduler.state_dict()
        except AttributeError:
            print("警告: scheduler没有state_dict方法，无法保存scheduler状态")
    
    # 保存检查点
    torch.save(
        {
            "epoch": epoch,
            "stats": stats,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler_state,
        },
        path,
    )


def load(path, model, optimizer=None, scheduler=None, load_model="object_detector"):
    """
    加载模型检查点
    
    Args:
        path: 检查点路径
        model: 模型
        optimizer: 优化器（可选）
        scheduler: 学习率调度器（可选）
        load_model: 加载模式
            - "object_detector": 只加载检测器
            - "vit": 只加载ViT
            - "decoder": 只加载解码器
            - "full": 加载完整模型
            
    Returns:
        (epoch, stats) 元组
    """
    checkpoint = torch.load(path, weights_only=False)
    epoch = checkpoint.get("epoch", -1)
    stats = checkpoint.get("stats", None)
    
    if "model_state_dict" not in checkpoint:
        print("检查点中没有找到模型状态字典！")
        return epoch, stats
    
    checkpoint_state_dict = checkpoint["model_state_dict"]
    
    # 根据load_model参数提取相应的权重
    filtered_state_dict = _filter_state_dict(checkpoint_state_dict, load_model)
    
    # 加载state_dict到模型
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    
    # 打印加载信息
    _print_load_info(missing_keys, unexpected_keys, model)
    
    # 加载优化器
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception as e:
            print(f"无法加载优化器: {e}")
    
    # 加载scheduler
    if scheduler is not None:
        _load_scheduler(scheduler, checkpoint, epoch)
    
    return epoch, stats


def _filter_state_dict(checkpoint_state_dict, load_model):
    """根据load_model参数过滤state_dict"""
    
    if load_model == "object_detector":
        print("加载目标检测器参数...")
        return _extract_prefix(checkpoint_state_dict, "object_detector.")
        
    elif load_model == "vit":
        print("加载ViT图像编码器参数...")
        return _extract_prefix(checkpoint_state_dict, "image_encoder.")
        
    elif load_model == "decoder":
        print("加载报告生成解码器参数...")
        # 尝试两种前缀
        filtered = _extract_prefix(checkpoint_state_dict, "findings_decoder.decoder.")
        if not filtered:
            filtered = _extract_prefix(checkpoint_state_dict, "findings_decoder.")
        if not filtered:
            print("警告：在检查点中未找到解码器权重！")
        return filtered
        
    elif load_model == "full":
        print("加载完整模型参数...")
        return checkpoint_state_dict
        
    else:
        return checkpoint_state_dict


def _extract_prefix(state_dict, prefix):
    """提取带有特定前缀的权重"""
    filtered = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            filtered[new_key] = value
    return filtered


def _print_load_info(missing_keys, unexpected_keys, model):
    """打印加载信息"""
    if len(missing_keys) > 0:
        print(f"Missing keys ({len(missing_keys)}): {missing_keys[:5]}...")
        if len(missing_keys) > 5:
            print(f"... 以及其他 {len(missing_keys) - 5} 个缺失的键")
    
    if len(unexpected_keys) > 0:
        print(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
        if len(unexpected_keys) > 5:
            print(f"... 以及其他 {len(unexpected_keys) - 5} 个意外的键")
    
    # 计算加载成功率
    total_params = len(model.state_dict())
    loaded_params = total_params - len(missing_keys)
    load_success_rate = loaded_params / total_params * 100 if total_params > 0 else 0
    print(f"权重加载成功率: {load_success_rate:.2f}% ({loaded_params}/{total_params})")


def _load_scheduler(scheduler, checkpoint, epoch):
    """加载scheduler状态"""
    try:
        if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
            if hasattr(scheduler, 'load_state_dict'):
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                print("成功加载scheduler状态")
            else:
                print("scheduler不支持state_dict，将跳过加载")
        else:
            print("检查点中没有scheduler状态")
    except Exception as e:
        print(f"恢复scheduler状态失败: {e}")

