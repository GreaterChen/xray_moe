"""检查点管理工具"""
import os
import logging
import torch

# 获取logger
checkpoint_logger = logging.getLogger("train_logger")


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
            checkpoint_logger.warning("警告: scheduler没有state_dict方法，无法保存scheduler状态")
    
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


def load(path, model, optimizer=None, scheduler=None, load_model="object_detector", device=None):
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
        device: 目标设备（可选）。如果不提供，默认使用CPU
            
    Returns:
        (epoch, stats) 元组
    """
    # 如果没有指定设备，默认使用CPU（避免设备不匹配错误）
    if device is None:
        device = torch.device('cpu')
    checkpoint = torch.load(path, weights_only=False, map_location=device)
    epoch = checkpoint.get("epoch", -1)
    stats = checkpoint.get("stats", None)
    
    if "model_state_dict" not in checkpoint:
        checkpoint_logger.error("检查点中没有找到模型状态字典！")
        return epoch, stats
    
    checkpoint_state_dict = checkpoint["model_state_dict"]
    
    # 根据load_model参数提取相应的权重
    filtered_state_dict = _filter_state_dict(checkpoint_state_dict, load_model)
    
    # 智能适配 'module.' 前缀（处理单卡/多卡互相加载的情况）
    filtered_state_dict = _adapt_module_prefix(filtered_state_dict, model)
    
    # 加载state_dict到模型
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    
    # 打印加载信息
    _print_load_info(missing_keys, unexpected_keys, model)
    
    # 加载优化器
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception as e:
            checkpoint_logger.error(f"无法加载优化器: {e}")
    
    # 加载scheduler
    if scheduler is not None:
        _load_scheduler(scheduler, checkpoint, epoch)
    
    return epoch, stats


def _filter_state_dict(checkpoint_state_dict, load_model):
    """根据load_model参数过滤state_dict"""
    
    if load_model == "object_detector":
        checkpoint_logger.info("加载目标检测器参数...")
        return _extract_prefix(checkpoint_state_dict, "detector.")
        
    elif load_model == "vit":
        checkpoint_logger.info("加载ViT图像编码器参数...")
        return _extract_prefix(checkpoint_state_dict, "image_encoder.")
        
    elif load_model == "decoder":
        checkpoint_logger.info("加载报告生成解码器参数...")
        # 尝试两种前缀
        filtered = _extract_prefix(checkpoint_state_dict, "findings_decoder.decoder.")
        if not filtered:
            filtered = _extract_prefix(checkpoint_state_dict, "findings_decoder.")
        if not filtered:
            checkpoint_logger.warning("警告：在检查点中未找到解码器权重！")
        return filtered
        
    elif load_model == "full":
        checkpoint_logger.info("加载完整模型参数...")
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


def _adapt_module_prefix(checkpoint_state_dict, model):
    """
    智能适配'module.'前缀，确保检查点和模型的键名匹配
    
    处理以下情况：
    1. 检查点有'module.'前缀，模型没有 → 去除前缀（多卡保存，单卡加载）
    2. 检查点没有'module.'前缀，模型有 → 添加前缀（单卡保存，多卡加载）
    3. 两者都有或都没有 → 不变
    
    Args:
        checkpoint_state_dict: 检查点的state_dict
        model: 当前模型
        
    Returns:
        适配后的state_dict
    """
    if len(checkpoint_state_dict) == 0:
        return checkpoint_state_dict
    
    # 检查是否在分布式环境中（通过环境变量）
    import os
    is_distributed = (
        int(os.environ.get('WORLD_SIZE', -1)) > 1 or
        int(os.environ.get('RANK', -1)) >= 0 or
        int(os.environ.get('LOCAL_RANK', -1)) >= 0
    )
    
    # 检查检查点的键名是否有 'module.' 前缀
    checkpoint_has_module = any(key.startswith('module.') for key in checkpoint_state_dict.keys())
    
    # 检查模型的键名是否有 'module.' 前缀
    model_state_dict = model.state_dict()
    model_has_module = any(key.startswith('module.') for key in model_state_dict.keys())
    
    # 情况1: 检查点有module前缀，但模型没有
    if checkpoint_has_module and not model_has_module:
        # 无论是单卡还是多卡环境，都需要去除module前缀来匹配当前模型
        # 在分布式环境中，模型稍后会被DDP包装，自动加上module前缀
        if is_distributed:
            checkpoint_logger.info("检测到检查点使用了DDP保存，当前处于分布式环境（模型将被DDP包装），正在适配权重加载...")
        else:
            checkpoint_logger.info("检测到检查点使用了DataParallel/DDP保存，正在适配单卡加载...")
        
        new_state_dict = {}
        for key, value in checkpoint_state_dict.items():
            if key.startswith('module.'):
                new_key = key[7:]  # 去掉 'module.'
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        return new_state_dict
    
    # 情况2: 检查点没有module前缀，但模型有 → 添加前缀
    elif not checkpoint_has_module and model_has_module:
        checkpoint_logger.info("检测到检查点为单卡保存，正在适配DDP加载...")
        new_state_dict = {}
        for key, value in checkpoint_state_dict.items():
            new_key = f'module.{key}'
            new_state_dict[new_key] = value
        return new_state_dict
    
    # 情况3: 两者匹配，不需要修改
    else:
        if checkpoint_has_module and model_has_module:
            checkpoint_logger.info("检测到DDP环境，权重格式匹配...")
        return checkpoint_state_dict


def _print_load_info(missing_keys, unexpected_keys, model):
    """打印加载信息"""
    if len(missing_keys) > 0:
        checkpoint_logger.warning(f"Missing keys ({len(missing_keys)}): {missing_keys[:5]}...")
        if len(missing_keys) > 5:
            checkpoint_logger.warning(f"... 以及其他 {len(missing_keys) - 5} 个缺失的键")
    
    if len(unexpected_keys) > 0:
        checkpoint_logger.warning(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
        if len(unexpected_keys) > 5:
            checkpoint_logger.warning(f"... 以及其他 {len(unexpected_keys) - 5} 个意外的键")
    
    # 计算加载成功率
    total_params = len(model.state_dict())
    loaded_params = total_params - len(missing_keys)
    load_success_rate = loaded_params / total_params * 100 if total_params > 0 else 0
    checkpoint_logger.info(f"权重加载成功率: {load_success_rate:.2f}% ({loaded_params}/{total_params})")


def _load_scheduler(scheduler, checkpoint, epoch):
    """加载scheduler状态"""
    try:
        if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
            if hasattr(scheduler, 'load_state_dict'):
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                checkpoint_logger.info("成功加载scheduler状态")
            else:
                checkpoint_logger.warning("scheduler不支持state_dict，将跳过加载")
        else:
            checkpoint_logger.warning("检查点中没有scheduler状态")
    except Exception as e:
        checkpoint_logger.error(f"恢复scheduler状态失败: {e}")

