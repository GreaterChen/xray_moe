"""模型构建工具 - 提取公共的模型构建逻辑"""
import torch
from models.fast_rcnn_classifier import DetectionOnlyFastRCNN, EnhancedFastRCNN
from models.vit import MedicalVisionTransformer
from utils import load


def build_detection_model(config, logger=None, device=None):
    """
    统一的检测模型构建
    
    Args:
        config: 配置对象
        logger: 日志记录器（可选）
        device: 目标设备（可选）
        
    Returns:
        EnhancedFastRCNN实例
    """
    if logger:
        logger.info("加载目标检测器...")
    
    # 加载预训练的检测器
    detection_model = DetectionOnlyFastRCNN()
    _, _ = load(
        config.DETECTION_CHECKPOINT_PATH_FROM,
        detection_model,
        load_model="full",
        device=device
    )
    
    # 创建增强型FastRCNN
    enhanced_rcnn = EnhancedFastRCNN(
        pretrained_detector=detection_model,
        num_regions=29,
        feature_dim=768
    )
    
    if logger:
        logger.info("✅ 目标检测器已加载")
    
    return enhanced_rcnn


def build_vit_model(config, load_pretrained=False, logger=None, device=None):
    """
    统一的ViT模型构建
    
    Args:
        config: 配置对象
        load_pretrained: 是否加载预训练权重
        logger: 日志记录器（可选）
        device: 目标设备（可选）
        
    Returns:
        MedicalVisionTransformer实例
    """
    if logger:
        logger.info("初始化Vision Transformer...")
    
    vit_model = MedicalVisionTransformer(config=config)
    
    # 如果需要加载预训练权重
    if load_pretrained and hasattr(config, 'VIT_CHECKPOINT_PATH_FROM') and \
       config.VIT_CHECKPOINT_PATH_FROM:
        _, _ = load(config.VIT_CHECKPOINT_PATH_FROM, vit_model, load_model="vit", device=device)
        if logger:
            logger.info("✅ ViT预训练权重已加载")
    
    return vit_model


def freeze_model_parameters(model, logger=None, model_name="模型"):
    """
    冻结模型的所有参数
    
    Args:
        model: 要冻结的模型
        logger: 日志记录器（可选）
        model_name: 模型名称（用于日志）
    """
    for param in model.parameters():
        param.requires_grad = False
    
    if logger:
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✅ {model_name}参数已冻结 ({total_params:,} 参数)")


def count_trainable_parameters(model):
    """
    统计模型的可训练参数数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        (trainable_params, total_params) 元组
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return trainable_params, total_params


def log_model_parameters(model, logger, model_name="模型"):
    """
    记录模型参数统计信息
    
    Args:
        model: PyTorch模型
        logger: 日志记录器
        model_name: 模型名称
    """
    trainable, total = count_trainable_parameters(model)
    frozen = total - trainable
    
    logger.info(f"{model_name}参数统计:")
    logger.info(f"  - 总参数: {total:,}")
    logger.info(f"  - 可训练参数: {trainable:,}")
    logger.info(f"  - 冻结参数: {frozen:,}")

