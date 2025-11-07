"""训练相关工具函数"""
import os
import gc
import logging
import torch
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity
from contextlib import nullcontext
from utils.data_utils import prepare_batch_data, args_to_kwargs
from utils.memory_utils import analyze_gpu_memory

# 获取logger
train_utils_logger = logging.getLogger("train_logger")


def train(
    config,
    data_loader,
    model,
    optimizer,
    criterion,
    num_epochs,
    current_epoch,
    scheduler=None,
    device="cpu",
    kw_src=None,
    kw_tgt=None,
    kw_out=None,
    scaler=None,
    writer=None,
    enable_profile=False,
    device_manager=None,
):
    """
    训练一个epoch
    
    Args:
        config: 配置对象
        data_loader: 数据加载器
        model: 模型
        optimizer: 优化器
        criterion: 损失函数（未使用）
        num_epochs: 总epoch数
        current_epoch: 当前epoch
        scheduler: 学习率调度器
        device: 设备
        kw_src: 源关键字列表
        kw_tgt: 目标关键字列表
        kw_out: 输出关键字列表
        scaler: 混合精度训练scaler
        writer: TensorBoard writer
        enable_profile: 是否启用性能分析
        device_manager: 设备管理器
        
    Returns:
        epoch平均损失
    """
    # 清理内存
    torch.cuda.empty_cache()
    gc.collect()
    
    model.train()
    running_loss = 0
    
    # 记录当前学习率
    if writer is not None:
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("Learning Rate", current_lr, current_epoch)
    
    # TensorBoard记录频率
    log_freq = 500
    
    # 性能分析器
    profiler = None
    if enable_profile:
        profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        profiler.start()
    
    # 训练循环
    prog_bar = tqdm(data_loader, desc=f"Training Epoch {current_epoch}")
    
    for i, batch in enumerate(prog_bar):
        # 内存分析
        if i % 100 == 0 and enable_profile:
            train_utils_logger.info(f"\nBatch {i} - GPU内存使用:")
            train_utils_logger.info(f"已分配: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            train_utils_logger.info(f"已缓存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
            analyze_gpu_memory()
        
        if enable_profile and i == 4:
            break
        
        # 准备批次数据
        with record_function("data_preparation") if enable_profile else nullcontext():
            source, target, _ = _prepare_phase_data(
                config, batch, data_loader, device
            )
        
        # 转换为kwargs
        source = args_to_kwargs(source)
        target = args_to_kwargs(target)
        
        # 添加阶段信息
        source["phase"] = config.PHASE
        source["mode"] = "train"
        source["current_epoch"] = current_epoch
        source["total_epochs"] = num_epochs
        
        optimizer.zero_grad()
        
        # 前向传播和损失计算
        with torch.amp.autocast("cuda", enabled=scaler is not None):
            with record_function("model_forward") if enable_profile else nullcontext():
                # 根据训练阶段选择不同的forward方式
                if config.PHASE == "TRAIN_DETECTION":
                    # 目标检测：model(images, targets)
                    output = model(source["image"], source["bbox_targets"])
                else:
                    # 其他阶段：使用字典传递参数
                    output = model(**source)
            
            loss = _compute_loss(config, output)
        
        # 同步损失（分布式）
        if device_manager is not None and device_manager.distributed:
            loss_tensor = loss.detach().clone()
            loss_reduced = device_manager.reduce_tensor(loss_tensor)
            running_loss += loss_reduced.item()
        else:
            running_loss += loss.item()
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step(cur_epoch=current_epoch, cur_step=i)
        
        # 更新进度条
        current_lr = optimizer.param_groups[0]["lr"]
        prog_bar.set_postfix({
            'loss': f'{running_loss/(i+1):.4f}',
            'lr': f'{current_lr:.2e}'
        })
        
        # 反向传播
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # 记录到TensorBoard (在删除变量之前)
        if writer is not None and i % log_freq == 0:
            _log_training_metrics(writer, config, loss, output, current_epoch, i, len(data_loader))
        
        # 每个batch结束后立即清理，防止内存累积
        # 删除不再需要的变量并detach
        del loss, output, source, target
        
        # 定期深度内存清理
        if i % 50 == 0 and i > 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        # 性能分析
        if enable_profile and profiler is not None:
            profiler.step()
    
    # 停止性能分析
    if enable_profile and profiler is not None:
        profiler.stop()
        train_utils_logger.info(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        profiler.export_chrome_trace("trace_training.json")
    
    # 记录epoch平均损失
    epoch_loss = running_loss / len(data_loader)
    if writer is not None:
        writer.add_scalar("Train/Epoch_Loss", epoch_loss, current_epoch)
    
    return epoch_loss


def _prepare_phase_data(config, batch, data_loader, device):
    """根据训练阶段准备数据"""
    phase = config.PHASE
    
    if phase == "TRAIN_DETECTION":
        return prepare_batch_data(
            config, batch, data_loader, device,
            findings=False, history=False, label=False, bbox=True
        )
    elif phase == "PRETRAIN_VIT":
        return prepare_batch_data(
            config, batch, data_loader, device,
            findings=True, history=False, label=True, bbox=True
        )
    elif phase in ["FINETUNE_BERT"]:
        return prepare_batch_data(
            config, batch, data_loader, device,
            findings=True, history=True, label=True, bbox=True
        )
    else:
        raise ValueError(f"Invalid phase: {phase}")


def _compute_loss(config, output):
    """计算损失"""
    phase = config.PHASE
    
    if phase == "FINETUNE_BERT":
        # 微调阶段：生成损失 + RGAT损失
        loss = output.loss
        
        # 添加RGAT疾病分类损失
        if hasattr(output, 'rgat_loss') and output.rgat_loss is not None:
            rgat_weight = getattr(config, 'RGAT_LOSS_WEIGHT', 1.0)
            loss += rgat_weight * output.rgat_loss
        
        return loss
    
    # 其他阶段需要组合损失
    output = args_to_kwargs(output)
    
    if phase == "TRAIN_DETECTION":
        # 修复：避免保留整个output字典的计算图
        # 将损失相加并立即释放中间变量
        total_loss = None
        for loss_value in output.values():
            if total_loss is None:
                total_loss = loss_value
            else:
                total_loss = total_loss + loss_value
        return total_loss
    
    elif phase == "PRETRAIN_VIT":
        # 预训练阶段：根据配置选择对比损失类型
        loss_type = getattr(config, 'CONTRASTIVE_LOSS_TYPE', 'region')
        
        # 根据损失类型选择对应的损失键
        if loss_type == 'region':
            loss_key = 'region_itc_loss'
        elif loss_type == 'clip':
            loss_key = 'clip_itc_loss'
        elif loss_type == 'simple_region_clip':
            loss_key = 'simple_region_clip_loss'
        else:
            train_utils_logger.warning(f"⚠️ 未知的对比损失类型: {loss_type}")
            loss_key = 'region_itc_loss'
        
        if loss_key in output and output[loss_key] is not None:
            contrastive_weight = getattr(config, 'REGION_ITC_WEIGHT', 1.0)
            loss = contrastive_weight * output[loss_key]
            return loss
        else:
            # 如果没有相应的对比损失，返回零损失
            train_utils_logger.warning(f"⚠️ PRETRAIN_VIT阶段未检测到{loss_key}")
            return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu', requires_grad=True)
    
    else:
        raise ValueError(f"Invalid phase: {phase}")


def _log_training_metrics(writer, config, loss, output, epoch, step, total_steps):
    """记录训练指标到TensorBoard"""
    global_step = epoch * total_steps + step
    
    # 记录总损失
    writer.add_scalar("Train/Total_Loss", loss.item(), global_step)
    
    # 根据阶段记录特定损失
    phase = config.PHASE
    
    if phase == "TRAIN_DETECTION":
        for loss_name, loss_value in output.items():
            writer.add_scalar(
                f"Train/Detection/{loss_name}",
                loss_value.item(),
                global_step
            )
    
    elif phase == "PRETRAIN_VIT":
        output_dict = args_to_kwargs(output)
        vit_losses = {}
        loss_type = getattr(config, 'CONTRASTIVE_LOSS_TYPE', 'region')
        if loss_type == 'region':
            if "region_itc_loss" in output_dict and output_dict["region_itc_loss"] is not None:
                vit_losses["Train/ViT/Region_ITC_Loss"] = output_dict["region_itc_loss"].item()
        elif loss_type == 'clip':
            if "clip_itc_loss" in output_dict and output_dict["clip_itc_loss"] is not None:
                vit_losses["Train/ViT/CLIP_ITC_Loss"] = output_dict["clip_itc_loss"].item()
        elif loss_type == 'simple_region_clip':
            if "simple_region_clip_loss" in output_dict and output_dict["simple_region_clip_loss"] is not None:
                vit_losses["Train/ViT/Simple_Region_CLIP_Loss"] = output_dict["simple_region_clip_loss"].item()
        for tag, value in vit_losses.items():
            writer.add_scalar(tag, value, global_step)
    
    elif phase == "FINETUNE_BERT":
        writer.add_scalar("Train/Finetune/Generation_Loss", output.loss.item(), global_step)
        
        # 记录RGAT损失
        if hasattr(output, 'rgat_loss') and output.rgat_loss is not None:
            writer.add_scalar("Train/Finetune/RGAT_Loss", output.rgat_loss.item(), global_step)
