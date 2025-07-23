import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops import box_iou
import matplotlib.pyplot as plt
import logging
import os
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import metrics
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.profiler import profile, record_function, ProfilerActivity
from contextlib import nullcontext
import gc
import json
import pickle
from datasets import MIMIC

# Set the environment variable to disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ------ Helper Functions ------
def data_to_device(data, device="cpu"):
    if isinstance(data, torch.Tensor):
        data = data.to(device)
    elif isinstance(data, tuple):
        data = tuple(data_to_device(item, device) for item in data)
    elif isinstance(data, list):
        data = list(data_to_device(item, device) for item in data)
    elif isinstance(data, dict):
        data = dict((k, data_to_device(v, device)) for k, v in data.items())
    # else:
    # raise TypeError('Unsupported Datatype! Must be a Tensor/List/Tuple/Dict.')

    return data


def data_concatenate(iterable_data, dim=0):
    data = iterable_data[0]  # can be a list / tuple / dict / tensor
    if isinstance(data, torch.Tensor):
        return torch.cat([*iterable_data], dim=dim)
    elif isinstance(data, tuple):
        num_cols = len(data)
        num_rows = len(iterable_data)
        return_data = []
        for col in range(num_cols):
            data_col = []
            for row in range(num_rows):
                data_col.append(iterable_data[row][col])
            return_data.append(torch.cat([*data_col], dim=dim))
        return tuple(return_data)
    elif isinstance(data, list):
        num_cols = len(data)
        num_rows = len(iterable_data)
        return_data = []
        for col in range(num_cols):
            data_col = []
            for row in range(num_rows):
                data_col.append(iterable_data[row][col])
            return_data.append(torch.cat([*data_col], dim=dim))
        return list(return_data)
    elif isinstance(data, dict):
        num_cols = len(data)
        num_rows = len(iterable_data)
        return_data = []
        for col in data.keys():
            data_col = []
            for row in range(num_rows):
                data_col.append(iterable_data[row][col])
            return_data.append(torch.cat([*data_col], dim=dim))
        return dict((k, return_data[i]) for i, k in enumerate(data.keys()))
    else:
        raise TypeError("Unsupported Datatype! Must be a Tensor/List/Tuple/Dict.")


def data_distributor(model, source):
    if isinstance(source, torch.Tensor):
        output = model(source)
    elif isinstance(source, tuple) or isinstance(source, list):
        output = model(*source)
    elif isinstance(source, dict):
        output = model(**source)
    else:
        raise TypeError("Unsupported DataType! Try List/Tuple!")
    return output


def args_to_kwargs(
    args, kwargs_list=None
):  # This function helps distribute input to corresponding arguments in Torch models
    if kwargs_list != None:
        if isinstance(args, dict):  # Nothing to do here
            return args
        else:  # args is a list or tuple or single element
            if isinstance(args, torch.Tensor):  # single element
                args = [args]
            assert len(args) == len(kwargs_list)
            return dict(zip(kwargs_list, args))
    else:  # Nothing to do here
        return args


def prepare_batch_data(
    config,
    batch,
    data_loader,
    device,
    findings=True,
    history=True,
    label=True,
    bbox=True,
):
    """准备批次数据，对整个batch进行tokenization

    Args:
        config: 配置参数
        batch: 输入的批次数据
        data_loader: 数据加载器
        device: 计算设备
        findings: 是否处理findings字段
        history: 是否处理history字段
        label: 是否处理label字段
        bbox: 是否处理bbox_targets字段

    Returns:
        source_data: 源数据字典
        target_data: 目标数据字典
    """
    source = {}
    target = {}

    # 处理图像数据 - 直接在GPU上处理
    if "image" in batch:
        source["image"] = batch["image"] = batch["image"].to(device, non_blocking=True)

    # 确定需要处理的文本字段
    text_fields_to_process = []
    if findings and "findings" in batch:
        text_fields_to_process.append(("findings", config.MAX_LEN_FINDINGS))
    if history and "history" in batch:
        text_fields_to_process.append(("history", config.MAX_LEN_HISTORY))

    # 获取tokenizer并确保设置正确的padding_side
    tokenizer = data_loader.dataset.tokenizer
    original_padding_side = tokenizer.padding_side
    
    # 确保使用右侧padding，与BERT预训练一致
    tokenizer.padding_side = 'right'

    # 批量处理文本字段
    for field, max_len in text_fields_to_process:
        texts = batch[field]

        # 对整个batch进行tokenization，使用non_blocking=True加速GPU传输
        encoded = tokenizer(
            texts,
            max_length=max_len,
            padding="max_length",  # 使用max_length保持一致性
            truncation=True,
            return_tensors="pt",
        ).to(device)

        batch[field] = encoded
        source[field] = encoded  # 直接使用同一个对象引用
        target[field] = encoded  # 直接使用同一个对象引用

    # 恢复tokenizer原始padding_side设置
    tokenizer.padding_side = original_padding_side

    # 优化: 一次性处理标签数据
    if label and "label" in batch:
        source["label"] = target["label"] = batch["label"] = batch["label"].to(
            device, non_blocking=True
        )

    # 处理边界框数据 - 注意这是一个列表，每个元素是字典
    if bbox and "bbox_targets" in batch:
        # 创建一个新的bbox_targets列表，将每个字典中的tensor移动到设备上
        processed_bbox_targets = []

        for bbox_target in batch["bbox_targets"]:
            # 对字典中的每个tensor进行处理
            processed_target = {}
            for key, value in bbox_target.items():
                if isinstance(value, torch.Tensor):
                    processed_target[key] = value.to(device, non_blocking=True)
                else:
                    processed_target[key] = value
            processed_bbox_targets.append(processed_target)

        batch["bbox_targets"] = processed_bbox_targets
        source["bbox_targets"] = processed_bbox_targets

    # 处理image_id（用于文本增强）
    if "image_id" in batch:
        source["image_ids"] = batch["image_id"]  # image_id是字符串列表，不需要移动到设备
    
    # 处理解剖区域嵌入数据（用于区域级别ITC损失）
    if "anatomical_embeddings" in batch:
        source["anatomical_embeddings_batch"] = batch["anatomical_embeddings"]  # 解剖区域嵌入数据列表

    return source, target, None


# ------ Core Functions ------
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
    device_manager=None,  # 新增设备管理器参数
):
    import gc  # 在函数开头导入gc模块
    torch.cuda.empty_cache()
    gc.collect()
    model.train()
    running_loss = 0

    # 记录当前学习率，仅在每个epoch开始时记录一次
    if writer is not None:
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("Learning Rate", current_lr, current_epoch)

    # 设置TensorBoard记录频率，如每10个批次记录一次
    log_freq = 500

    # 根据enable_profile参数决定是否使用性能分析器
    if enable_profile:
        profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        profiler.start()
    else:
        profiler = None

    prog_bar = tqdm(data_loader)
    for i, batch in enumerate(prog_bar):

        # 每100个批次分析一次内存
        if i % 100 == 0 and enable_profile:
            print(f"\nBatch {i} - 当前GPU内存使用情况:")
            print(f"已分配: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"已缓存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
            analyze_gpu_memory()  # 详细分析内存使用

        if enable_profile and i == 4:  # 只在进行性能分析时提前结束
            break
        # 准备批次数据
        with record_function("data_preparation") if enable_profile else nullcontext():
            if config.PHASE == "TRAIN_DETECTION":
                source, target, _ = prepare_batch_data(
                    config,
                    batch,
                    data_loader,
                    device,
                    findings=False,
                    history=False,
                    label=False,
                    bbox=True,
                )
            elif config.PHASE == "PRETRAIN_VIT":
                source, target, _ = prepare_batch_data(
                    config,
                    batch,
                    data_loader,
                    device,
                    findings=True,
                    history=False,
                    label=True,
                    bbox=True,
                )
            elif config.PHASE == "FINETUNE_MISTRAL" or config.PHASE == "FINETUNE_LLAMA" or config.PHASE == "FINETUNE_BERT":
                source, target, _ = prepare_batch_data(
                    config,
                    batch,
                    data_loader,
                    device,
                    findings=True,
                    history=True,
                    label=True,
                    bbox=True,
                )
            else:
                raise ValueError("Invalid phase")
        # 转换为kwargs格式
        source = args_to_kwargs(source)
        target = args_to_kwargs(target)

        source["phase"] = config.PHASE
        source["mode"] = "train"
        source["current_epoch"] = current_epoch
        source["total_epochs"] = num_epochs

        optimizer.zero_grad()

        # 根据不同阶段执行不同的训练逻辑
        if config.PHASE == "FINETUNE_MISTRAL" or config.PHASE == "FINETUNE_LLAMA" or config.PHASE == "FINETUNE_BERT":
            with torch.amp.autocast("cuda", enabled=scaler is not None):
                # 直接将所有数据分发给模型，由模型内部处理各组件间的逻辑
                output = data_distributor(model, source)
                loss = output.loss
        else:
            with torch.amp.autocast("cuda"):
                with record_function(
                    "model_forward"
                ) if enable_profile else nullcontext():
                    output = data_distributor(model, source)
                output = args_to_kwargs(output)

                if config.PHASE == "TRAIN_DETECTION":
                    # 汇总所有损失项
                    loss = sum(loss for loss in output.values())
                elif config.PHASE == "PRETRAIN_VIT":
                    loss = output["ltc_loss"] + output["cls_loss"]
                    
                    # 添加区域级别ITC损失（如果存在的话）
                    if "region_itc_loss" in output and output["region_itc_loss"] is not None:
                        # 可以调整权重，这里使用与其他损失相同的权重
                        region_itc_weight = getattr(config, 'REGION_ITC_WEIGHT', 1.0)
                        loss += region_itc_weight * output["region_itc_loss"]
                else:
                    raise ValueError("Invalid phase")

        # 分布式训练时同步损失
        if device_manager is not None and device_manager.distributed:
            # 同步损失值
            loss_tensor = loss.detach().clone()
            loss_reduced = device_manager.reduce_tensor(loss_tensor)
            running_loss += loss_reduced.item()
        else:
            running_loss += loss.item()

        # 学习率更新
        if scheduler is not None:
            scheduler.step(cur_epoch=current_epoch, cur_step=i)

        current_lr = optimizer.param_groups[0]["lr"]
        prog_bar.set_description(f"Loss: {running_loss/(i+1)} | LR: {current_lr}")

        # 反向传播与优化
        if scaler is not None:
            scaler.scale(loss).backward()
            # 添加梯度裁剪，防止梯度爆炸导致数值不稳定
            scaler.unscale_(optimizer)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # 每log_freq个批次记录一次TensorBoard
        if writer is not None and i % log_freq == 0:
            global_step = current_epoch * len(data_loader) + i

            # 记录总损失
            writer.add_scalar("Train/Total_Loss", loss.item(), global_step)

            # 根据阶段记录特定损失
            if config.PHASE == "TRAIN_DETECTION":
                # 批量记录多个损失项
                for loss_name, loss_value in output.items():
                    writer.add_scalar(
                        f"Train/Detection/{loss_name}", loss_value.item(), global_step
                    )
            elif config.PHASE == "PRETRAIN_VIT":
                # 批量记录ViT相关损失
                vit_losses = {
                    "Train/ViT/LTC_Loss": output["ltc_loss"].item(),
                    "Train/ViT/CLS_Loss": output["cls_loss"].item(),
                }
                
                # 记录区域ITC损失（如果存在的话）
                if "region_itc_loss" in output and output["region_itc_loss"] is not None:
                    vit_losses["Train/ViT/Region_ITC_Loss"] = output["region_itc_loss"].item()
                
                for tag, value in vit_losses.items():
                    writer.add_scalar(tag, value, global_step)
            elif config.PHASE == "FINETUNE_MISTRAL":
                # 记录Mistral相关损失
                writer.add_scalar("Train/Mistral/Loss", loss.item(), global_step)

        # 定期内存清理（每50个批次）
        if i % 50 == 0 and i > 0:
            # 清理未使用的中间变量
            del loss, output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # 强制垃圾回收
            gc.collect()

        # 如果启用了性能分析，调用prof.step()
        if enable_profile:
            profiler.step()

    # 如果启用了性能分析，打印和导出结果
    if enable_profile:
        profiler.stop()
        print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        profiler.export_chrome_trace("trace_optim_tb_od_np.json")

    # 记录每个epoch的平均损失
    epoch_loss = running_loss / len(data_loader)
    if writer is not None:
        writer.add_scalar("Train/Epoch_Loss", epoch_loss, current_epoch)

    return epoch_loss


def test(
    config,
    data_loader,
    model,
    logger,
    mode="val",
    metric_ftns=None,
    criterion=None,
    device="cpu",
    kw_src=None,
    kw_tgt=None,
    kw_out=None,
    epoch=None,
):
    model.eval()
    running_loss = 0

    # 初始化存储列表
    findings_gts_list = []
    findings_preds_list = []
    impression_gts_list = []
    impression_preds_list = []
    image_paths_list = []
    splits_list = []
    labels_list = []

    with torch.no_grad():
        prog_bar = tqdm(data_loader)
        for i, batch in enumerate(prog_bar):
            # 收集元数据
            image_paths_list.extend(batch["image_path"])
            splits_list.extend(batch["split"])
            labels_list.extend(batch["label"].cpu().numpy().tolist())

            # 收集ground truth
            findings_gts_list.extend([gt for gt in batch["gts"][0]])
            impression_gts_list.extend([gt for gt in batch["gts"][1]])

            # 准备批次数据
            if config.PHASE == "TRAIN_DETECTION":
                source, target, _ = prepare_batch_data(
                    config,
                    batch,
                    data_loader,
                    device,
                    findings=False,
                    history=False,
                    label=False,
                    bbox=True,
                )
            elif config.PHASE == "PRETRAIN_VIT":
                source, target, _ = prepare_batch_data(
                    config,
                    batch,
                    data_loader,
                    device,
                    findings=False,
                    history=False,
                    label=True,
                    bbox=True,
                )
            else:
                pass

            # 转换为kwargs格式
            source = args_to_kwargs(source, kw_src)
            target = args_to_kwargs(target, kw_tgt)

            source["phase"] = config.PHASE
            source["mode"] = mode

            # 模型推理
            output = data_distributor(model, source)
            output = args_to_kwargs(output, kw_out)

            # 收集预测结果
            findings_preds_list.extend([re for re in output["findings_text"]])

            # 计算损失
            if criterion is not None:
                loss = torch.tensor(0.0)
                running_loss += loss.item()
            prog_bar.set_description("Loss: {}".format(running_loss / (i + 1)))

        # 创建结果数据字典
        results_data = {
            "image_path": image_paths_list,
            "split": splits_list,
            "findings_gt": findings_gts_list,
            "findings_pred": findings_preds_list,
            "labels": labels_list,
            "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
            * len(findings_gts_list),
        }

        # 计算评估指标
        findings_met = metric_ftns(
            {i: [gt] for i, gt in enumerate(findings_gts_list)},
            {i: [re] for i, re in enumerate(findings_preds_list)},
        )

        # 创建结果目录
        results_dir = os.path.join(config.CHECKPOINT_PATH_TO, "test_results")
        os.makedirs(results_dir, exist_ok=True)

        # 将结果转换为DataFrame并保存
        results_df = pd.DataFrame(results_data)

        # 保存为CSV文件，添加epoch信息
        epoch_str = str(epoch) if epoch is not None else "TEST"
        csv_filename = f"{mode}_results_epoch_{epoch_str}.csv"
        results_df.to_csv(os.path.join(results_dir, csv_filename), index=False)
        logger.info(f"结果已保存到CSV文件: {csv_filename}")

        # 计算并保存评估指标
        metrics_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": mode,
            "epoch": epoch_str,
            "loss": running_loss / len(data_loader),
        }

        # 添加findings指标
        for metric_name, value in findings_met.items():
            metrics_data[f"findings_{metric_name}"] = value

        # 保存评估指标，添加epoch信息
        metrics_df = pd.DataFrame([metrics_data])
        metrics_filename = f"{mode}_metrics_epoch_{epoch_str}.csv"
        metrics_df.to_csv(os.path.join(results_dir, metrics_filename), index=False)
        logger.info(f"评估指标已保存到CSV文件: {metrics_filename}")

        # 返回结果
        result = {
            "findings_met": findings_met,
            "loss": running_loss / len(data_loader),
            "results_df": results_df,
            "metrics_df": metrics_df,
        }

    return running_loss / len(data_loader), result



def save(path, model, optimizer=None, scheduler=None, epoch=-1, stats=None):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    
    # 检查scheduler是否有state_dict方法
    scheduler_state = None
    if scheduler is not None:
        try:
            scheduler_state = scheduler.state_dict()
        except AttributeError:
            print("警告: scheduler没有state_dict方法，无法保存scheduler状态")
            scheduler_state = None
    
    torch.save(
        {
            # --- Model Statistics ---
            "epoch": epoch,
            "stats": stats,
            # --- Model Parameters ---
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": (
                optimizer.state_dict() if optimizer != None else None
            ),
            'scheduler_state_dict': scheduler_state,
        },
        path,
    )


def load(path, model, optimizer=None, scheduler=None, load_model="object_detector"):
    checkpoint = torch.load(path, weights_only=False)
    # --- Model Statistics ---
    epoch = checkpoint["epoch"]
    stats = checkpoint["stats"]

    # --- 提取模型参数 ---
    if "model_state_dict" in checkpoint:
        checkpoint_state_dict = checkpoint["model_state_dict"]

        # 根据load_model参数决定加载哪部分权重
        if load_model == "object_detector":
            print("加载目标检测器参数...")
            # 检查是否是MOE模型的检查点
            is_moe_checkpoint = any(
                key.startswith("object_detector.")
                for key in checkpoint_state_dict.keys()
            )

            if is_moe_checkpoint:
                print("检测到MOE模型检查点，正在提取目标检测器参数...")
                # 创建一个新的state_dict，只包含detector部分
                filtered_state_dict = {}

                # 遍历checkpoint中的所有键
                for key, value in checkpoint_state_dict.items():
                    # 如果键以"object_detector."开头，则提取
                    if key.startswith("object_detector."):
                        # 去掉"object_detector."前缀
                        new_key = key[len("object_detector.") :]
                        filtered_state_dict[new_key] = value
            else:
                # 如果不是MOE检查点，直接使用原state_dict
                filtered_state_dict = checkpoint_state_dict

        elif load_model == "vit":
            print("加载ViT图像编码器参数...")
            # 提取以image_encoder.开头的权重
            filtered_state_dict = {}

            # 遍历checkpoint中的所有键
            for key, value in checkpoint_state_dict.items():
                # 如果键以"image_encoder."开头，则提取
                if key.startswith("image_encoder."):
                    # 去掉"image_encoder."前缀
                    new_key = key[len("image_encoder.") :]
                    filtered_state_dict[new_key] = value
                    
        elif load_model == "decoder":
            print("加载报告生成解码器参数...")
            # 提取解码器部分的权重
            decoder_prefix = "findings_decoder.decoder."
            
            filtered_state_dict = {}

            # 遍历checkpoint中的所有键
            for key, value in checkpoint_state_dict.items():
                # 如果键以decoder_prefix开头，则提取
                if key.startswith(decoder_prefix):
                    # 去掉decoder_prefix前缀
                    new_key = key[len(decoder_prefix):]
                    filtered_state_dict[new_key] = value
            
            # 尝试使用 'findings_decoder.' 作为前缀 （兼容旧逻辑或 Mistral/Llama）
            if not filtered_state_dict:
                print(f"未找到前缀为 '{decoder_prefix}' 的键，尝试使用 'findings_decoder.' ...")
                legacy_prefix = "findings_decoder."
                for key, value in checkpoint_state_dict.items():
                    if key.startswith(legacy_prefix):
                        new_key = key[len(legacy_prefix):]
                        filtered_state_dict[new_key] = value
                        
            if not filtered_state_dict:
                print("警告：在检查点中未找到解码器权重！将使用随机初始化的权重。")
                # 返回空字典，让模型使用初始权重
                pass # filtered_state_dict 已经是 {} 了

        elif load_model == "full":
            print("加载完整模型参数...")
            # 直接使用完整的state_dict，不进行过滤
            filtered_state_dict = checkpoint_state_dict
        else:
            # 默认情况，加载完整模型
            filtered_state_dict = checkpoint_state_dict

        # 加载提取后的state_dict到模型
        missing_keys, unexpected_keys = model.load_state_dict(
            filtered_state_dict, strict=True
        )
    else:
        print("检查点中没有找到模型状态字典！")
        return epoch, stats

    # 打印未加载和多余的参数信息
    if len(missing_keys) > 0:
        print(f"Missing keys ({len(missing_keys)}): {missing_keys[:10]}...")  # 只显示前10个
        if len(missing_keys) > 10:
            print(f"... 以及其他 {len(missing_keys) - 10} 个缺失的键")
    if len(unexpected_keys) > 0:
        print(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:10]}...")  # 只显示前10个
        if len(unexpected_keys) > 10:
            print(f"... 以及其他 {len(unexpected_keys) - 10} 个意外的键")
            
    # 计算权重加载成功率
    total_model_params = len(model.state_dict())
    loaded_params = total_model_params - len(missing_keys)
    load_success_rate = loaded_params / total_model_params * 100
    print(f"权重加载成功率: {load_success_rate:.2f}% ({loaded_params}/{total_model_params})")

    if optimizer != None:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except:  # Input optimizer doesn't fit the checkpoint one --> should be ignored
            print("Cannot load the optimizer")
    
    # 恢复scheduler状态
    if scheduler is not None and epoch is not None:
        try:
            # 检查scheduler是否有load_state_dict方法
            has_state_dict_method = hasattr(scheduler, 'load_state_dict')
            
            # 如果检查点包含scheduler状态，且scheduler支持state_dict，则直接加载
            if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None and has_state_dict_method:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                print(f"成功加载scheduler状态")
            else:
                # 不能直接加载状态的原因
                if not has_state_dict_method:
                    print(f"当前scheduler不支持state_dict，将使用step模拟")
                elif "scheduler_state_dict" not in checkpoint or checkpoint["scheduler_state_dict"] is None:
                    print(f"检查点中没有scheduler状态，将使用step模拟")
                
                # 估算数据加载器的长度（每个epoch的batch数）
                est_batches_per_epoch = 2500
                
                print(f"模拟scheduler调度到epoch {epoch}...")
                print(f"提示：scheduler在每个batch都会被调用，我们将模拟调用到epoch {epoch}的最后一个batch")
                
                # 模拟每个epoch的所有batch
                for e in range(epoch):
                    # 模拟该epoch的最后一个batch
                    # 这是一个简化，实际上我们应该模拟所有batch，但为了效率只模拟最后一个
                    scheduler.step(cur_epoch=e, cur_step=est_batches_per_epoch-1)
                
                print(f"scheduler状态已恢复到接近epoch {epoch}的状态")
                print(f"当前学习率：{optimizer.param_groups[0]['lr']}")
                
                # 验证学习率是否合理
                current_lr = optimizer.param_groups[0]['lr']
                if current_lr < 1e-8:
                    print("⚠️  警告: 当前学习率非常小，可能会导致训练缓慢或损失异常！")
                elif current_lr > 1e-2:
                    print("⚠️  警告: 当前学习率较大，可能会导致训练不稳定！")
        except Exception as e:
            print(f"恢复scheduler状态失败: {e}")
            print("继续训练，但scheduler状态可能不正确")
            print("提示：您可以手动设置学习率：optimizer.param_groups[0]['lr'] = 您想要的学习率值")

    return epoch, stats


def log_metrics(logger, epoch, train_loss, test_loss, result):
    """记录训练和测试的评估指标

    Args:
        logger: 日志记录器
        epoch: 当前轮次(可选,如果是None则不输出)
        train_loss: 训练损失(可选,如果是None则不输出)
        test_loss: 测试损失
        result: 包含metrics_df的结果字典
    """
    if epoch is not None:
        logger.info(f"epoch: {epoch}")
    if train_loss is not None:
        logger.info(f"train_loss: {train_loss}")
    logger.info(f"test_loss: {test_loss:.4f}")

    # 输出评估指标
    metrics_df = result["metrics_df"]
    for index, row in metrics_df.iterrows():
        for column in metrics_df.columns:
            value = row[column]
            if isinstance(value, (int, float)):
                logger.info(f"{column}: {value:.4f}")
            else:
                logger.info(f"{column}: {value}")


def count_parameters(model):
    """Count the number of parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def visual_parameters(modules, parameters):
    # 数据
    total_parameters = sum(parameters)

    # 计算占比
    percentages = [param / total_parameters * 100 for param in parameters]

    # 绘制饼状图
    plt.figure(figsize=(8, 8))
    plt.pie(percentages, labels=modules, autopct="%1.1f%%", startangle=140)
    plt.title("Parameter Distribution Among Modules")
    plt.savefig(
        "/home/chenlb/xray_report_generation/results/parameter_distribution.png"
    )


# 绘制直方图
def plot_length_distribution(distribution, title):
    # 准备数据
    lengths = list(distribution.keys())  # Token 长度
    counts = list(distribution.values())  # 出现次数

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.bar(lengths, counts)
    plt.title(title)
    plt.xlabel("Token Length")
    plt.ylabel("Count")

    # 计算合适的刻度间隔
    max_length = max(lengths)
    min_length = min(lengths)
    step = max(1, (max_length - min_length) // 5)  # 最多显示约10个刻度

    # 设置x轴刻度
    plt.xticks(range(min_length, max_length + 1, step))

    plt.tight_layout()
    plt.savefig(
        f"/home/chenlb/xray_report_generation/results/{title.replace(' ', '_')}.png"
    )


def setup_logger(log_dir="logs"):
    """
    设置logger，同时输出到控制台和文件

    Args:
        log_dir: 日志文件存储目录
    Returns:
        logger: 配置好的logger对象
    """
    # 创建日志目录
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 生成日志文件名（使用当前时间）
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{current_time}.log")

    # 创建logger对象
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)

    # 创建文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 设置日志格式
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def clean_report_mimic_cxr(report):
    report_cleaner = (
        lambda t: t.replace("\n", " ")
        .replace("__", "_")
        .replace("__", "_")
        .replace("__", "_")
        .replace("__", "_")
        .replace("__", "_")
        .replace("__", "_")
        .replace("__", "_")
        .replace("  ", " ")
        .replace("  ", " ")
        .replace("  ", " ")
        .replace("  ", " ")
        .replace("  ", " ")
        .replace("  ", " ")
        .replace("..", ".")
        .replace("..", ".")
        .replace("..", ".")
        .replace("..", ".")
        .replace("..", ".")
        .replace("..", ".")
        .replace("..", ".")
        .replace("..", ".")
        .replace("1. ", "")
        .replace(". 2. ", ". ")
        .replace(". 3. ", ". ")
        .replace(". 4. ", ". ")
        .replace(". 5. ", ". ")
        .replace(" 2. ", ". ")
        .replace(" 3. ", ". ")
        .replace(" 4. ", ". ")
        .replace(" 5. ", ". ")
        .strip()
        .lower()
        .split(". ")
    )
    sent_cleaner = lambda t: re.sub(
        "[.,?;*!%^&_+():-\[\]{}]",
        "",
        t.replace('"', "")
        .replace("/", "")
        .replace("\\", "")
        .replace("'", "")
        .strip()
        .lower(),
    )
    tokens = [
        sent_cleaner(sent)
        for sent in report_cleaner(report)
        if sent_cleaner(sent) != []
    ]
    report = " . ".join(tokens) + " ."
    return report


def analyze_results_from_csv(csv_path, metric_ftns=None):
    """从CSV文件中分析结果并计算评估指标，包括findings、impression以及它们的组合

    Args:
        csv_path: CSV文件路径
        metric_ftns: 计算指标的函数，默认为None

    Returns:
        dict: 包含findings、impression和combined指标的字典
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)

    # 检查必要的列是否存在
    required_columns = ["findings_gt", "findings_pred"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV文件必须包含以下列: {required_columns}")

    # 准备findings的ground truth和预测结果
    findings_gts = {i: [gt] for i, gt in enumerate(df["findings_gt"])}
    findings_preds = {i: [pred] for i, pred in enumerate(df["findings_pred"])}

    # 计算findings的指标
    findings_metrics = metric_ftns(findings_gts, findings_preds) if metric_ftns else {}

    # 初始化impression和combined的指标为None
    impression_metrics = None
    combined_metrics = None

    # 如果存在impression相关列，计算impression和combined的指标
    if "impression_gt" in df.columns and "impression_pred" in df.columns:
        # 计算impression的指标
        impression_gts = {}
        impression_preds = {}
        idx = 0
        for i in range(len(df)):
            if (
                df["impression_gt"][i].strip() != ""
                and df["impression_pred"][i].strip() != ""
            ):
                impression_gts[idx] = [df["impression_gt"][i]]
                impression_preds[idx] = [df["impression_pred"][i].replace(".", " .")]
                idx += 1
        # impression_gts = {i: [gt] for i, gt in enumerate(df["impression_gt"])}
        # impression_preds = {i: [pred] for i, pred in enumerate(df["impression_pred"])}
        impression_metrics = (
            metric_ftns(impression_gts, impression_preds) if metric_ftns else {}
        )

        # 计算combined (findings + impression)的指标
        combined_gts = {}
        combined_preds = {}
        for i, (f_gt, i_gt, f_pred, i_pred) in enumerate(
            zip(
                df["findings_gt"],
                df["impression_gt"],
                df["findings_pred"],
                df["impression_pred"],
            )
        ):
            # 只有当impression不为空时才组合
            if isinstance(i_gt, str) and len(i_gt.strip()) > 0:
                combined_gts[i] = [f"{f_gt} {i_gt}"]
                combined_preds[i] = [f"{f_pred} {i_pred}"]
            else:
                combined_gts[i] = [f_gt]
                combined_preds[i] = [f_pred]

        combined_metrics = (
            metric_ftns(combined_gts, combined_preds) if metric_ftns else {}
        )

    # 整理结果
    results = {
        "findings_metrics": findings_metrics,
        "impression_metrics": impression_metrics,
        "combined_metrics": combined_metrics,
    }

    return results


def save_generations(
    config,
    data_loader,
    model,
    logger,
    save_dir,
    mode="test",
    device="cpu",
    kw_src=None,
    kw_tgt=None,
    kw_out=None,
):
    """保存模型生成的findings和impression结果

    Args:
        config: 配置参数
        data_loader: 数据加载器
        model: 模型
        logger: 日志记录器
        save_dir: 保存结果的目录
        mode: 运行模式，默认为"test"
        device: 计算设备
        kw_src: source关键字参数列表
        kw_tgt: target关键字参数列表
        kw_out: output关键字参数列表
    """
    model.eval()

    # 初始化存储列表
    findings_gts_list = []
    findings_preds_list = []
    impression_gts_list = []
    impression_preds_list = []
    image_paths_list = []
    splits_list = []
    labels_list = []

    with torch.no_grad():
        prog_bar = tqdm(data_loader)
        for batch in prog_bar:
            # 收集元数据
            image_paths_list.extend(batch["image_path"])
            splits_list.extend(batch["split"])
            labels_list.extend(batch["label"].cpu().numpy().tolist())

            # 收集ground truth
            findings_gts_list.extend([gt for gt in batch["gts"][0]])
            impression_gts_list.extend([gt for gt in batch["gts"][1]])

            # 准备批次数据
            source, target, _ = prepare_batch_data(config, batch, data_loader, device)

            # 转换为kwargs格式
            source = args_to_kwargs(source, kw_src)
            target = args_to_kwargs(target, kw_tgt)

            source["phase"] = config.PHASE
            source["mode"] = mode
            
            # 提取image_ids用于检测缓存
            image_ids = [os.path.basename(path).split('.')[0] for path in batch["image_path"]]
            source["image_ids"] = image_ids

            # 模型推理
            output = data_distributor(model, source)
            output = args_to_kwargs(output, kw_out)

            # 收集预测结果
            findings_preds_list.extend([re for re in output["findings_text"]])

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 创建结果数据字典
    results_data = {
        "image_path": image_paths_list,
        "split": splits_list,
        "findings_gt": findings_gts_list,
        "findings_pred": findings_preds_list,
        "labels": labels_list,
        "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        * len(findings_gts_list),
    }

    # 将结果转换为DataFrame并保存
    results_df = pd.DataFrame(results_data)

    # 生成文件名并保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{mode}_generations_{timestamp}.csv"
    save_path = os.path.join(save_dir, csv_filename)
    results_df.to_csv(save_path, index=False)

    logger.info(f"生成结果已保存到: {save_path}")
    logger.info(f"总共保存了 {len(findings_gts_list)} 条记录")


def test_detection(
    config,
    data_loader,
    model,
    logger,
    mode="val",
    iou_threshold=0.5,
    confidence_threshold=0.5,
    device="cuda",
    epoch=None,
    writer=None,
):
    """
    评估目标检测模型性能 - 简化版

    参数:
        config: 配置参数
        data_loader: 测试数据加载器
        model: DetectionOnlyFastRCNN模型实例
        logger: 日志记录器
        mode: 评估模式 ("val" 或 "test")
        iou_threshold: 判定为成功检测的IoU阈值
        confidence_threshold: 检测置信度阈值
        device: 计算设备
        epoch: 当前训练轮次

    返回:
        float: 平均损失
        dict: 包含评估结果的字典
    """
    model.eval()
    running_loss = 0

    # 初始化存储结构
    all_predictions = []
    all_ground_truths = []
    image_paths_list = []

    # 按类别存储预测和真值
    num_classes = 29  # 假设有29个区域类别
    class_predictions = {i: [] for i in range(1, num_classes + 1)}
    class_ground_truths = {i: [] for i in range(1, num_classes + 1)}

    # 创建进度条
    prog_bar = tqdm(data_loader, desc=f"{mode} Detection Evaluation")

    with torch.no_grad():
        for i, batch in enumerate(prog_bar):
            # 收集图像路径
            image_paths_list.extend(batch["image_path"])

            # 准备数据 - 简化处理
            images = batch["image"].to(device)
            targets = []

            for target_dict in batch["bbox_targets"]:
                # 将目标数据移动到设备上
                target = {
                    "boxes": target_dict["boxes"].to(device),
                    "labels": target_dict["labels"].to(device),
                    "image_id": target_dict["image_id"].to(device),
                    "area": target_dict["area"].to(device),
                    "iscrowd": target_dict["iscrowd"].to(device),
                }
                targets.append(target)

            # 进行前向传播，获取检测结果
            try:
                # 尝试检测模式
                detections = model(images)
            except Exception as e:
                # 如果失败，尝试传入空目标以避免计算损失
                logger.warning(f"检测异常: {e}，尝试传入空目标进行推理")
                detections = model(images, [])

            # 计算损失（如果需要）- 这步是可选的
            if targets:
                try:
                    loss_dict = model(images, targets)
                    if isinstance(loss_dict, dict) and all(
                        k.startswith("loss") for k in loss_dict.keys()
                    ):
                        batch_loss = sum(loss for loss in loss_dict.values())
                        running_loss += batch_loss.item()
                except Exception as e:
                    logger.warning(f"损失计算异常: {e}")

            # 处理检测结果
            for j, (detection, target) in enumerate(
                zip(detections, targets if targets else [None] * len(detections))
            ):
                # 应用置信度阈值
                keep = detection["scores"] > confidence_threshold
                pred_boxes = detection["boxes"][keep]
                pred_labels = detection["labels"][keep]
                pred_scores = detection["scores"][keep]

                # 存储预测结果
                img_pred = {
                    "boxes": pred_boxes.cpu(),
                    "labels": pred_labels.cpu(),
                    "scores": pred_scores.cpu(),
                    "image_id": i * len(images) + j,
                }
                all_predictions.append(img_pred)

                # 存储真值
                if target is not None:
                    img_gt = {
                        "boxes": target["boxes"].cpu(),
                        "labels": target["labels"].cpu(),
                        "image_id": i * len(images) + j,
                    }
                    all_ground_truths.append(img_gt)

                    # 按类别存储预测和真值
                    for class_id in range(1, num_classes + 1):
                        # 提取当前类别的预测
                        class_pred_mask = pred_labels.cpu() == class_id
                        class_predictions[class_id].append(
                            {
                                "boxes": (
                                    pred_boxes.cpu()[class_pred_mask]
                                    if class_pred_mask.sum() > 0
                                    else torch.zeros((0, 4))
                                ),
                                "scores": (
                                    pred_scores.cpu()[class_pred_mask]
                                    if class_pred_mask.sum() > 0
                                    else torch.zeros(0)
                                ),
                                "image_id": i * len(images) + j,
                            }
                        )

                        # 提取当前类别的真值
                        class_gt_mask = target["labels"].cpu() == class_id
                        class_ground_truths[class_id].append(
                            {
                                "boxes": (
                                    target["boxes"].cpu()[class_gt_mask]
                                    if class_gt_mask.sum() > 0
                                    else torch.zeros((0, 4))
                                ),
                                "image_id": i * len(images) + j,
                            }
                        )

            # 更新进度条
            prog_bar.set_description(
                f"Loss: {running_loss/(i+1):.4f}"
                if running_loss > 0
                else "Evaluating..."
            )

    # 计算指标
    logger.info("计算目标检测评估指标...")

    # 计算整体mAP
    overall_metrics = calculate_detection_metrics(
        all_predictions, all_ground_truths, iou_threshold
    )

    # 计算每个类别的指标
    class_metrics = {}
    for class_id in range(1, num_classes + 1):
        class_metrics[class_id] = calculate_class_metrics(
            class_predictions[class_id], class_ground_truths[class_id], iou_threshold
        )

    # 计算平均指标
    valid_classes = [
        c for c in class_metrics.keys() if class_metrics[c]["num_samples"] > 0
    ]
    if valid_classes:
        average_precision = np.mean([class_metrics[c]["AP"] for c in valid_classes])
        average_recall = np.mean([class_metrics[c]["recall"] for c in valid_classes])
        average_f1 = np.mean([class_metrics[c]["f1_score"] for c in valid_classes])
    else:
        average_precision = 0.0
        average_recall = 0.0
        average_f1 = 0.0

    # 创建结果目录
    results_dir = os.path.join(config.CHECKPOINT_PATH_TO, "detection_results")
    os.makedirs(results_dir, exist_ok=True)

    # 整理结果数据
    metrics_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "epoch": str(epoch) if epoch is not None else "TEST",
        "mAP": average_precision,
        "mRecall": average_recall,
        "mF1": average_f1,
        "loss": running_loss / len(data_loader) if running_loss > 0 else 0.0,
    }

    # 添加每个类别的指标
    for class_id, metrics in class_metrics.items():
        metrics_data[f"class_{class_id}_AP"] = metrics["AP"]
        metrics_data[f"class_{class_id}_Precision"] = metrics["precision"]
        metrics_data[f"class_{class_id}_Recall"] = metrics["recall"]
        metrics_data[f"class_{class_id}_F1"] = metrics["f1_score"]

    # 保存评估指标
    metrics_df = pd.DataFrame([metrics_data])
    epoch_str = str(epoch) if epoch is not None else "TEST"
    metrics_filename = (
        f"{overall_metrics['mAP']}{mode}_detection_metrics_epoch_{epoch_str}.csv"
    )
    metrics_df.to_csv(os.path.join(results_dir, metrics_filename), index=False)
    logger.info(f"目标检测评估指标已保存到CSV文件: {metrics_filename}")

    # 打印主要指标
    logger.info(f"mAP@{iou_threshold}: {average_precision:.4f}")
    logger.info(f"Mean Recall: {average_recall:.4f}")
    logger.info(f"Mean F1 Score: {average_f1:.4f}")

    # 按AP值排序类别
    sorted_classes = sorted(
        [(c, class_metrics[c]["AP"]) for c in valid_classes],
        key=lambda x: x[1],
        reverse=True,
    )

    # 打印表现最好的5个类别
    logger.info("\n表现最好的5个解剖区域:")
    for class_id, ap in sorted_classes[:5]:
        metrics = class_metrics[class_id]
        logger.info(
            f"区域 {class_id}: AP={ap:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}"
        )

    # 打印表现最差的5个类别
    logger.info("\n表现最差的5个解剖区域:")
    for class_id, ap in sorted_classes[-5:]:
        metrics = class_metrics[class_id]
        logger.info(
            f"区域 {class_id}: AP={ap:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}"
        )

    # 记录评估指标到 TensorBoard
    if writer is not None and epoch is not None:
        writer.add_scalar(f"{mode}/Detection/mAP", metrics_data["mAP"], epoch)
        writer.add_scalar(
            f"{mode}/Detection/Mean_Recall", metrics_data["mRecall"], epoch
        )
        writer.add_scalar(f"{mode}/Detection/Mean_F1", metrics_data["mF1"], epoch)
        writer.add_scalar(f"{mode}/Detection/Loss", metrics_data["loss"], epoch)

        # 记录每个类别的指标
        for class_id, metrics in class_metrics.items():
            writer.add_scalar(
                f"{mode}/Detection/Class_{class_id}/AP", metrics["AP"], epoch
            )
            writer.add_scalar(
                f"{mode}/Detection/Class_{class_id}/Precision",
                metrics["precision"],
                epoch,
            )
            writer.add_scalar(
                f"{mode}/Detection/Class_{class_id}/Recall", metrics["recall"], epoch
            )
            writer.add_scalar(
                f"{mode}/Detection/Class_{class_id}/F1", metrics["f1_score"], epoch
            )

    # 构建返回结果
    result = {
        "overall_metrics": overall_metrics,
        "class_metrics": class_metrics,
        "mAP": average_precision,
        "mRecall": average_recall,
        "mF1": average_f1,
        "loss": running_loss / len(data_loader) if running_loss > 0 else 0.0,
        "metrics_df": metrics_df,
    }

    return running_loss / len(data_loader) if running_loss > 0 else 0.0, result


def test_vit(
    config,
    data_loader,
    model,
    logger,
    mode="val",
    device="cuda",
    epoch=None,
    writer=None,
    use_consistent_eval=False,  # 新增参数：是否使用一致性评估模式
):
    """
    评估PRETRAIN_VIT阶段的模型性能，只保留全局疾病分类性能评估

    参数:
        config: 配置参数
        data_loader: 测试数据加载器
        model: MOE模型实例
        logger: 日志记录器
        mode: 评估模式 ("val" 或 "test")
        device: 计算设备
        epoch: 当前训练轮次
        writer: TensorBoard写入器

    返回:
        float: 平均损失
        dict: 包含评估结果的字典
    """
    model.eval()
    running_loss = 0
    running_cls_loss = 0
    running_ltc_loss = 0

    # 获取ViT模型的总层数和实际有分类器的层数
    total_layers = model.image_encoder.num_layers
    classifier_layers = total_layers // 2  # 只有偶数层有分类器

    # 初始化存储结构
    image_paths_list = []
    labels_list = []
    all_disease_preds = []
    all_labels = []

    # 创建进度条
    prog_bar = tqdm(data_loader, desc=f"{mode} ViT Evaluation")

    with torch.no_grad():
        for i, batch in enumerate(prog_bar):
            # 收集图像路径和标签
            image_paths_list.extend(batch["image_path"])
            labels = batch["label"].to(device)
            labels_list.extend(labels.cpu().numpy().tolist())
            all_labels.append(labels)

            source, target, _ = prepare_batch_data(
                config,
                batch,
                data_loader,
                device,
                findings=True,
                history=False,
                label=True,
                bbox=True,
            )
            # 转换为kwargs格式
            source = args_to_kwargs(source)
            target = args_to_kwargs(target)

            source["phase"] = config.PHASE
            source["mode"] = "test"
            source["use_consistent_eval"] = use_consistent_eval  # 传递一致性评估参数

            outputs = data_distributor(model, source)
            outputs = args_to_kwargs(outputs)

            # 获取最后一层的疾病预测结果和损失
            if outputs["final_disease_preds"] is not None:
                disease_preds = outputs["final_disease_preds"]  # [B, num_diseases]
                all_disease_preds.append(disease_preds.detach().cpu())

                # 收集损失信息
                if "cls_loss" in outputs and outputs["cls_loss"] is not None:
                    running_loss += outputs["cls_loss"].item()
                    running_cls_loss += outputs["cls_loss"].item()

                # 如果有ltc_loss，也加到总损失中
                if "ltc_loss" in outputs and outputs["ltc_loss"] is not None:
                    running_loss += outputs["ltc_loss"].item()
                    running_ltc_loss += outputs["ltc_loss"].item()

    # 合并所有批次的预测和标签
    all_labels = torch.cat(all_labels, dim=0)
    all_disease_preds = torch.cat(all_disease_preds, dim=0)

    # 计算评估指标
    logger.info("计算ViT模型疾病分类评估指标...")

    # 创建结果目录
    results_dir = os.path.join(config.CHECKPOINT_PATH_TO, "vit_results")
    os.makedirs(results_dir, exist_ok=True)

    # 初始化结果数据
    metrics_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "epoch": str(epoch) if epoch is not None else "TEST",
    }

    # 应用sigmoid获取概率
    disease_probs = torch.sigmoid(all_disease_preds)

    # 使用0.5作为阈值获取二值预测
    disease_binary = (disease_probs > 0.5).float()

    # 计算全局疾病分类指标 - 手动计算而不是使用sklearn

    # 将张量转换为CPU上的张量进行计算
    disease_binary = disease_binary.cpu()
    disease_probs = disease_probs.cpu()
    all_labels = all_labels.cpu()

    # 计算每个样本的TP, FP, FN
    tp = (disease_binary == 1) & (all_labels == 1)  # 真阳性：预测有疾病且真实有疾病
    fp = (disease_binary == 1) & (all_labels == 0)  # 假阳性：预测有疾病但真实无疾病
    fn = (disease_binary == 0) & (all_labels == 1)  # 假阴性：预测无疾病但真实有疾病

    # 对每个样本的每个疾病求和，得到每个样本的TP, FP, FN总数
    tp_sum = tp.sum(dim=1).float()  # [N]
    fp_sum = fp.sum(dim=1).float()  # [N]
    fn_sum = fn.sum(dim=1).float()  # [N]

    # 计算每个样本的精确率、召回率、F1分数
    # 注意处理分母为0的情况
    precision_per_sample = torch.zeros_like(tp_sum)
    recall_per_sample = torch.zeros_like(tp_sum)
    f1_per_sample = torch.zeros_like(tp_sum)

    # 只在有预测的样本上计算精确率
    valid_precision = (tp_sum + fp_sum) > 0
    precision_per_sample[valid_precision] = tp_sum[valid_precision] / (
        tp_sum[valid_precision] + fp_sum[valid_precision]
    )

    # 只在有真实正样本的样本上计算召回率
    valid_recall = (tp_sum + fn_sum) > 0
    recall_per_sample[valid_recall] = tp_sum[valid_recall] / (
        tp_sum[valid_recall] + fn_sum[valid_recall]
    )

    # 计算F1（注意避免除以0）
    valid_f1 = (precision_per_sample + recall_per_sample) > 0
    f1_per_sample[valid_f1] = (
        2
        * precision_per_sample[valid_f1]
        * recall_per_sample[valid_f1]
        / (precision_per_sample[valid_f1] + recall_per_sample[valid_f1])
    )

    # 计算平均指标（样本级别）
    precision = precision_per_sample.mean().item()
    recall = recall_per_sample.mean().item()
    f1 = f1_per_sample.mean().item()

    # 计算每个类别的指标（类别级别）
    tp_per_class = tp.sum(dim=0).float()  # [num_diseases]
    fp_per_class = fp.sum(dim=0).float()  # [num_diseases]
    fn_per_class = fn.sum(dim=0).float()  # [num_diseases]

    precision_per_class = torch.zeros_like(tp_per_class)
    recall_per_class = torch.zeros_like(tp_per_class)
    f1_per_class = torch.zeros_like(tp_per_class)

    valid_precision_class = (tp_per_class + fp_per_class) > 0
    precision_per_class[valid_precision_class] = tp_per_class[valid_precision_class] / (
        tp_per_class[valid_precision_class] + fp_per_class[valid_precision_class]
    )

    valid_recall_class = (tp_per_class + fn_per_class) > 0
    recall_per_class[valid_recall_class] = tp_per_class[valid_recall_class] / (
        tp_per_class[valid_recall_class] + fn_per_class[valid_recall_class]
    )

    valid_f1_class = (precision_per_class + recall_per_class) > 0
    f1_per_class[valid_f1_class] = (
        2
        * precision_per_class[valid_f1_class]
        * recall_per_class[valid_f1_class]
        / (precision_per_class[valid_f1_class] + recall_per_class[valid_f1_class])
    )

    # 计算类别平均（宏平均）
    precision_macro = precision_per_class.mean().item()
    recall_macro = recall_per_class.mean().item()
    f1_macro = f1_per_class.mean().item()

    # 计算准确率 - 整体准确率
    correct = (disease_binary == all_labels).float()
    accuracy = correct.mean().item()

    # 计算每个类别的AUC和AP (如果需要的话)
    num_diseases = all_disease_preds.size(1)
    aucs = []
    aps = []

    # 将张量转换为NumPy数组以便使用sklearn
    disease_probs_np = disease_probs.numpy()
    labels_np = all_labels.numpy()

    for i in range(num_diseases):
        # 只有当类别有正样本和负样本时才计算AUC
        if len(np.unique(labels_np[:, i])) > 1:
            try:
                auc = roc_auc_score(labels_np[:, i], disease_probs_np[:, i])
                ap = average_precision_score(labels_np[:, i], disease_probs_np[:, i])
                aucs.append(auc)
                aps.append(ap)
            except Exception as e:
                logger.warning(f"计算类别 {i} 的AUC/AP时出错: {e}")

    # 计算宏平均AUC和AP
    macro_auc = np.mean(aucs) if aucs else 0
    macro_ap = np.mean(aps) if aps else 0

    # 将指标保存到结果中
    disease_metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "auc_macro": macro_auc,
        "ap_macro": macro_ap,
    }

    # 计算平均损失
    avg_loss = running_loss / len(data_loader) if running_loss > 0 else 0.0
    avg_cls_loss = running_cls_loss / len(data_loader) if running_cls_loss > 0 else 0.0
    avg_ltc_loss = running_ltc_loss / len(data_loader) if running_ltc_loss > 0 else 0.0

    # 添加到指标数据
    metrics_data.update(
        {
            "disease_classification": disease_metrics,
            "loss": avg_loss,
            "cls_loss": avg_cls_loss,
            "ltc_loss": avg_ltc_loss,
        }
    )

    # 记录到wandb或TensorBoard
    if writer is not None and epoch is not None:
        writer.add_scalar(f"{mode}/Disease/Accuracy", accuracy, epoch)
        writer.add_scalar(f"{mode}/Disease/Precision", precision, epoch)
        writer.add_scalar(f"{mode}/Disease/Recall", recall, epoch)
        writer.add_scalar(f"{mode}/Disease/F1", f1, epoch)
        writer.add_scalar(f"{mode}/Disease/AUC", macro_auc, epoch)
        writer.add_scalar(f"{mode}/Disease/AP", macro_ap, epoch)
        # 记录损失
        writer.add_scalar(f"{mode}/ViT/Loss", avg_loss, epoch)
        # 分别记录cls_loss和ltc_loss
        writer.add_scalar(f"{mode}/ViT/CLS_Loss", avg_cls_loss, epoch)
        writer.add_scalar(f"{mode}/ViT/LTC_Loss", avg_ltc_loss, epoch)

    # 打印评估结果
    logger.info(f"全局疾病分类性能 (Epoch {epoch}):")
    logger.info(f"  准确率: {accuracy:.4f}")
    logger.info(f"  精确率: {precision:.4f}")
    logger.info(f"  召回率: {recall:.4f}")
    logger.info(f"  F1分数: {f1:.4f}")
    logger.info(f"  宏平均AUC: {macro_auc:.4f}")
    logger.info(f"  宏平均AP: {macro_ap:.4f}")
    logger.info(f"  平均总损失: {avg_loss:.4f}")
    logger.info(f"  平均分类损失: {avg_cls_loss:.4f}")
    logger.info(f"  平均LTC损失: {avg_ltc_loss:.4f}")

    # 保存到文件
    result_file = os.path.join(results_dir, f"{mode}_epoch_{epoch}_vit_results.json")
    with open(result_file, "w") as f:
        json.dump(metrics_data, f, indent=2)

    logger.info(f"评估结果已保存到 {result_file}")

    # 构建返回结果
    result = {
        "overall_metrics": {
            "ce_accuracy": accuracy,
            "ce_precision": precision,
            "ce_recall": recall,
            "ce_f1": f1,
            "ce_auc": macro_auc,
            "ce_ap": macro_ap,
        },
        "metrics_data": metrics_data,
        "loss": metrics_data["loss"],
    }

    return running_loss / len(data_loader) if running_loss > 0 else 0.0, result


def calculate_detection_metrics(predictions, ground_truths, iou_threshold=0.5):
    """
    计算目标检测的整体指标

    参数:
        predictions: 预测结果列表
        ground_truths: 真值列表
        iou_threshold: IoU阈值

    返回:
        dict: 包含mAP等指标的字典
    """
    if not predictions or not ground_truths:
        return {"mAP": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}

    # 按图像ID和类别整理预测和真值
    pred_by_img_cls = {}
    gt_by_img_cls = {}

    for pred in predictions:
        img_id = pred["image_id"]
        for i, (box, label, score) in enumerate(
            zip(pred["boxes"], pred["labels"], pred["scores"])
        ):
            key = (img_id, label.item())
            if key not in pred_by_img_cls:
                pred_by_img_cls[key] = []
            pred_by_img_cls[key].append((box, score))

    for gt in ground_truths:
        img_id = gt["image_id"]
        for i, (box, label) in enumerate(zip(gt["boxes"], gt["labels"])):
            key = (img_id, label.item())
            if key not in gt_by_img_cls:
                gt_by_img_cls[key] = []
            gt_by_img_cls[key].append(box)

    # 计算所有类别的AP
    all_classes = set(
        [k[1] for k in pred_by_img_cls.keys()] + [k[1] for k in gt_by_img_cls.keys()]
    )
    aps = []

    for cls in all_classes:
        # 收集此类别的所有预测和真值
        all_preds = []
        all_gt_count = 0

        for img_id in set(
            [k[0] for k in pred_by_img_cls.keys()]
            + [k[0] for k in gt_by_img_cls.keys()]
        ):
            key = (img_id, cls)

            # 获取此图像中此类别的预测
            img_preds = pred_by_img_cls.get(key, [])
            img_preds.sort(key=lambda x: x[1], reverse=True)  # 按置信度排序

            # 获取此图像中此类别的真值
            img_gts = gt_by_img_cls.get(key, [])
            all_gt_count += len(img_gts)

            # 判断每个预测是否为TP
            gt_matched = [False] * len(img_gts)

            for pred_box, score in img_preds:
                # 计算与所有真值的IoU
                max_iou = 0
                max_idx = -1

                for j, gt_box in enumerate(img_gts):
                    if gt_matched[j]:
                        continue

                    # 计算IoU
                    iou = box_iou(pred_box.unsqueeze(0), gt_box.unsqueeze(0)).item()
                    if iou > max_iou:
                        max_iou = iou
                        max_idx = j

                # 添加预测结果 (score, tp)
                if max_idx >= 0 and max_iou >= iou_threshold:
                    all_preds.append((score.item(), 1))
                    gt_matched[max_idx] = True
                else:
                    all_preds.append((score.item(), 0))

        # 计算此类别的AP
        if all_preds:
            # 按置信度排序
            all_preds.sort(reverse=True)

            # 计算累积TP和FP
            tp_cumsum = 0
            fp_cumsum = 0
            precisions = []
            recalls = []

            for score, tp in all_preds:
                if tp == 1:
                    tp_cumsum += 1
                else:
                    fp_cumsum += 1

                precision = tp_cumsum / (tp_cumsum + fp_cumsum)
                recall = tp_cumsum / max(all_gt_count, 1)

                precisions.append(precision)
                recalls.append(recall)

            # 计算AP (11点插值法)
            ap = 0
            for t in np.arange(0, 1.1, 0.1):
                if not recalls or max(recalls) < t:
                    p = 0
                else:
                    # 找到所有大于等于t的recall点
                    p_vals = [
                        precisions[i] for i in range(len(recalls)) if recalls[i] >= t
                    ]
                    p = max(p_vals) if p_vals else 0
                ap += p / 11

            aps.append(ap)

    # 计算平均AP
    mAP = np.mean(aps) if aps else 0.0

    # 计算整体精确度、召回率和F1
    total_tp = sum(
        pred[1] for preds in pred_by_img_cls.values() for pred in preds if pred[1] == 1
    )
    total_fp = sum(
        1 for preds in pred_by_img_cls.values() for pred in preds if pred[1] == 0
    )
    total_gt = sum(len(gts) for gts in gt_by_img_cls.values())

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
    recall = total_tp / total_gt if total_gt > 0 else 0.0
    f1_score = (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    )

    return {"mAP": mAP, "precision": precision, "recall": recall, "f1_score": f1_score}


def calculate_class_metrics(class_predictions, class_ground_truths, iou_threshold=0.5):
    """
    计算单个类别的检测指标

    参数:
        class_predictions: 此类别的预测列表
        class_ground_truths: 此类别的真值列表
        iou_threshold: IoU阈值

    返回:
        dict: 包含AP等指标的字典
    """
    # 计算预测总数和真值总数
    total_preds = sum(len(pred["boxes"]) for pred in class_predictions)
    total_gts = sum(len(gt["boxes"]) for gt in class_ground_truths)

    # 如果没有预测或真值，返回零指标
    if total_preds == 0 or total_gts == 0:
        return {
            "AP": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "TP": 0,
            "FP": total_preds,
            "FN": total_gts,
            "num_samples": total_gts,
        }

    # 初始化TP、FP计数和得分列表
    all_scores = []
    all_tp = []

    # 对每张图像进行评估
    for i, (pred, gt) in enumerate(zip(class_predictions, class_ground_truths)):
        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]
        gt_boxes = gt["boxes"]

        # 跳过没有预测或真值的图像
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            continue

        # 计算所有预测框和真值框之间的IoU
        ious = box_iou(pred_boxes, gt_boxes)

        # 对每个预测框，找到最大IoU的真值框
        max_ious, max_idx = ious.max(dim=1)

        # 初始化匹配标记
        gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool)

        # 按置信度排序预测框
        _, sorted_idx = torch.sort(pred_scores, descending=True)

        # 判断每个预测是TP还是FP
        for idx in sorted_idx:
            if max_ious[idx] >= iou_threshold:
                gt_idx = max_idx[idx].item()
                if not gt_matched[gt_idx]:
                    # 真阳性
                    all_tp.append(1)
                    gt_matched[gt_idx] = True
                else:
                    # 假阳性(重复检测)
                    all_tp.append(0)
            else:
                # 假阳性(IoU过低)
                all_tp.append(0)

            all_scores.append(pred_scores[idx].item())

    # 如果没有有效预测，返回零指标
    if not all_scores:
        return {
            "AP": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "TP": 0,
            "FP": total_preds,
            "FN": total_gts,
            "num_samples": total_gts,
        }

    # 按置信度排序
    sorted_indices = np.argsort(all_scores)[::-1]
    all_tp = [all_tp[i] for i in sorted_indices]

    # 计算累积TP和FP
    tp_cumsum = np.cumsum(all_tp)
    fp_cumsum = np.cumsum([1 - tp for tp in all_tp])

    # 计算精确度和召回率
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    recalls = tp_cumsum / total_gts

    # 计算AP (11点插值法)
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if len(recalls) == 0 or max(recalls) < t:
            p = 0
        else:
            p_vals = [precisions[i] for i in range(len(recalls)) if recalls[i] >= t]
            p = max(p_vals) if p_vals else 0
        ap += p / 11

    # 计算最终TP、FP、FN和整体指标
    TP = sum(all_tp)
    FP = len(all_tp) - TP
    FN = total_gts - TP

    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / total_gts if total_gts > 0 else 0.0
    f1_score = (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    )

    return {
        "AP": ap,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "TP": int(TP),
        "FP": int(FP),
        "FN": int(FN),
        "num_samples": total_gts,
    }


def analyze_gpu_memory():
    """分析当前GPU内存使用情况，并打印内存占用最大的前20个张量"""
    print("\n--------------------GPU内存分析--------------------")

    # 收集所有张量信息
    tensor_info = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                # 计算内存占用（MB）
                memory = obj.element_size() * obj.nelement() / 1024 / 1024

                # 收集张量信息
                info = {
                    "type": type(obj),
                    "size": tuple(obj.size()),
                    "dtype": obj.dtype,
                    "device": obj.device,
                    "memory": memory,
                    "tensor": obj,  # 保存张量对象以便后续查找变量名
                }
                tensor_info.append(info)
        except:
            pass

    # 按内存占用降序排序
    tensor_info.sort(key=lambda x: x["memory"], reverse=True)

    # 显示前20个最大的张量
    total_memory = 0
    print(f"{'内存(MB)':>10} | {'类型':<15} | {'大小':<20} | {'数据类型':<10} | {'设备':<10}")
    print("-" * 80)

    for i, info in enumerate(tensor_info[:20]):
        total_memory += info["memory"]
        print(
            f"{info['memory']:>10.2f} | {str(info['type'].__name__):<15} | {str(info['size']):<20} | {str(info['dtype']).split('.')[-1]:<10} | {str(info['device']):<10}"
        )

    print("-" * 80)
    print(f"前20个张量总内存: {total_memory:.2f} MB")
    print(f"GPU已分配内存: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"GPU缓存内存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    print("------------------------------------------------\n")


def build_anatomical_database(
    config,
    model,
    data_loader,
    logger,
    device="cuda"
):
    """
    构建解剖区域特征数据库（分批保存版）
    
    Args:
        config: 配置对象
        model: MOE模型
        data_loader: 数据加载器（通常是训练集）
        logger: 日志记录器
        device: 计算设备
    """
    import numpy as np
    from datasets import MIMIC
    
    logger.info(f"开始构建解剖区域特征数据库...")
    logger.info(f"数据集大小: {len(data_loader.dataset)}")
    logger.info(f"每1000个batch保存一次")
    
    # 创建输出目录
    output_dir = config.DATABASE_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取解剖区域名称
    anatomical_regions = MIMIC.ANATOMICAL_REGIONS
    num_regions = len(anatomical_regions)
    
    logger.info(f"解剖区域数量: {num_regions}")
    
    # 设置模型为评估模式
    model.eval()
    
    # 分批保存参数
    save_every_n_batches = 300
    current_part = 0
    
    # 初始化当前批次的存储结构
    def init_storage():
        return {}
    
    def save_current_batch(image_database, part_num):
        """保存当前批次的数据"""
        save_format = config.SAVE_FEATURES_FORMAT.lower()
        
        if save_format == "npz":
            # NPZ格式直接存储image_database
            filename = f"anatomical_database_part_{part_num:04d}.npz"
            np.savez_compressed(
                os.path.join(output_dir, filename),
                image_database=image_database
            )
            logger.info(f"第{part_num}部分已保存为NPZ格式: {filename}")
            
        elif save_format == "pkl":
            # Pickle格式直接存储image_database
            filename = f"anatomical_database_part_{part_num:04d}.pkl"
            pkl_path = os.path.join(output_dir, filename)
            with open(pkl_path, 'wb') as f:
                pickle.dump(image_database, f)
            logger.info(f"第{part_num}部分已保存为Pickle格式: {filename}")
        
        return filename
    
    def cleanup_memory():
        """清理内存和显存"""
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("内存和显存已清理")
    
    # 初始化存储
    image_database = init_storage()
    total_samples = 0
    total_detected_regions = 0
    saved_files = []
    
    # 处理每个批次
    with torch.no_grad():
        prog_bar = tqdm(data_loader, desc="构建数据库")
        
        for batch_idx, batch in enumerate(prog_bar):
            # 准备批次数据
            source, target, _ = prepare_batch_data(
                config,
                batch,
                data_loader,
                device,
                findings=False,  # 不需要findings
                history=False,   # 不需要history
                label=False,     # 不需要疾病标签
                bbox=True,       # 需要边界框
            )
            
            # 转换为kwargs格式
            source = args_to_kwargs(source)
            source["phase"] = "BUILD_DATABASE"
            source["mode"] = "eval"
            
            # 获取图像ID
            image_ids = batch["image_id"]
            
            # 模型推理
            outputs = data_distributor(model, source)
            
            # 提取特征
            region_features = outputs["region_features"]  # [B, 29, 768]
            region_detected = outputs["region_detected"]  # [B, 29]
            
            batch_size = region_features.size(0)
            
            # 处理每个样本
            for i in range(batch_size):
                sample_features = region_features[i].cpu().numpy()  # [29, 768]
                sample_detected = region_detected[i].cpu().numpy()  # [29]
                image_id = image_ids[i]
                sample_idx = total_samples + i
                
                # 构建该图像的区域特征字典
                image_regions = {}
                sample_detected_count = 0
                
                for region_idx in range(num_regions):
                    if sample_detected[region_idx]:  # 只保存检测到的区域
                        image_regions[region_idx] = {
                            "region_name": anatomical_regions[region_idx],
                            "feature": sample_features[region_idx]  # [768]
                        }
                        sample_detected_count += 1
                        total_detected_regions += 1
                
                # 保存到数据库
                image_database[image_id] = {
                    "regions": image_regions,
                    "sample_idx": sample_idx,
                    "detected_count": sample_detected_count
                }
            
            total_samples += batch_size
            
            # 更新进度条信息
            avg_detected = total_detected_regions / total_samples if total_samples > 0 else 0
            prog_bar.set_description(
                f"样本: {total_samples}, 平均检测区域: {avg_detected:.1f}/{num_regions}, 当前部分: {current_part}"
            )
            
            # 检查是否需要保存
            if (batch_idx + 1) % save_every_n_batches == 0:
                logger.info(f"\n到达{save_every_n_batches}个batch，开始保存第{current_part}部分...")
                
                # 保存当前批次
                filename = save_current_batch(image_database, current_part)
                saved_files.append(filename)
                
                # 统计当前部分信息
                current_images = len(image_database)
                current_detected = sum(img_data['detected_count'] for img_data in image_database.values())
                
                logger.info(f"第{current_part}部分统计:")
                logger.info(f"  图像数: {current_images}")
                logger.info(f"  检测区域数: {current_detected}")
                logger.info(f"  平均每图像检测区域数: {current_detected / current_images:.2f}")
                
                # 清理当前数据
                image_database = init_storage()
                current_part += 1
                
                # 清理内存
                cleanup_memory()
                
                logger.info(f"开始处理第{current_part}部分...\n")
    
    # 保存最后剩余的数据（如果有）
    if image_database:
        logger.info(f"\n保存最后剩余的数据（第{current_part}部分）...")
        filename = save_current_batch(image_database, current_part)
        saved_files.append(filename)
        
        # 统计最后部分信息
        current_images = len(image_database)
        current_detected = sum(img_data['detected_count'] for img_data in image_database.values())
        
        logger.info(f"第{current_part}部分统计:")
        logger.info(f"  图像数: {current_images}")
        logger.info(f"  检测区域数: {current_detected}")
        logger.info(f"  平均每图像检测区域数: {current_detected / current_images:.2f}")
    
    # 最终清理
    cleanup_memory()
    
    logger.info(f"\n=== 数据库构建完成 ===")
    logger.info(f"总样本数: {total_samples}")
    logger.info(f"总检测区域数: {total_detected_regions}")
    logger.info(f"平均每样本检测区域数: {total_detected_regions / total_samples:.2f}")
    logger.info(f"总共保存了 {len(saved_files)} 个文件:")
    
    for i, filename in enumerate(saved_files):
        logger.info(f"  {i+1}. {filename}")
    
    logger.info(f"\n数据库文件保存在: {output_dir}")
    
    # 创建文件列表
    files_list_path = os.path.join(output_dir, "database_files.txt")
    with open(files_list_path, 'w') as f:
        for filename in saved_files:
            f.write(f"{filename}\n")
    logger.info(f"文件列表已保存: database_files.txt")
    
    return {
        "output_dir": output_dir,
        "total_samples": total_samples,
        "total_detected_regions": total_detected_regions,
        "saved_files": saved_files,
        "num_parts": len(saved_files),
    }


def get_memory_profiler(enable_profile=False, log_path=None):
    """创建内存分析器的上下文管理器"""
    if enable_profile:
        return profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
            with_modules=True,
            with_flops=True,
            on_trace_ready=(
                torch.profiler.tensorboard_trace_handler(log_path) if log_path else None
            ),
        )
    return nullcontext()


def test_llm(
    config,
    data_loader,
    model,
    logger,
    mode="val",
    metric_ftns=None,
    device="cuda",
    epoch=None,
    writer=None,
    chexbert_metrics=None,  # 新增CheXbert评估器参数
):
    """测试语言模型的生成效果（支持MISTRAL/LLAMA/BERT）

    Args:
        config: 配置对象
        data_loader: 数据加载器
        model: MOE模型
        logger: 日志记录器
        mode: 测试模式（val或test）
        metric_ftns: 计算指标的函数
        device: 设备
        epoch: 当前训练轮数
        writer: TensorBoard写入器
        chexbert_metrics: CheXbert评估指标计算器

    Returns:
        test_loss: 测试损失
        result: 测试结果（包含各种评估指标）
    """
    torch.cuda.empty_cache()
    gc.collect()
    model.eval()

    # 记录总测试损失和样本数
    running_loss = 0
    total_samples = 0

    # 存储所有的预测结果和真实值以及元数据
    all_preds = []
    all_targets = []
    image_paths_list = []
    labels_list = []

    # 设置进度条
    prog_bar = tqdm(data_loader, desc=f"{mode} Evaluation")

    with torch.no_grad():
        # 遍历批次数据
        for batch_idx, batch in enumerate(prog_bar):
            # 收集元数据
            image_paths_list.extend(batch["image_path"])
            if "label" in batch:
                labels_list.extend(batch["label"].cpu().numpy().tolist())
                
            # 在这里先保存原始的findings字符串
            original_findings = batch["findings"]
            # 确保findings是字符串列表
            if isinstance(original_findings, list) and all(isinstance(f, str) for f in original_findings):
                target_texts = original_findings.copy()  # 直接使用原始字符串列表
            else:
                # 如果不是字符串列表，初始化为空列表，后续再填充
                target_texts = []

            # 准备批次数据
            source, target, _ = prepare_batch_data(
                config,
                batch,
                data_loader,
                device,
                findings=True,
                history=True,
                label=True,
                bbox=True,
            )

            # 转换为kwargs格式
            source = args_to_kwargs(source)
            target = args_to_kwargs(target)

            # 设置模型阶段和模式
            source["phase"] = config.PHASE
            source["mode"] = "test"
            
            # 提取image_ids用于检测缓存
            image_ids = [os.path.basename(path).split('.')[0] for path in batch["image_path"]]
            source["image_ids"] = image_ids

            # 使用数据分发器执行模型推理
            outputs = data_distributor(model, source)

            # 获取批次大小
            batch_size = (
                source["image"].size(0) if "image" in source else len(batch["findings"])
            )

            # 记录损失（如果有）
            if (hasattr(outputs, "loss") and outputs.loss is not None):
                loss = outputs.loss
                running_loss += loss.item() * batch_size
                total_samples += batch_size

            # 提取生成的文本
            if isinstance(outputs, dict) and "findings_text" in outputs:
                # MOE模型的输出格式
                generated_texts = outputs["findings_text"]
            elif hasattr(outputs, "decoded_texts"):
                # BERT模型的输出格式
                generated_texts = outputs.decoded_texts
            else:
                logger.error(f"不支持的输出格式: {type(outputs)}")
                generated_texts = ["生成失败"] * batch_size

            # 如果target_texts为空（不是字符串列表的情况），则需要解码获取
            if not target_texts:
                # 检查batch["findings"]是否为字符串列表
                if "findings" in batch and isinstance(batch["findings"], list) and len(batch["findings"]) > 0 and isinstance(batch["findings"][0], str):
                    target_texts = batch["findings"]
                # 检查batch["findings"]是否为BatchEncoding类型
                elif "findings" in batch and hasattr(batch["findings"], "input_ids"):
                    # 处理BatchEncoding对象
                    target_texts = []
                    for idx in range(batch_size):
                        findings_ids = batch["findings"].input_ids[idx]
                        if hasattr(model.findings_decoder, "tokenizer"):
                            tokenizer = model.findings_decoder.tokenizer
                            target_texts.append(
                                tokenizer.decode(findings_ids, skip_special_tokens=True)
                            )
                        elif hasattr(model.findings_decoder, "decoder") and hasattr(model.findings_decoder.decoder, "tokenizer"):
                            tokenizer = model.findings_decoder.decoder.tokenizer
                            target_texts.append(
                                tokenizer.decode(findings_ids, skip_special_tokens=True)
                            )
                        else:
                            target_texts.append(f"[BatchEncoding]")
                # 检查target中的findings
                elif "findings" in target and "input_ids" in target["findings"]:
                    # 如果findings是已编码的token IDs，进行解码
                    target_texts = []
                    for idx in range(batch_size):
                        findings_ids = target["findings"]["input_ids"][idx]
                        # 尝试使用模型内部的tokenizer解码
                        if hasattr(model.findings_decoder, "tokenizer"):
                            tokenizer = model.findings_decoder.tokenizer
                            target_texts.append(
                                tokenizer.decode(findings_ids, skip_special_tokens=True)
                            )
                        elif hasattr(model.findings_decoder, "decoder") and hasattr(model.findings_decoder.decoder, "tokenizer"):
                            # 针对BERT解码器的特殊处理
                            tokenizer = model.findings_decoder.decoder.tokenizer
                            target_texts.append(
                                tokenizer.decode(findings_ids, skip_special_tokens=True)
                            )
                        else:
                            # 如果无法直接访问tokenizer，可以将ID保存为字符串
                            target_texts.append(f"[IDs:{findings_ids.tolist()}]")
                else:
                    target_texts = ["[无目标文本]"] * batch_size

            # 收集预测结果和真实值
            all_preds.extend(generated_texts)
            all_targets.extend(target_texts)

            # 更新进度条
            prog_bar.set_description(f"Loss: {running_loss/(batch_idx+1):.4f}")

    # 计算平均损失
    avg_loss = running_loss / max(total_samples, 1)

    # 创建结果数据字典
    results_data = {
        "image_path": image_paths_list,
        "findings_pred": all_preds,
        "findings_gt": all_targets,
        "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")] * len(all_targets),
    }
    
    # 如果有标签数据，也添加进去
    if labels_list:
        results_data["labels"] = labels_list

    # 计算评估指标 - 使用与test函数相同的方式
    report_metrics = {}
    if metric_ftns is not None:
        report_metrics = metric_ftns(
            {i: [gt] for i, gt in enumerate(all_targets)},
            {i: [pred] for i, pred in enumerate(all_preds)},
        )
        
        # 记录评估指标
        for metric_name, value in report_metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")

    # 计算CheXbert临床评估指标
    ce_metrics = {}
    if chexbert_metrics is not None:
        logger.info("计算CheXbert临床评估指标...")
        try:
            ce_metrics = chexbert_metrics.compute(all_targets, all_preds)
            # 记录CheXbert指标
            for metric_name, value in ce_metrics.items():
                logger.info(f"CheXbert - {metric_name}: {value:.4f}")
        except Exception as e:
            logger.error(f"计算CheXbert指标时出错: {e}")

    # 创建结果目录
    results_dir = os.path.join(config.CHECKPOINT_PATH_TO, "test_results")
    os.makedirs(results_dir, exist_ok=True)

    # 将结果转换为DataFrame并保存
    results_df = pd.DataFrame(results_data)

    # 保存为CSV文件，添加epoch信息
    epoch_str = str(epoch) if epoch is not None else "TEST"
    csv_filename = f"{mode}_results_epoch_{epoch_str}.csv"
    results_df.to_csv(os.path.join(results_dir, csv_filename), index=False)
    logger.info(f"结果已保存到CSV文件: {csv_filename}")
    
    # 计算并保存评估指标
    metrics_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "epoch": epoch_str,
        "loss": avg_loss,
    }
    
    # 添加常规评估指标
    if report_metrics:
        for metric_name, value in report_metrics.items():
            metrics_data[f"{metric_name}"] = value
            
    # 添加CheXbert指标
    if ce_metrics:
        for metric_name, value in ce_metrics.items():
            metrics_data[f"ce_{metric_name}"] = value

    # 保存评估指标，添加epoch信息
    metrics_df = pd.DataFrame([metrics_data])
    metrics_filename = f"{mode}_metrics_epoch_{epoch_str}.csv"
    metrics_df.to_csv(os.path.join(results_dir, metrics_filename), index=False)
    logger.info(f"评估指标已保存到CSV文件: {metrics_filename}")
    
    # 如果有writer和epoch，记录到TensorBoard
    if writer is not None and epoch is not None:
        # 记录常规指标
        for metric_name, value in report_metrics.items():
            writer.add_scalar(f"{mode}/{metric_name}", value, epoch)
        
        # 记录CheXbert指标
        for metric_name, value in ce_metrics.items():
            writer.add_scalar(f"{mode}/CheXbert/{metric_name}", value, epoch)
            
        # 记录损失
        writer.add_scalar(f"{mode}/loss", avg_loss, epoch)

    # 汇总结果
    result = {
        "report_generation_metrics": report_metrics,
        "chexbert_metrics": ce_metrics,
        "loss": avg_loss,
        "results_df": results_df,
        "metrics_df": metrics_df,
    }

    return avg_loss, result