"""评估相关工具函数"""
import os
import re
import gc
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from utils.data_utils import prepare_batch_data, args_to_kwargs
from utils.detection_metrics import (
    calculate_comprehensive_metrics,
    generate_detection_report,
    save_results_to_csv
)
from utils.logging_utils import clean_report_mimic_cxr
import metrics


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
        for batch_idx, batch in enumerate(prog_bar):
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

            # 模型推理（直接使用**kwargs传递）
            output = model(**source)
            output = args_to_kwargs(output, kw_out)

            # 收集预测结果
            findings_preds_list.extend([re for re in output["findings_text"]])
            
            # 【修复】及时清理batch数据
            del source, target, output, batch
            
            # 定期深度清理
            if batch_idx % 50 == 0 and batch_idx > 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

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

            # 模型推理（直接使用**kwargs传递）
            output = model(**source)
            output = args_to_kwargs(output, kw_out)

            # 收集预测结果
            findings_preds_list.extend([re for re in output["findings_text"]])

            # 计算损失
            if criterion is not None:
                loss = torch.tensor(0.0)
                running_loss += loss.item()
            prog_bar.set_description("Loss: {}".format(running_loss / (i + 1)))
            
            del source, target, output, batch
            
            # 定期深度清理
            if i % 50 == 0 and i > 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

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


def test_detection(
    config,
    data_loader,
    model,
    logger,
    mode="test",
    confidence_threshold=0.5,
    device="cuda",
    epoch=None,
    writer=None,
):
    """
    全面的目标检测评估函数 - 适用于论文报告
    
    提供以下评估指标：
    1. 多IoU阈值下的mAP (mAP@0.3, mAP@0.5, mAP@0.75, mAP@0.9)
    2. 每个解剖区域的AP, Precision, Recall, F1-Score
    3. 检测率、TP/FP/FN统计
    4. 详细的文本报告和CSV结果（用于论文）
    
    Args:
        config: 配置参数
        data_loader: 数据加载器
        model: 检测模型
        logger: 日志记录器
        mode: 评估模式 ("val", "test", "validate")
        confidence_threshold: 检测置信度阈值
        device: 计算设备
        epoch: 当前训练轮次
        writer: TensorBoard writer
    
    Returns:
        avg_loss: 平均损失
        results: 包含详细评估结果的字典
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0
    
    # 存储所有预测和真值
    all_predictions = []
    all_ground_truths = []
    
    logger.info(f"开始{mode}集目标检测评估...")
    logger.info(f"置信度阈值: {confidence_threshold}")
    
    # 创建进度条
    prog_bar = tqdm(data_loader, desc=f"{mode.capitalize()} Detection")
    
    with torch.no_grad():
        for batch in prog_bar:
            # 准备数据
            images = batch["image"].to(device)
            targets = []
            
            for target_dict in batch["bbox_targets"]:
                target = {
                    "boxes": target_dict["boxes"].to(device),
                    "labels": target_dict["labels"].to(device),
                    "image_id": target_dict["image_id"].to(device),
                    "area": target_dict["area"].to(device),
                    "iscrowd": target_dict["iscrowd"].to(device),
                }
                targets.append(target)
            
            # 计算损失（训练模式）
            if targets:
                try:
                    model.train()  # 临时切换到训练模式以计算损失
                    loss_dict = model(images, targets)
                    model.eval()  # 切回评估模式
                    
                    if isinstance(loss_dict, dict):
                        # 【修复】立即计算并释放loss_dict，避免保留计算图
                        batch_loss = 0.0
                        for loss_value in loss_dict.values():
                            if isinstance(loss_value, torch.Tensor):
                                batch_loss += loss_value.item()
                        running_loss += batch_loss
                        num_batches += 1
                        # 立即删除loss_dict
                        del loss_dict, batch_loss
                except Exception as e:
                    logger.warning(f"损失计算异常: {e}")
            
            # 获取检测结果（推理模式）
            detections = model(images)
            
            # 处理每个图像的检测结果
            for detection, target in zip(detections, targets):
                # 应用置信度阈值
                keep = detection["scores"] > confidence_threshold
                # 【修复】立即detach并转为numpy，减少内存占用
                pred_boxes = detection["boxes"][keep].detach().cpu()
                pred_labels = detection["labels"][keep].detach().cpu()
                pred_scores = detection["scores"][keep].detach().cpu()
                
                # 存储预测结果
                all_predictions.append({
                    "boxes": pred_boxes,
                    "labels": pred_labels,
                    "scores": pred_scores,
                    "image_id": len(all_predictions)
                })
                
                # 存储真值
                all_ground_truths.append({
                    "boxes": target["boxes"].detach().cpu(),
                    "labels": target["labels"].detach().cpu(),
                    "image_id": len(all_ground_truths)
                })
            
            # 【修复】及时清理batch数据
            del images, targets, detections
            
            # 更新进度条
            if num_batches > 0:
                avg_loss = running_loss / num_batches
                prog_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
    
    # 计算平均损失
    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    
    # 计算全面的评估指标
    logger.info("计算全面的目标检测评估指标...")
    logger.info(f"总样本数: {len(all_predictions)}")
    
    # 使用多个IoU阈值计算指标
    iou_thresholds = getattr(config, 'DETECTION_IOU_THRESHOLDS', [0.3, 0.5, 0.75, 0.9])
    comprehensive_results = calculate_comprehensive_metrics(
        all_predictions,
        all_ground_truths,
        num_classes=29,
        iou_thresholds=iou_thresholds
    )
    
    # 创建结果目录
    results_dir = os.path.join(config.CHECKPOINT_PATH_TO, "detection_evaluation")
    os.makedirs(results_dir, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    epoch_str = f"epoch_{epoch}" if epoch is not None else "final"
    
    # 生成并保存详细报告
    report_path = os.path.join(results_dir, f"{mode}_{epoch_str}_report_{timestamp}.txt")
    report_text = generate_detection_report(comprehensive_results, save_path=report_path)
    
    # 打印报告到日志
    logger.info("\n" + report_text)
    
    # 保存CSV格式的结果（便于论文制表）
    csv_path = os.path.join(results_dir, f"{mode}_{epoch_str}_results_{timestamp}.csv")
    save_results_to_csv(comprehensive_results, csv_path)
    
    # 记录到TensorBoard
    if writer is not None and epoch is not None:
        overall = comprehensive_results['overall']
        
        # 记录总体指标
        writer.add_scalar(f"{mode}/Detection/mAP", overall['mAP'], epoch)
        writer.add_scalar(f"{mode}/Detection/mAP@0.3", overall['mAP@0.3'], epoch)
        writer.add_scalar(f"{mode}/Detection/mAP@0.5", overall['mAP@0.5'], epoch)
        writer.add_scalar(f"{mode}/Detection/mAP@0.75", overall['mAP@0.75'], epoch)
        writer.add_scalar(f"{mode}/Detection/Mean_Precision", overall['mean_precision'], epoch)
        writer.add_scalar(f"{mode}/Detection/Mean_Recall", overall['mean_recall'], epoch)
        writer.add_scalar(f"{mode}/Detection/Mean_F1", overall['mean_f1'], epoch)
        writer.add_scalar(f"{mode}/Detection/Loss", avg_loss, epoch)
        
        # 记录每个区域的AP
        per_class = comprehensive_results['per_class']
        for region_name, metrics in per_class.items():
            if region_name != 'mAP':
                # 清理region名称用于tensorboard
                clean_name = region_name.replace(' ', '_')
                writer.add_scalar(f"{mode}/Detection/Regions/{clean_name}/AP", metrics['ap'], epoch)
                writer.add_scalar(f"{mode}/Detection/Regions/{clean_name}/F1", metrics['f1'], epoch)
    
    # 构建返回结果
    result = {
        "comprehensive_results": comprehensive_results,
        "overall_metrics": comprehensive_results['overall'],
        "per_class_metrics": comprehensive_results['per_class'],
        "mAP": comprehensive_results['overall']['mAP'],
        "mAP@0.5": comprehensive_results['overall']['mAP@0.5'],
        "mean_precision": comprehensive_results['overall']['mean_precision'],
        "mean_recall": comprehensive_results['overall']['mean_recall'],
        "mean_f1": comprehensive_results['overall']['mean_f1'],
        "loss": avg_loss,
        "report_path": report_path,
        "csv_path": csv_path
    }
    
    logger.info(f"✅ 评估完成！")
    logger.info(f"📊 报告已保存至: {report_path}")
    logger.info(f"📊 CSV结果已保存至: {csv_path}")
    
    return avg_loss, result

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
    评估PRETRAIN_VIT阶段的模型性能，只评估region-level ITC损失

    参数:
        config: 配置参数
        data_loader: 测试数据加载器
        model: MedicalReportGenerator模型实例
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
    running_region_itc_loss = 0
    num_batches = 0

    # 创建进度条
    prog_bar = tqdm(data_loader, desc=f"{mode} ViT Evaluation")

    with torch.no_grad():
        for i, batch in enumerate(prog_bar):
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

            # 模型推理（直接使用**kwargs传递）
            outputs = model(**source)
            outputs = args_to_kwargs(outputs)

            # 收集region_itc_loss
            if "region_itc_loss" in outputs and outputs["region_itc_loss"] is not None:
                loss_val = outputs["region_itc_loss"].item()
                running_loss += loss_val
                running_region_itc_loss += loss_val
                num_batches += 1
            
            # 【修复】及时清理batch数据
            del source, target, outputs, batch

    # 计算平均损失
    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    avg_region_itc_loss = running_region_itc_loss / num_batches if num_batches > 0 else 0.0

    # 记录到TensorBoard
    if writer is not None and epoch is not None:
        writer.add_scalar(f"{mode}/ViT/Region_ITC_Loss", avg_region_itc_loss, epoch)

    # 打印评估结果
    logger.info(f"ViT预训练阶段评估 (Epoch {epoch}):")
    logger.info(f"  平均Region-ITC损失: {avg_region_itc_loss:.4f}")

    # 构建返回结果
    result = {
        "overall_metrics": {
            "ce_f1": avg_region_itc_loss,  # 使用region_itc_loss作为主要指标(用于保存最佳模型)
        },
        "loss": avg_loss,
        "region_itc_loss": avg_region_itc_loss,
    }

    return avg_loss, result

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
    """测试语言模型的生成效果（BERT微调阶段）

    Args:
        config: 配置对象
        data_loader: 数据加载器
        model: MedicalReportGenerator模型
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

            # 模型推理（直接使用**kwargs传递）
            outputs = model(**source)

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
                # 医学报告生成模型的输出格式
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
            
            # 【修复】及时清理batch数据，防止内存累积
            del source, target, outputs, batch, generated_texts, target_texts
            
            # 定期深度清理
            if batch_idx % 50 == 0 and batch_idx > 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

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

    # 将结果转换为DataFrame并保存（不在结果中返回DataFrame，避免长时间占用内存）
    results_df = pd.DataFrame(results_data)
    epoch_str = str(epoch) if epoch is not None else "TEST"
    csv_filename = f"{mode}_results_epoch_{epoch_str}.csv"
    results_csv_path = os.path.join(results_dir, csv_filename)
    results_df.to_csv(results_csv_path, index=False)
    logger.info(f"结果已保存到CSV文件: {csv_filename}")
    # 释放DataFrame内存，避免跨epoch常驻
    del results_df
    
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

    # 保存评估指标，添加epoch信息（同样不在结果中返回DataFrame）
    metrics_df = pd.DataFrame([metrics_data])
    metrics_filename = f"{mode}_metrics_epoch_{epoch_str}.csv"
    metrics_csv_path = os.path.join(results_dir, metrics_filename)
    metrics_df.to_csv(metrics_csv_path, index=False)
    logger.info(f"评估指标已保存到CSV文件: {metrics_filename}")
    del metrics_df
    
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
    # 返回轻量级结果，避免将大型DataFrame对象保存在内存/检查点中
    result = {
        "report_generation_metrics": report_metrics,
        "chexbert_metrics": ce_metrics,
        "loss": avg_loss,
        "results_csv_path": results_csv_path,
        "metrics_csv_path": metrics_csv_path,
    }

    return avg_loss, result