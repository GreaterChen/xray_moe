"""评估相关工具函数"""
import os
import re
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from utils.data_utils import prepare_batch_data, args_to_kwargs
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

            # 模型推理（直接使用**kwargs传递）
            output = model(**source)
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

            # 模型推理（直接使用**kwargs传递）
            outputs = model(**source)
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