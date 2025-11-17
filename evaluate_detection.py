import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from torchvision.ops import box_iou
import matplotlib.pyplot as plt
import os
import json
import logging
from sklearn.metrics import precision_recall_curve, average_precision_score

from models.fast_rcnn_classifier import DetectionOnlyFastRCNN


# 解剖结构名称映射（索引从1开始）
ANATOMY_ORDER = [
    'left hemidiaphragm',      # 1
    'right atrium',            # 2
    'right hilar structures',  # 3
    'cardiac silhouette',      # 4
    'abdomen',                 # 5
    'trachea',                 # 6
    'right apical zone',       # 7
    'right lung',              # 8
    'right upper lung zone',   # 9
    'right costophrenic angle',# 10
    'svc',                     # 11
    'left lung',               # 12
    'right mid lung zone',     # 13
    'cavoatrial junction',     # 14
    'left costophrenic angle', # 15
    'left hilar structures',   # 16
    'mediastinum',             # 17
    'right lower lung zone',   # 18
    'left mid lung zone',      # 19
    'spine',                   # 20
    'left upper lung zone',    # 21
    'right hemidiaphragm',     # 22
    'left clavicle',           # 23
    'aortic arch',             # 24
    'right clavicle',          # 25
    'left apical zone',        # 26
    'left lower lung zone',    # 27
    'carina',                  # 28
    'upper mediastinum',       # 29
]


def evaluate_detection_model(
    model,
    test_loader,
    num_classes=29,
    iou_threshold=0.5,
    confidence_threshold=0.5,
    output_dir="./evaluation_results",
):
    """
    Evaluate the performance of object detection model and calculate metrics for each region

    Args:
        model: Detection model
        test_loader: Test data loader
        num_classes: Number of classes (excluding background)
        iou_threshold: IoU threshold for determining correct detection
        confidence_threshold: Detection confidence threshold
        output_dir: Output directory for saving results

    Returns:
        dict: Dictionary containing various evaluation metrics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    model.eval()  # Set to evaluation mode

    # Store all predictions and ground truths
    all_predictions = []
    all_ground_truths = []

    # Store predictions and ground truths by class
    class_predictions = defaultdict(list)
    class_ground_truths = defaultdict(list)

    logger = logging.getLogger("train_logger")
    logger.info("Starting object detection model evaluation...")
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader, desc="Evaluation Progress")):
            # MIMIC collate_fn returns dict format
            images = data["image"]  # shape: (batch_size, 3, 224, 224)
            targets = data["bbox_targets"]  # list of dicts

            # Move all content to the same device as model
            device = next(model.parameters()).device
            images = [img.to(device) for img in images]

            # Get model predictions
            if hasattr(model, "detector"):
                # EnhancedFastRCNN model
                detections = model.detector(images)
            else:
                # DetectionOnlyFastRCNN model or native FasterRCNN
                detections = model(images)

            # Process prediction results for each image
            for i, (detection, target) in enumerate(zip(detections, targets)):
                # Get predicted bounding boxes
                pred_boxes = detection["boxes"].cpu()
                pred_scores = detection["scores"].cpu()
                pred_labels = detection["labels"].cpu()

                # Apply confidence threshold
                keep = pred_scores > confidence_threshold
                pred_boxes = pred_boxes[keep]
                pred_scores = pred_scores[keep]
                pred_labels = pred_labels[keep]

                # Get ground truth bounding boxes
                gt_boxes = target["boxes"].cpu()
                gt_labels = target["labels"].cpu()

                # Store predictions and ground truths for current image
                img_pred = {
                    "boxes": pred_boxes,
                    "scores": pred_scores,
                    "labels": pred_labels,
                    "image_id": batch_idx * len(images) + i,
                }
                img_gt = {
                    "boxes": gt_boxes,
                    "labels": gt_labels,
                    "image_id": batch_idx * len(images) + i,
                }

                all_predictions.append(img_pred)
                all_ground_truths.append(img_gt)

                # Store predictions and ground truths by class
                for label in range(1, num_classes + 1):
                    pred_idx = (pred_labels == label).nonzero(as_tuple=True)[0]
                    gt_idx = (gt_labels == label).nonzero(as_tuple=True)[0]

                    class_predictions[label].append(
                        {
                            "boxes": (
                                pred_boxes[pred_idx]
                                if len(pred_idx) > 0
                                else torch.zeros((0, 4))
                            ),
                            "scores": (
                                pred_scores[pred_idx]
                                if len(pred_idx) > 0
                                else torch.zeros(0)
                            ),
                            "image_id": batch_idx * len(images) + i,
                        }
                    )

                    class_ground_truths[label].append(
                        {
                            "boxes": (
                                gt_boxes[gt_idx]
                                if len(gt_idx) > 0
                                else torch.zeros((0, 4))
                            ),
                            "image_id": batch_idx * len(images) + i,
                        }
                    )

    # Calculate overall evaluation metrics
    logger.info("Computing evaluation metrics...")
    overall_metrics = compute_map(all_predictions, all_ground_truths, iou_threshold)

    # Calculate evaluation metrics for each class
    class_metrics = {}
    for class_id in range(1, num_classes + 1):
        class_metrics[class_id] = compute_class_metrics(
            class_predictions[class_id], class_ground_truths[class_id], iou_threshold
        )

    # Aggregate results
    results = {"overall": overall_metrics, "per_class": class_metrics}

    # Calculate average metrics
    average_precision = np.mean([metrics["AP"] for metrics in class_metrics.values()])
    average_recall = np.mean([metrics["recall"] for metrics in class_metrics.values()])
    average_f1 = np.mean([metrics["f1_score"] for metrics in class_metrics.values()])
    average_mean_iou = np.mean([metrics["mean_iou"] for metrics in class_metrics.values()])
    average_missing_rate = np.mean([metrics["missing_rate"] for metrics in class_metrics.values()])

    results["mAP"] = average_precision
    results["mRecall"] = average_recall
    results["mF1"] = average_f1
    results["mMeanIoU"] = average_mean_iou
    results["mMissingRate"] = average_missing_rate

    # Save results to JSON file (removing non-serializable fields)
    # Helper function: Convert all values to Python native types
    def convert_to_native_types(obj):
        """Recursively convert numpy/torch types to Python native types"""
        if isinstance(obj, dict):
            return {k: convert_to_native_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # PyTorch tensors and numpy scalars
            return obj.item()
        else:
            return obj

    # Create a clean copy, removing precision_curve and recall_curve
    results_for_json = {
        "overall": convert_to_native_types(results["overall"]),
        "per_class": {},
        "mAP": convert_to_native_types(results["mAP"]),
        "mRecall": convert_to_native_types(results["mRecall"]),
        "mF1": convert_to_native_types(results["mF1"]),
        "mMeanIoU": convert_to_native_types(results["mMeanIoU"]),
        "mMissingRate": convert_to_native_types(results["mMissingRate"]),
    }

    # Copy metrics for each class, excluding curve data
    # 按mean_iou排序并用label名称作为key
    sorted_class_items = sorted(
        results["per_class"].items(),
        key=lambda x: x[1]['mean_iou'],
        reverse=True
    )
    for class_id, metrics in sorted_class_items:
        label_name = ANATOMY_ORDER[class_id - 1] if 1 <= class_id <= len(ANATOMY_ORDER) else f"Region_{class_id}"
        results_for_json["per_class"][label_name] = {
            "class_id": class_id,
            **{k: convert_to_native_types(v) for k, v in metrics.items()
               if k not in ["precision_curve", "recall_curve"]}
        }

    with open(os.path.join(output_dir, "detection_metrics.json"), "w") as f:
        json.dump(results_for_json, f, indent=4)

    # Visualize class AP
    visualize_class_metrics(class_metrics, output_dir)

    # Print overall metrics
    logger.info(f"\n========== Overall Evaluation Metrics (IoU Threshold={iou_threshold}) ==========")
    logger.info(f"mAP@{iou_threshold}: {average_precision:.4f}")
    logger.info(f"Mean Recall@{iou_threshold}: {average_recall:.4f}")
    logger.info(f"Mean F1@{iou_threshold}: {average_f1:.4f}")
    logger.info(f"Mean IoU (Average Localization Accuracy of TP): {average_mean_iou:.4f}")
    logger.info(f"Mean Missing Rate (Average Undetection Rate): {average_missing_rate:.4f}")

    # Print detailed metrics for each class (按mean_iou排序)
    logger.info("\n========== Detailed Metrics by Anatomical Region (按Mean IoU排序) ==========")
    # 按mean_iou排序
    sorted_class_metrics = sorted(
        class_metrics.items(),
        key=lambda x: x[1]['mean_iou'],
        reverse=True
    )
    
    # 将详细指标输出到txt文件
    txt_output_path = os.path.join(output_dir, "detailed_metrics.txt")
    with open(txt_output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Detailed Metrics by Anatomical Region (按Mean IoU排序)\n")
        f.write("=" * 80 + "\n\n")
        
        for class_id, metrics in sorted_class_metrics:
            # 获取对应的label名称（索引从1开始）
            label_name = ANATOMY_ORDER[class_id - 1] if 1 <= class_id <= len(ANATOMY_ORDER) else f"Region_{class_id}"
            line = (
                f"{label_name} (ID={class_id}): "
                f"AP@{iou_threshold}={metrics['AP']:.4f}, "
                f"P@{iou_threshold}={metrics['precision']:.4f}, "
                f"R@{iou_threshold}={metrics['recall']:.4f}, "
                f"F1@{iou_threshold}={metrics['f1_score']:.4f}, "
                f"Mean IoU={metrics['mean_iou']:.4f}, "
                f"Missing Rate={metrics['missing_rate']:.4f}\n"
            )
            f.write(line)
            logger.info(line.strip())
    
    logger.info(f"\n详细指标已保存到: {txt_output_path}")

    return results


def compute_map(predictions, ground_truths, iou_threshold=0.5):
    """
    Calculate overall mAP
    """
    # Merge predictions and ground truths from all images
    all_gt_boxes = []
    all_gt_labels = []
    all_gt_image_ids = []

    all_pred_boxes = []
    all_pred_scores = []
    all_pred_labels = []
    all_pred_image_ids = []

    for pred, gt in zip(predictions, ground_truths):
        all_gt_boxes.append(gt["boxes"])
        all_gt_labels.append(gt["labels"])
        all_gt_image_ids.extend([gt["image_id"]] * len(gt["boxes"]))

        all_pred_boxes.append(pred["boxes"])
        all_pred_scores.append(pred["scores"])
        all_pred_labels.append(pred["labels"])
        all_pred_image_ids.extend([pred["image_id"]] * len(pred["boxes"]))

    if len(all_gt_boxes) == 0 or len(all_pred_boxes) == 0:
        return {"mAP": 0.0}

    all_gt_boxes = torch.cat(all_gt_boxes) if all_gt_boxes else torch.zeros((0, 4))
    all_gt_labels = (
        torch.cat(all_gt_labels) if all_gt_labels else torch.zeros(0, dtype=torch.int64)
    )
    all_gt_image_ids = (
        torch.tensor(all_gt_image_ids)
        if all_gt_image_ids
        else torch.zeros(0, dtype=torch.int64)
    )

    all_pred_boxes = (
        torch.cat(all_pred_boxes) if all_pred_boxes else torch.zeros((0, 4))
    )
    all_pred_scores = torch.cat(all_pred_scores) if all_pred_scores else torch.zeros(0)
    all_pred_labels = (
        torch.cat(all_pred_labels)
        if all_pred_labels
        else torch.zeros(0, dtype=torch.int64)
    )
    all_pred_image_ids = (
        torch.tensor(all_pred_image_ids)
        if all_pred_image_ids
        else torch.zeros(0, dtype=torch.int64)
    )

    # 按置信度排序所有预测
    sorted_indices = torch.argsort(all_pred_scores, descending=True)
    all_pred_boxes = all_pred_boxes[sorted_indices]
    all_pred_scores = all_pred_scores[sorted_indices]
    all_pred_labels = all_pred_labels[sorted_indices]
    all_pred_image_ids = all_pred_image_ids[sorted_indices]

    # 计算精确度-召回率曲线
    num_preds = len(all_pred_boxes)
    num_gts = len(all_gt_boxes)

    tp = torch.zeros(num_preds)
    fp = torch.zeros(num_preds)

    # 跟踪已匹配的真值框（使用(image_id, gt_idx)作为唯一标识）
    matched_gt_boxes = set()

    # 对每个预测，判断是否为真阳性
    for i in range(num_preds):
        pred_box = all_pred_boxes[i]
        pred_label = all_pred_labels[i]
        pred_image_id = all_pred_image_ids[i]

        # 找到同一图像中相同类别的所有真值边界框
        mask = (all_gt_labels == pred_label) & (all_gt_image_ids == pred_image_id)
        gt_boxes_same_class = all_gt_boxes[mask]
        gt_indices_same_class = torch.where(mask)[0]

        if len(gt_boxes_same_class) == 0:
            fp[i] = 1  # 假阳性
            continue

        # 计算与所有真值边界框的IoU
        ious = box_iou(pred_box.unsqueeze(0), gt_boxes_same_class)[0]

        # 如果最大IoU大于阈值，则为真阳性
        if ious.max() >= iou_threshold:
            # 找到最大IoU对应的真值边界框
            max_iou_idx = ious.argmax()
            gt_idx = gt_indices_same_class[max_iou_idx].item()

            # 检查该真值框是否已被匹配
            match_key = (pred_image_id.item(), gt_idx)
            if match_key not in matched_gt_boxes:
                tp[i] = 1
                matched_gt_boxes.add(match_key)
            else:
                fp[i] = 1  # 该GT框已被匹配，这是重复检测
        else:
            fp[i] = 1  # 假阳性

    # 计算累积TP和FP
    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)

    # 计算精确度和召回率
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
    recall = tp_cumsum / (num_gts + 1e-10)

    # 计算AP（精确度-召回率曲线下面积）
    ap = 0.0
    for t in torch.arange(0, 1.1, 0.1):
        if torch.sum(recall >= t) == 0:
            p = 0
        else:
            p = torch.max(precision[recall >= t])
        ap = ap + p / 11.0

    return {
        "mAP": ap.item(),
        "precision": precision[-1].item() if len(precision) > 0 else 0.0,
        "recall": recall[-1].item() if len(recall) > 0 else 0.0,
        "f1_score": (
            2 * precision[-1] * recall[-1] / (precision[-1] + recall[-1] + 1e-10)
            if len(precision) > 0 and len(recall) > 0
            else 0.0
        ),
    }


def compute_class_metrics(class_predictions, class_ground_truths, iou_threshold=0.5):
    """
    计算单个类别的评价指标

    新增指标：
    - mean_iou: 所有TP预测的平均IoU（定位精度）
    - missing_rate: 完全漏检率 = 有GT但无任何检测的图像数 / 有GT的图像总数
    """
    # 提取所有预测和真值
    all_gt_boxes = []
    all_gt_image_ids = []

    all_pred_boxes = []
    all_pred_scores = []
    all_pred_image_ids = []

    # 统计图像级别的GT和预测情况（用于计算missing rate）
    images_with_gt = set()  # 有该区域GT的图像ID
    images_with_detection = set()  # 有该区域检测的图像ID

    for pred, gt in zip(class_predictions, class_ground_truths):
        if len(gt["boxes"]) > 0:
            all_gt_boxes.append(gt["boxes"])
            all_gt_image_ids.extend([gt["image_id"]] * len(gt["boxes"]))
            images_with_gt.add(gt["image_id"])

        if len(pred["boxes"]) > 0:
            all_pred_boxes.append(pred["boxes"])
            all_pred_scores.append(pred["scores"])
            all_pred_image_ids.extend([pred["image_id"]] * len(pred["boxes"]))
            images_with_detection.add(pred["image_id"])

    # 如果没有预测或真值，返回零指标
    if not all_gt_boxes or not all_pred_boxes:
        # 计算missing rate（没有真值时为0，没有预测时为1）
        missing_rate = 1.0 if all_gt_boxes and not all_pred_boxes else 0.0
        return {
            "AP": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "mean_iou": 0.0,
            "missing_rate": missing_rate,
            "TP": 0,
            "FP": 0,
            "FN": sum(len(gt["boxes"]) for gt in class_ground_truths),
        }

    all_gt_boxes = torch.cat(all_gt_boxes)
    all_gt_image_ids = torch.tensor(all_gt_image_ids)

    all_pred_boxes = torch.cat(all_pred_boxes)
    all_pred_scores = torch.cat(all_pred_scores)
    all_pred_image_ids = torch.tensor(all_pred_image_ids)

    # 按置信度排序预测
    sorted_indices = torch.argsort(all_pred_scores, descending=True)
    all_pred_boxes = all_pred_boxes[sorted_indices]
    all_pred_scores = all_pred_scores[sorted_indices]
    all_pred_image_ids = all_pred_image_ids[sorted_indices]

    # 计算TP和FP
    num_preds = len(all_pred_boxes)
    num_gts = len(all_gt_boxes)

    tp = torch.zeros(num_preds)
    fp = torch.zeros(num_preds)
    tp_ious = []  # 存储所有TP的IoU值

    # 跟踪已匹配的真值框
    matched_gt_boxes = set()

    for i in range(num_preds):
        pred_box = all_pred_boxes[i]
        pred_image_id = all_pred_image_ids[i]

        # 找到同一图像中的真值框
        same_image_mask = all_gt_image_ids == pred_image_id
        gt_boxes_same_image = all_gt_boxes[same_image_mask]
        gt_indices_same_image = torch.where(same_image_mask)[0]

        if len(gt_boxes_same_image) == 0:
            fp[i] = 1  # 假阳性
            continue

        # 计算IoU
        ious = box_iou(pred_box.unsqueeze(0), gt_boxes_same_image)[0]

        if ious.max() >= iou_threshold:
            # 找到最大IoU对应的真值框
            max_iou_idx = ious.argmax()
            max_iou_value = ious[max_iou_idx].item()
            gt_idx = gt_indices_same_image[max_iou_idx].item()

            # 检查该真值框是否已被匹配
            match_key = (pred_image_id.item(), gt_idx)
            if match_key not in matched_gt_boxes:
                tp[i] = 1
                tp_ious.append(max_iou_value)  # 记录TP的IoU
                matched_gt_boxes.add(match_key)
            else:
                fp[i] = 1
        else:
            fp[i] = 1

    # 计算累积TP和FP
    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)

    # 计算精确度和召回率
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
    recall = tp_cumsum / (num_gts + 1e-10)

    # 计算AP
    ap = compute_average_precision(precision, recall)

    # 计算最终指标
    final_precision = precision[-1].item() if len(precision) > 0 else 0.0
    final_recall = recall[-1].item() if len(recall) > 0 else 0.0
    f1_score = (
        2 * final_precision * final_recall / (final_precision + final_recall + 1e-10)
    )

    # 计算TP、FP、FN
    TP = tp.sum().item()
    FP = fp.sum().item()
    FN = num_gts - TP

    # 计算平均IoU（仅针对TP）
    mean_iou = np.mean(tp_ious) if len(tp_ious) > 0 else 0.0

    # 计算Missing Rate（图像级别：有GT但完全没检测到的图像比例）
    images_completely_missed = images_with_gt - images_with_detection
    missing_rate = (
        len(images_completely_missed) / len(images_with_gt)
        if len(images_with_gt) > 0
        else 0.0
    )

    return {
        "AP": ap,
        "precision": final_precision,
        "recall": final_recall,
        "f1_score": f1_score,
        "mean_iou": mean_iou,
        "missing_rate": missing_rate,
        "TP": int(TP),
        "FP": int(FP),
        "FN": int(FN),
        "num_images_with_gt": len(images_with_gt),
        "num_images_missed": len(images_completely_missed),
        "precision_curve": precision.numpy(),
        "recall_curve": recall.numpy(),
    }


def compute_average_precision(precision, recall):
    """
    使用插值方法计算AP
    """
    # 转换为numpy数组
    if isinstance(precision, torch.Tensor):
        precision = precision.numpy()
    if isinstance(recall, torch.Tensor):
        recall = recall.numpy()

    # 按11点法计算AP
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap = ap + p / 11.0

    return ap


def visualize_class_metrics(class_metrics, output_dir):
    """
    Visualize metrics for each class
    """
    # Prepare data
    class_ids = list(class_metrics.keys())
    aps = [metrics["AP"] for metrics in class_metrics.values()]
    precisions = [metrics["precision"] for metrics in class_metrics.values()]
    recalls = [metrics["recall"] for metrics in class_metrics.values()]
    f1_scores = [metrics["f1_score"] for metrics in class_metrics.values()]
    mean_ious = [metrics["mean_iou"] for metrics in class_metrics.values()]
    missing_rates = [metrics["missing_rate"] for metrics in class_metrics.values()]

    # Sort by AP value
    sorted_indices = np.argsort(aps)[::-1]
    sorted_class_ids = [class_ids[i] for i in sorted_indices]
    sorted_aps = [aps[i] for i in sorted_indices]
    sorted_precisions = [precisions[i] for i in sorted_indices]
    sorted_recalls = [recalls[i] for i in sorted_indices]
    sorted_f1_scores = [f1_scores[i] for i in sorted_indices]
    sorted_mean_ious = [mean_ious[i] for i in sorted_indices]
    sorted_missing_rates = [missing_rates[i] for i in sorted_indices]

    # Plot AP bar chart
    plt.figure(figsize=(14, 8))
    plt.bar(range(len(sorted_class_ids)), sorted_aps, color="skyblue")
    plt.xlabel("Anatomical Region ID")
    plt.ylabel("AP")
    plt.title("Average Precision by Anatomical Region")
    plt.xticks(range(len(sorted_class_ids)), sorted_class_ids, rotation=90)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ap_by_class.png"))

    # Plot Precision-Recall-F1 scatter plot
    plt.figure(figsize=(10, 8))
    for i, class_id in enumerate(sorted_class_ids):
        plt.scatter(
            sorted_recalls[i],
            sorted_precisions[i],
            s=sorted_f1_scores[i] * 100,
            alpha=0.6,
            label=f"Region {class_id}",
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Relationship by Region (Bubble Size = F1 Score)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "precision_recall_f1.png"))

    # Plot Precision-Recall curves for each region
    plt.figure(figsize=(12, 10))

    # Select top 10 classes by AP to plot curves
    top_n = min(10, len(sorted_class_ids))
    for i in range(top_n):
        class_id = sorted_class_ids[i]
        if (
            "precision_curve" in class_metrics[class_id]
            and "recall_curve" in class_metrics[class_id]
        ):
            plt.plot(
                class_metrics[class_id]["recall_curve"],
                class_metrics[class_id]["precision_curve"],
                label=f'Region {class_id} (AP: {class_metrics[class_id]["AP"]:.4f})',
            )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curves for Top {top_n} Regions")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_pr_curves.png"))

    # Plot Mean IoU bar chart
    plt.figure(figsize=(14, 8))
    plt.bar(range(len(sorted_class_ids)), sorted_mean_ious, color="lightgreen")
    plt.xlabel("Anatomical Region ID")
    plt.ylabel("Mean IoU")
    plt.title("Mean IoU by Anatomical Region (Localization Accuracy)")
    plt.xticks(range(len(sorted_class_ids)), sorted_class_ids, rotation=90)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.ylim(0, 1.0)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='IoU=0.5 Threshold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mean_iou_by_class.png"))

    # Plot Missing Rate bar chart
    plt.figure(figsize=(14, 8))
    plt.bar(range(len(sorted_class_ids)), sorted_missing_rates, color="coral")
    plt.xlabel("Anatomical Region ID")
    plt.ylabel("Missing Rate")
    plt.title("Missing Rate by Anatomical Region (Proportion of Undetected Images)")
    plt.xticks(range(len(sorted_class_ids)), sorted_class_ids, rotation=90)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "missing_rate_by_class.png"))

    plt.close("all")


def test_detection_model(model, test_loader, device, output_dir="./test_results"):
    """
    测试目标检测模型并保存结果

    参数:
        model: 检测模型
        test_loader: 测试数据加载器
        device: 计算设备
        output_dir: 输出结果保存目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 设置模型为评估模式
    model.eval()

    # 获取类别名称映射
    class_names = {i + 1: f"Region_{i+1}" for i in range(29)}  # 可替换为实际的区域名称

    logger = logging.getLogger("train_logger")
    logger.info("开始测试目标检测模型...")

    # 评估模型
    evaluation_results = evaluate_detection_model(
        model,
        test_loader,
        num_classes=29,
        iou_threshold=0.5,
        confidence_threshold=0.5,
        output_dir=output_dir,
    )

    # 打印总体性能
    logger.info(f"\n========== 总体性能汇总 ==========")
    logger.info(f"mAP@0.5: {evaluation_results['mAP']:.4f}")
    logger.info(f"Mean Recall@0.5: {evaluation_results['mRecall']:.4f}")
    logger.info(f"Mean F1@0.5: {evaluation_results['mF1']:.4f}")
    logger.info(f"Mean IoU (定位精度): {evaluation_results['mMeanIoU']:.4f}")
    logger.info(f"Mean Missing Rate (漏检率): {evaluation_results['mMissingRate']:.4f}")

    # 打印表现最好和最差的区域（按mean_iou排序）
    per_class_metrics = evaluation_results["per_class"]
    class_mean_ious = [
        (class_id, metrics["mean_iou"]) for class_id, metrics in per_class_metrics.items()
    ]
    class_mean_ious.sort(key=lambda x: x[1], reverse=True)

    logger.info("\n表现最好的5个区域 (按Mean IoU排序):")
    for class_id, mean_iou in class_mean_ious[:5]:
        metrics = per_class_metrics[class_id]
        label_name = ANATOMY_ORDER[class_id - 1] if 1 <= class_id <= len(ANATOMY_ORDER) else f"Region_{class_id}"
        logger.info(
            f"{label_name} (ID={class_id}): "
            f"AP@0.5={metrics['AP']:.4f}, "
            f"P@0.5={metrics['precision']:.4f}, "
            f"R@0.5={metrics['recall']:.4f}, "
            f"F1@0.5={metrics['f1_score']:.4f}, "
            f"Mean IoU={mean_iou:.4f}, "
            f"Missing Rate={metrics['missing_rate']:.4f}"
        )

    logger.info("\n表现最差的5个区域 (按Mean IoU排序):")
    for class_id, mean_iou in class_mean_ious[-5:]:
        metrics = per_class_metrics[class_id]
        label_name = ANATOMY_ORDER[class_id - 1] if 1 <= class_id <= len(ANATOMY_ORDER) else f"Region_{class_id}"
        logger.info(
            f"{label_name} (ID={class_id}): "
            f"AP@0.5={metrics['AP']:.4f}, "
            f"P@0.5={metrics['precision']:.4f}, "
            f"R@0.5={metrics['recall']:.4f}, "
            f"F1@0.5={metrics['f1_score']:.4f}, "
            f"Mean IoU={mean_iou:.4f}, "
            f"Missing Rate={metrics['missing_rate']:.4f}"
        )

    return evaluation_results


def create_test_data_loader():
    """
    创建目标检测评估用的测试数据加载器

    Returns:
        test_loader: 测试数据加载器
    """
    from datasets import MIMIC, mimic_collate_fn
    from configs import config
    from transformers import BertTokenizer
    import torch.utils.data as data

    # 创建tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})

    # 加载共享数据
    MIMIC.load_shared_data(
        directory=config.DATA_DIR,
        ann_dir=config.ANN_DIR,
        mode=config.MODE,
        binary_mode=True,
        split_csv_path=config.SPLIT_CSV_PATH,
        generation_target=config.GENERATION_TARGET
    )

    # 创建测试数据集
    input_size = (config.IMAGE_SIZE, config.IMAGE_SIZE)
    test_data = MIMIC(
        directory=config.DATA_DIR,
        ann_dir=config.ANN_DIR,
        images_dir=config.IMAGES_DIR,
        input_size=input_size,
        random_transform=False,  # 测试时不做随机变换
        tokenizer=tokenizer,
        mode="test",
        generation_target=config.GENERATION_TARGET
    )

    # 创建数据加载器
    test_loader = data.DataLoader(
        test_data,
        batch_size=config.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=mimic_collate_fn
    )

    print(f"✅ 测试数据集加载完成: {len(test_data)} 个样本")
    print(f"   Batch size: {config.VAL_BATCH_SIZE}")
    print(f"   Total batches: {len(test_loader)}")

    return test_loader


# 使用示例
def main():
    """
    评估目标检测模型的主函数

    使用方法:
        python evaluate_detection.py
    """
    # 1. 加载模型
    print("=" * 80)
    print("目标检测模型评估")
    print("=" * 80)

    model = DetectionOnlyFastRCNN(num_regions=29)
    checkpoint_path = "/home/chenlb/xray_moe/results/detection/best_mAP05_0.8840_mAP_0.6994.pth"

    # 先确定目标设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n加载模型权重: {checkpoint_path}")
    print(f"目标设备: {device}")

    # PyTorch 2.6+ 默认 weights_only=True，需要显式设置为 False 以加载包含 numpy 对象的检查点
    # 使用 map_location 将模型映射到当前设备（避免跨设备加载错误）
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 处理 state_dict：去掉 DDP/DataParallel 的 'module.' 前缀
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if any(key.startswith('module.') for key in state_dict.keys()):
        print("检测到 DDP/DataParallel 模型，去除 'module.' 前缀...")
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(device)

    print(f"✅ 模型已成功加载到设备: {device}")

    # 2. 创建测试数据加载器
    print("\n创建测试数据加载器...")
    test_loader = create_test_data_loader()

    # 3. 执行测试
    print("\n开始评估...")
    output_dir = './evaluation_results'
    results = test_detection_model(model, test_loader, device, output_dir=output_dir)

    print("\n" + "=" * 80)
    print(f"评估完成! 结果已保存至: {output_dir}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    main()
