"""
目标检测全面评估指标计算模块
适用于医学影像解剖区域检测任务的论文级评估
"""

import torch
import numpy as np
import logging
from collections import defaultdict
import pandas as pd
from typing import Dict, List, Tuple

# 获取logger
detection_metrics_logger = logging.getLogger("train_logger")


# 解剖区域名称映射 (label_id -> region_name)
REGION_NAMES = {
    1: "hemidiaphragm", 2: "right_atrium", 3: "right_hilar_structures",
    4: "cardiac_silhouette", 5: "abdomen", 6: "trachea",
    7: "right_apical_zone", 8: "right_lung", 9: "right_upper_lung_zone",
    10: "right_costophrenic_angle", 11: "svc", 12: "left_lung",
    13: "right_mid_lung_zone", 14: "cavoatrial_junction", 15: "left_costophrenic_angle",
    16: "left_hilar_structures", 17: "mediastinum", 18: "right_lower_lung_zone",
    19: "left_mid_lung_zone", 20: "spine", 21: "left_upper_lung_zone",
    22: "right_hemidiaphragm", 23: "left_clavicle", 24: "aortic_arch",
    25: "right_clavicle", 26: "left_apical_zone", 27: "left_lower_lung_zone",
    28: "carina", 29: "upper_mediastinum"
}


def calculate_iou(box1, box2):
    """
    计算两个边界框的IoU
    
    Args:
        box1, box2: [x1, y1, x2, y2] 格式的边界框
    
    Returns:
        float: IoU值
    """
    # 计算交集区域
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # 计算交集面积
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height
    
    # 计算并集面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    # 计算IoU
    if union_area == 0:
        return 0.0
    return inter_area / union_area


def calculate_ap(precisions, recalls):
    """
    计算Average Precision (AP)
    使用11点插值法
    
    Args:
        precisions: Precision列表
        recalls: Recall列表
    
    Returns:
        float: AP值
    """
    # 添加边界点
    precisions = np.concatenate(([0], precisions, [0]))
    recalls = np.concatenate(([0], recalls, [1]))
    
    # 计算precision的包络线（确保单调递减）
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # 找到recall变化的点
    indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
    
    # 计算AP（曲线下面积）
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
    
    return ap


def calculate_per_class_metrics(predictions, ground_truths, class_id, iou_threshold=0.5):
    """
    计算单个类别的检测指标
    
    Args:
        predictions: 预测结果列表，每个元素包含 {'boxes', 'scores', 'image_id'}
        ground_truths: 真实标注列表，每个元素包含 {'boxes', 'image_id'}
        class_id: 类别ID
        iou_threshold: IoU阈值
    
    Returns:
        dict: 包含AP, Precision, Recall, F1等指标
    """
    # 收集所有预测和真值
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_image_ids = []
    
    for pred in predictions:
        if len(pred['boxes']) > 0:
            all_pred_boxes.extend(pred['boxes'].numpy())
            all_pred_scores.extend(pred['scores'].numpy())
            all_pred_image_ids.extend([pred['image_id']] * len(pred['boxes']))
    
    # 按置信度降序排序
    if len(all_pred_scores) > 0:
        sorted_indices = np.argsort(all_pred_scores)[::-1]
        all_pred_boxes = [all_pred_boxes[i] for i in sorted_indices]
        all_pred_scores = [all_pred_scores[i] for i in sorted_indices]
        all_pred_image_ids = [all_pred_image_ids[i] for i in sorted_indices]
    
    # 构建真值字典 (image_id -> boxes)
    gt_dict = {}
    total_gt = 0
    for gt in ground_truths:
        if len(gt['boxes']) > 0:
            gt_dict[gt['image_id']] = {
                'boxes': gt['boxes'].numpy(),
                'detected': [False] * len(gt['boxes'])
            }
            total_gt += len(gt['boxes'])
    
    # 计算TP和FP
    tp = np.zeros(len(all_pred_boxes))
    fp = np.zeros(len(all_pred_boxes))
    
    for i, (pred_box, pred_image_id) in enumerate(zip(all_pred_boxes, all_pred_image_ids)):
        if pred_image_id not in gt_dict:
            fp[i] = 1
            continue
        
        gt_boxes = gt_dict[pred_image_id]['boxes']
        gt_detected = gt_dict[pred_image_id]['detected']
        
        # 计算与所有真值框的IoU
        max_iou = 0
        max_idx = -1
        for j, gt_box in enumerate(gt_boxes):
            iou = calculate_iou(pred_box, gt_box)
            if iou > max_iou:
                max_iou = iou
                max_idx = j
        
        # 判断TP或FP
        if max_iou >= iou_threshold and not gt_detected[max_idx]:
            tp[i] = 1
            gt_detected[max_idx] = True
        else:
            fp[i] = 1
    
    # 计算累积TP和FP
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    # 计算Precision和Recall
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
    recalls = tp_cumsum / (total_gt + 1e-10)
    
    # 计算AP
    ap = calculate_ap(precisions, recalls) if len(precisions) > 0 else 0.0
    
    # 计算最终的Precision, Recall, F1
    final_precision = precisions[-1] if len(precisions) > 0 else 0.0
    final_recall = recalls[-1] if len(recalls) > 0 else 0.0
    final_f1 = 2 * final_precision * final_recall / (final_precision + final_recall + 1e-10)
    
    # 计算检测率（至少检测到一个的图像比例）
    num_images_with_gt = len(gt_dict)
    num_images_detected = sum(1 for img_id, info in gt_dict.items() if any(info['detected']))
    detection_rate = num_images_detected / num_images_with_gt if num_images_with_gt > 0 else 0.0
    
    return {
        'ap': ap,
        'precision': final_precision,
        'recall': final_recall,
        'f1': final_f1,
        'detection_rate': detection_rate,
        'total_predictions': len(all_pred_boxes),
        'total_ground_truths': total_gt,
        'true_positives': int(tp_cumsum[-1]) if len(tp_cumsum) > 0 else 0,
        'false_positives': int(fp_cumsum[-1]) if len(fp_cumsum) > 0 else 0,
        'false_negatives': total_gt - (int(tp_cumsum[-1]) if len(tp_cumsum) > 0 else 0)
    }


def calculate_comprehensive_metrics(all_predictions, all_ground_truths, num_classes=29, 
                                    iou_thresholds=[0.3, 0.5, 0.75, 0.9]):
    """
    计算全面的目标检测评估指标
    
    Args:
        all_predictions: 所有预测结果
        all_ground_truths: 所有真实标注
        num_classes: 类别数量
        iou_thresholds: IoU阈值列表
    
    Returns:
        dict: 包含全面评估结果的字典
    """
    results = {
        'overall': {},
        'per_class': {},
        'per_iou_threshold': {}
    }
    
    # 对每个IoU阈值计算指标
    for iou_thresh in iou_thresholds:
        threshold_results = {}
        all_aps = []
        
        # 对每个类别计算指标
        for class_id in range(1, num_classes + 1):
            # 提取该类别的预测和真值
            class_preds = []
            class_gts = []
            
            for pred in all_predictions:
                class_mask = pred['labels'] == class_id
                class_preds.append({
                    'boxes': pred['boxes'][class_mask],
                    'scores': pred['scores'][class_mask],
                    'image_id': pred['image_id']
                })
            
            for gt in all_ground_truths:
                class_mask = gt['labels'] == class_id
                class_gts.append({
                    'boxes': gt['boxes'][class_mask],
                    'image_id': gt['image_id']
                })
            
            # 计算该类别的指标
            class_metrics = calculate_per_class_metrics(
                class_preds, class_gts, class_id, iou_threshold=iou_thresh
            )
            
            region_name = REGION_NAMES.get(class_id, f"class_{class_id}")
            threshold_results[region_name] = class_metrics
            all_aps.append(class_metrics['ap'])
        
        # 计算该IoU阈值下的mAP
        threshold_results['mAP'] = np.mean(all_aps)
        results['per_iou_threshold'][f'IoU@{iou_thresh}'] = threshold_results
    
    # 计算默认IoU@0.5的per_class指标（用于详细报告）
    results['per_class'] = results['per_iou_threshold']['IoU@0.5']
    
    # 计算总体指标
    results['overall'] = {
        'mAP@0.3': results['per_iou_threshold']['IoU@0.3']['mAP'],
        'mAP@0.5': results['per_iou_threshold']['IoU@0.5']['mAP'],
        'mAP@0.75': results['per_iou_threshold']['IoU@0.75']['mAP'],
        'mAP@0.9': results['per_iou_threshold']['IoU@0.9']['mAP'],
        'mAP': np.mean([results['per_iou_threshold'][f'IoU@{t}']['mAP'] for t in iou_thresholds])
    }
    
    # 计算平均Precision, Recall, F1
    all_precisions = []
    all_recalls = []
    all_f1s = []
    all_detection_rates = []
    
    for region_name, metrics in results['per_class'].items():
        if region_name != 'mAP':
            all_precisions.append(metrics['precision'])
            all_recalls.append(metrics['recall'])
            all_f1s.append(metrics['f1'])
            all_detection_rates.append(metrics['detection_rate'])
    
    results['overall']['mean_precision'] = np.mean(all_precisions)
    results['overall']['mean_recall'] = np.mean(all_recalls)
    results['overall']['mean_f1'] = np.mean(all_f1s)
    results['overall']['mean_detection_rate'] = np.mean(all_detection_rates)
    
    return results


def generate_detection_report(results, save_path=None):
    """
    生成目标检测评估报告
    
    Args:
        results: calculate_comprehensive_metrics的返回结果
        save_path: 保存路径（可选）
    
    Returns:
        str: 报告文本
    """
    report_lines = []
    
    # 总体指标
    report_lines.append("=" * 80)
    report_lines.append("目标检测评估报告 - 总体指标")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    overall = results['overall']
    report_lines.append(f"mAP (平均所有IoU阈值):        {overall['mAP']:.4f}")
    report_lines.append(f"mAP@0.3:                      {overall['mAP@0.3']:.4f}")
    report_lines.append(f"mAP@0.5:                      {overall['mAP@0.5']:.4f}")
    report_lines.append(f"mAP@0.75:                     {overall['mAP@0.75']:.4f}")
    report_lines.append(f"mAP@0.9:                      {overall['mAP@0.9']:.4f}")
    report_lines.append("")
    report_lines.append(f"平均Precision:                {overall['mean_precision']:.4f}")
    report_lines.append(f"平均Recall:                   {overall['mean_recall']:.4f}")
    report_lines.append(f"平均F1-Score:                 {overall['mean_f1']:.4f}")
    report_lines.append(f"平均检测率:                   {overall['mean_detection_rate']:.4f}")
    report_lines.append("")
    
    # 每个解剖区域的详细指标
    report_lines.append("=" * 80)
    report_lines.append("每个解剖区域的详细指标 (IoU@0.5)")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # 创建表格
    header = f"{'区域名称':<30} {'AP':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Det Rate':>10} {'GT':>6} {'Pred':>6}"
    report_lines.append(header)
    report_lines.append("-" * len(header))
    
    per_class = results['per_class']
    for region_name in sorted(per_class.keys()):
        if region_name == 'mAP':
            continue
        metrics = per_class[region_name]
        line = (f"{region_name:<30} "
                f"{metrics['ap']:>8.4f} "
                f"{metrics['precision']:>8.4f} "
                f"{metrics['recall']:>8.4f} "
                f"{metrics['f1']:>8.4f} "
                f"{metrics['detection_rate']:>10.4f} "
                f"{metrics['total_ground_truths']:>6d} "
                f"{metrics['total_predictions']:>6d}")
        report_lines.append(line)
    
    report_lines.append("")
    
    # 不同IoU阈值下的mAP对比
    report_lines.append("=" * 80)
    report_lines.append("不同IoU阈值下的mAP对比")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    for iou_key in sorted(results['per_iou_threshold'].keys()):
        mAP = results['per_iou_threshold'][iou_key]['mAP']
        report_lines.append(f"{iou_key:<15} mAP: {mAP:.4f}")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    report_text = "\n".join(report_lines)
    
    # 保存报告
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
    
    return report_text


def save_results_to_csv(results, save_path):
    """
    将评估结果保存为CSV格式（便于论文制表）
    
    Args:
        results: calculate_comprehensive_metrics的返回结果
        save_path: CSV文件保存路径
    """
    # 准备数据
    rows = []
    per_class = results['per_class']
    
    for region_name in sorted(per_class.keys()):
        if region_name == 'mAP':
            continue
        metrics = per_class[region_name]
        row = {
            'Region': region_name,
            'AP@0.5': metrics['ap'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1'],
            'Detection_Rate': metrics['detection_rate'],
            'Total_GT': metrics['total_ground_truths'],
            'Total_Pred': metrics['total_predictions'],
            'TP': metrics['true_positives'],
            'FP': metrics['false_positives'],
            'FN': metrics['false_negatives']
        }
        rows.append(row)
    
    # 添加总体指标行
    overall = results['overall']
    summary_row = {
        'Region': 'OVERALL',
        'AP@0.5': overall['mAP@0.5'],
        'Precision': overall['mean_precision'],
        'Recall': overall['mean_recall'],
        'F1-Score': overall['mean_f1'],
        'Detection_Rate': overall['mean_detection_rate'],
        'Total_GT': '-',
        'Total_Pred': '-',
        'TP': '-',
        'FP': '-',
        'FN': '-'
    }
    rows.append(summary_row)
    
    # 保存为CSV
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    detection_metrics_logger.info(f"✅ 评估结果已保存至: {save_path}")

