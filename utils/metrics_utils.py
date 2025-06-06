"""指标计算工具"""
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def count_parameters(model):
    """
    统计模型可训练参数数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        可训练参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def visual_parameters(modules, parameters):
    """
    可视化显示模型各模块的参数分布
    
    Args:
        modules: 模块名称列表
        parameters: 参数数量列表
    """
    print("\n" + "=" * 70)
    print("模型参数统计")
    print("=" * 70)
    
    total_params = sum(parameters)
    
    for module_name, param_count in zip(modules, parameters):
        percentage = (param_count / total_params * 100) if total_params > 0 else 0
        bar = "█" * int(percentage / 2)  # 每个█代表2%
        print(f"{module_name:25s} {param_count:12,d} ({percentage:5.1f}%) {bar}")
    
    print("=" * 70)
    print(f"{'总参数':25s} {total_params:12,d} (100.0%)")
    print("=" * 70)


def calculate_detection_metrics(predictions, ground_truths, iou_threshold=0.5):
    """
    计算目标检测的评估指标
    
    Args:
        predictions: 预测结果列表
        ground_truths: 真实标注列表
        iou_threshold: IoU阈值
        
    Returns:
        包含各种指标的字典
    """
    from torchvision.ops import box_iou
    
    total_gt = 0
    total_pred = 0
    true_positives = 0
    
    for pred, gt in zip(predictions, ground_truths):
        pred_boxes = pred['boxes']
        gt_boxes = gt['boxes']
        
        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)
        
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            continue
        
        # 计算IoU矩阵
        iou_matrix = box_iou(pred_boxes, gt_boxes)
        
        # 对于每个预测框，找到最佳匹配的GT框
        max_ious, _ = iou_matrix.max(dim=1)
        true_positives += (max_ious >= iou_threshold).sum().item()
    
    # 计算指标
    precision = true_positives / total_pred if total_pred > 0 else 0
    recall = true_positives / total_gt if total_gt > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'total_predictions': total_pred,
        'total_ground_truths': total_gt
    }


def calculate_class_metrics(class_predictions, class_ground_truths, iou_threshold=0.5):
    """
    计算各类别的检测指标
    
    Args:
        class_predictions: 按类别分组的预测
        class_ground_truths: 按类别分组的真实标注
        iou_threshold: IoU阈值
        
    Returns:
        按类别的指标字典
    """
    from collections import defaultdict
    
    class_metrics = defaultdict(lambda: {
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'count': 0
    })
    
    # 对每个类别计算指标
    all_classes = set(class_predictions.keys()) | set(class_ground_truths.keys())
    
    for class_id in all_classes:
        preds = class_predictions.get(class_id, [])
        gts = class_ground_truths.get(class_id, [])
        
        metrics = calculate_detection_metrics(preds, gts, iou_threshold)
        
        class_metrics[class_id] = {
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'count': len(gts)
        }
    
    return dict(class_metrics)


def calculate_classification_metrics(predictions, targets, num_classes=14):
    """
    计算多标签分类指标
    
    Args:
        predictions: 预测概率或logits [N, num_classes]
        targets: 真实标签 [N, num_classes]
        num_classes: 类别数量
        
    Returns:
        包含AUROC和AUPRC的字典
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # 对预测应用sigmoid（如果是logits）
    predictions = 1 / (1 + np.exp(-predictions))
    
    auroc_scores = []
    auprc_scores = []
    
    for i in range(num_classes):
        try:
            # 跳过全0或全1的类别
            if targets[:, i].sum() == 0 or targets[:, i].sum() == len(targets):
                continue
            
            auroc = roc_auc_score(targets[:, i], predictions[:, i])
            auprc = average_precision_score(targets[:, i], predictions[:, i])
            
            auroc_scores.append(auroc)
            auprc_scores.append(auprc)
        except Exception as e:
            print(f"计算类别 {i} 的指标时出错: {e}")
            continue
    
    return {
        'mean_auroc': np.mean(auroc_scores) if auroc_scores else 0.0,
        'mean_auprc': np.mean(auprc_scores) if auprc_scores else 0.0,
        'auroc_scores': auroc_scores,
        'auprc_scores': auprc_scores
    }

