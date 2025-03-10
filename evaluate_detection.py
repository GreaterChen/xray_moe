import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from torchvision.ops import box_iou
import matplotlib.pyplot as plt
import os
import json
from sklearn.metrics import precision_recall_curve, average_precision_score

from models.fast_rcnn_classifier import DetectionOnlyFastRCNN


def evaluate_detection_model(model, test_loader, num_classes=29, iou_threshold=0.5, 
                             confidence_threshold=0.5, output_dir='./evaluation_results'):
    """
    评估目标检测模型的性能，计算各个区域的精度指标
    
    参数:
        model: 检测模型
        test_loader: 测试数据加载器
        num_classes: 类别数量（不包括背景）
        iou_threshold: 判断检测为正确的IoU阈值
        confidence_threshold: 检测置信度阈值
        output_dir: 输出结果保存目录
    
    返回:
        dict: 包含各种评价指标的字典
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()  # 设置为评估模式
    
    # 用于存储所有预测和真值
    all_predictions = []
    all_ground_truths = []
    
    # 按类别存储预测和真值
    class_predictions = defaultdict(list)
    class_ground_truths = defaultdict(list)
    
    print("开始评估目标检测模型...")
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader, desc="评估进度")):
            images, targets = data
            
            # 将所有内容移动到与模型相同的设备
            device = next(model.parameters()).device
            images = [img.to(device) for img in images]
            
            # 获取模型预测
            if hasattr(model, 'detector'):
                # EnhancedFastRCNN模型
                detections = model.detector(images)
            else:
                # DetectionOnlyFastRCNN模型或原生FasterRCNN
                detections = model(images)
            
            # 处理每个图像的预测结果
            for i, (detection, target) in enumerate(zip(detections, targets)):
                # 获取预测边界框
                pred_boxes = detection['boxes'].cpu()
                pred_scores = detection['scores'].cpu()
                pred_labels = detection['labels'].cpu()
                
                # 应用置信度阈值
                keep = pred_scores > confidence_threshold
                pred_boxes = pred_boxes[keep]
                pred_scores = pred_scores[keep]
                pred_labels = pred_labels[keep]
                
                # 获取真值边界框
                gt_boxes = target['boxes'].cpu()
                gt_labels = target['labels'].cpu()
                
                # 存储当前图像的预测和真值
                img_pred = {
                    'boxes': pred_boxes,
                    'scores': pred_scores,
                    'labels': pred_labels,
                    'image_id': batch_idx * len(images) + i
                }
                img_gt = {
                    'boxes': gt_boxes,
                    'labels': gt_labels,
                    'image_id': batch_idx * len(images) + i
                }
                
                all_predictions.append(img_pred)
                all_ground_truths.append(img_gt)
                
                # 按类别存储预测和真值
                for label in range(1, num_classes + 1):
                    pred_idx = (pred_labels == label).nonzero(as_tuple=True)[0]
                    gt_idx = (gt_labels == label).nonzero(as_tuple=True)[0]
                    
                    class_predictions[label].append({
                        'boxes': pred_boxes[pred_idx] if len(pred_idx) > 0 else torch.zeros((0, 4)),
                        'scores': pred_scores[pred_idx] if len(pred_idx) > 0 else torch.zeros(0),
                        'image_id': batch_idx * len(images) + i
                    })
                    
                    class_ground_truths[label].append({
                        'boxes': gt_boxes[gt_idx] if len(gt_idx) > 0 else torch.zeros((0, 4)),
                        'image_id': batch_idx * len(images) + i
                    })
    
    # 计算总体评价指标
    print("计算评价指标...")
    overall_metrics = compute_map(all_predictions, all_ground_truths, iou_threshold)
    
    # 计算每个类别的评价指标
    class_metrics = {}
    for class_id in range(1, num_classes + 1):
        class_metrics[class_id] = compute_class_metrics(
            class_predictions[class_id], 
            class_ground_truths[class_id], 
            iou_threshold
        )
    
    # 汇总结果
    results = {
        'overall': overall_metrics,
        'per_class': class_metrics
    }
    
    # 计算平均指标
    average_precision = np.mean([metrics['AP'] for metrics in class_metrics.values()])
    average_recall = np.mean([metrics['recall'] for metrics in class_metrics.values()])
    average_f1 = np.mean([metrics['f1_score'] for metrics in class_metrics.values()])
    
    results['mAP'] = average_precision
    results['mRecall'] = average_recall
    results['mF1'] = average_f1
    
    # 保存结果到JSON文件
    with open(os.path.join(output_dir, 'detection_metrics.json'), 'w') as f:
        json.dump(results, f, indent=4, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
    
    # 可视化类别AP
    visualize_class_metrics(class_metrics, output_dir)
    
    # 打印总体指标
    print(f"mAP@{iou_threshold}: {average_precision:.4f}")
    print(f"Mean Recall: {average_recall:.4f}")
    print(f"Mean F1 Score: {average_f1:.4f}")
    
    # 打印每个类别的AP
    print("\n每个解剖区域的AP值:")
    for class_id, metrics in sorted(class_metrics.items()):
        print(f"区域 {class_id}: AP = {metrics['AP']:.4f}, Precision = {metrics['precision']:.4f}, Recall = {metrics['recall']:.4f}")
    
    return results


def compute_map(predictions, ground_truths, iou_threshold=0.5):
    """
    计算整体mAP
    """
    # 将所有图像的预测和真值合并
    all_gt_boxes = []
    all_gt_labels = []
    all_gt_image_ids = []
    
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_labels = []
    all_pred_image_ids = []
    
    for pred, gt in zip(predictions, ground_truths):
        all_gt_boxes.append(gt['boxes'])
        all_gt_labels.append(gt['labels'])
        all_gt_image_ids.extend([gt['image_id']] * len(gt['boxes']))
        
        all_pred_boxes.append(pred['boxes'])
        all_pred_scores.append(pred['scores'])
        all_pred_labels.append(pred['labels'])
        all_pred_image_ids.extend([pred['image_id']] * len(pred['boxes']))
    
    if len(all_gt_boxes) == 0 or len(all_pred_boxes) == 0:
        return {'mAP': 0.0}
    
    all_gt_boxes = torch.cat(all_gt_boxes) if all_gt_boxes else torch.zeros((0, 4))
    all_gt_labels = torch.cat(all_gt_labels) if all_gt_labels else torch.zeros(0, dtype=torch.int64)
    all_gt_image_ids = torch.tensor(all_gt_image_ids) if all_gt_image_ids else torch.zeros(0, dtype=torch.int64)
    
    all_pred_boxes = torch.cat(all_pred_boxes) if all_pred_boxes else torch.zeros((0, 4))
    all_pred_scores = torch.cat(all_pred_scores) if all_pred_scores else torch.zeros(0)
    all_pred_labels = torch.cat(all_pred_labels) if all_pred_labels else torch.zeros(0, dtype=torch.int64)
    all_pred_image_ids = torch.tensor(all_pred_image_ids) if all_pred_image_ids else torch.zeros(0, dtype=torch.int64)
    
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
    
    # 对每个预测，判断是否为真阳性
    for i in range(num_preds):
        pred_box = all_pred_boxes[i]
        pred_label = all_pred_labels[i]
        pred_image_id = all_pred_image_ids[i]
        
        # 找到同一图像中相同类别的所有真值边界框
        mask = (all_gt_labels == pred_label) & (all_gt_image_ids == pred_image_id)
        gt_boxes_same_class = all_gt_boxes[mask]
        
        if len(gt_boxes_same_class) == 0:
            fp[i] = 1  # 假阳性
            continue
        
        # 计算与所有真值边界框的IoU
        ious = box_iou(pred_box.unsqueeze(0), gt_boxes_same_class)[0]
        
        # 如果最大IoU大于阈值，则为真阳性
        if ious.max() >= iou_threshold:
            # 找到最大IoU对应的真值边界框
            max_iou_idx = ious.argmax()
            gt_idx = torch.where(mask)[0][max_iou_idx]
            
            # 检查该真值是否已被检测到
            already_matched = (tp.sum() > 0) and ((all_gt_image_ids == pred_image_id) & (all_gt_labels == pred_label)).sum() > 0
            
            if not already_matched:
                tp[i] = 1
            else:
                fp[i] = 1
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
        'mAP': ap.item(),
        'precision': precision[-1].item() if len(precision) > 0 else 0.0,
        'recall': recall[-1].item() if len(recall) > 0 else 0.0,
        'f1_score': 2 * precision[-1] * recall[-1] / (precision[-1] + recall[-1] + 1e-10) if len(precision) > 0 and len(recall) > 0 else 0.0
    }


def compute_class_metrics(class_predictions, class_ground_truths, iou_threshold=0.5):
    """
    计算单个类别的评价指标
    """
    # 提取所有预测和真值
    all_gt_boxes = []
    all_gt_image_ids = []
    
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_image_ids = []
    
    for pred, gt in zip(class_predictions, class_ground_truths):
        if len(gt['boxes']) > 0:
            all_gt_boxes.append(gt['boxes'])
            all_gt_image_ids.extend([gt['image_id']] * len(gt['boxes']))
        
        if len(pred['boxes']) > 0:
            all_pred_boxes.append(pred['boxes'])
            all_pred_scores.append(pred['scores'])
            all_pred_image_ids.extend([pred['image_id']] * len(pred['boxes']))
    
    # 如果没有预测或真值，返回零指标
    if not all_gt_boxes or not all_pred_boxes:
        return {
            'AP': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'TP': 0,
            'FP': 0,
            'FN': sum(len(gt['boxes']) for gt in class_ground_truths)
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
            gt_idx = gt_indices_same_image[max_iou_idx].item()
            
            # 检查该真值框是否已被匹配
            match_key = (pred_image_id.item(), gt_idx)
            if match_key not in matched_gt_boxes:
                tp[i] = 1
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
    f1_score = 2 * final_precision * final_recall / (final_precision + final_recall + 1e-10)
    
    # 计算TP、FP、FN
    TP = tp.sum().item()
    FP = fp.sum().item()
    FN = num_gts - TP
    
    return {
        'AP': ap,
        'precision': final_precision,
        'recall': final_recall,
        'f1_score': f1_score,
        'TP': int(TP),
        'FP': int(FP),
        'FN': int(FN),
        'precision_curve': precision.numpy(),
        'recall_curve': recall.numpy()
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
    可视化各类别指标
    """
    # 准备数据
    class_ids = list(class_metrics.keys())
    aps = [metrics['AP'] for metrics in class_metrics.values()]
    precisions = [metrics['precision'] for metrics in class_metrics.values()]
    recalls = [metrics['recall'] for metrics in class_metrics.values()]
    f1_scores = [metrics['f1_score'] for metrics in class_metrics.values()]
    
    # 按AP值排序
    sorted_indices = np.argsort(aps)[::-1]
    sorted_class_ids = [class_ids[i] for i in sorted_indices]
    sorted_aps = [aps[i] for i in sorted_indices]
    sorted_precisions = [precisions[i] for i in sorted_indices]
    sorted_recalls = [recalls[i] for i in sorted_indices]
    sorted_f1_scores = [f1_scores[i] for i in sorted_indices]
    
    # 绘制AP柱状图
    plt.figure(figsize=(14, 8))
    plt.bar(range(len(sorted_class_ids)), sorted_aps, color='skyblue')
    plt.xlabel('解剖区域ID')
    plt.ylabel('AP')
    plt.title('各解剖区域AP值')
    plt.xticks(range(len(sorted_class_ids)), sorted_class_ids, rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ap_by_class.png'))
    
    # 绘制精确度-召回率-F1散点图
    plt.figure(figsize=(10, 8))
    for i, class_id in enumerate(sorted_class_ids):
        plt.scatter(sorted_recalls[i], sorted_precisions[i], 
                   s=sorted_f1_scores[i]*100, alpha=0.6, 
                   label=f'区域 {class_id}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('各解剖区域的精确度-召回率关系（气泡大小代表F1分数）')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_recall_f1.png'))
    
    # 绘制各区域的精确度-召回率曲线
    plt.figure(figsize=(12, 10))
    
    # 选择AP值最高的10个类别绘制曲线
    top_n = min(10, len(sorted_class_ids))
    for i in range(top_n):
        class_id = sorted_class_ids[i]
        if 'precision_curve' in class_metrics[class_id] and 'recall_curve' in class_metrics[class_id]:
            plt.plot(class_metrics[class_id]['recall_curve'], 
                     class_metrics[class_id]['precision_curve'], 
                     label=f'区域 {class_id} (AP: {class_metrics[class_id]["AP"]:.4f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'前{top_n}个解剖区域的精确度-召回率曲线')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_pr_curves.png'))
    
    plt.close('all')


def test_detection_model(model, test_loader, device, output_dir='./test_results'):
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
    class_names = {i+1: f"Region_{i+1}" for i in range(29)}  # 可替换为实际的区域名称
    
    print("开始测试目标检测模型...")
    
    # 评估模型
    evaluation_results = evaluate_detection_model(
        model, 
        test_loader, 
        num_classes=29, 
        iou_threshold=0.5,
        confidence_threshold=0.5,
        output_dir=output_dir
    )
    
    # 打印总体性能
    print(f"\n总体性能指标:")
    print(f"mAP@0.5: {evaluation_results['mAP']:.4f}")
    print(f"Mean Recall: {evaluation_results['mRecall']:.4f}")
    print(f"Mean F1 Score: {evaluation_results['mF1']:.4f}")
    
    # 打印表现最好和最差的区域
    per_class_metrics = evaluation_results['per_class']
    class_aps = [(class_id, metrics['AP']) for class_id, metrics in per_class_metrics.items()]
    class_aps.sort(key=lambda x: x[1], reverse=True)
    
    print("\n表现最好的5个区域:")
    for class_id, ap in class_aps[:5]:
        metrics = per_class_metrics[class_id]
        print(f"区域 {class_id}: AP={ap:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1_score']:.4f}")
    
    print("\n表现最差的5个区域:")
    for class_id, ap in class_aps[-5:]:
        metrics = per_class_metrics[class_id]
        print(f"区域 {class_id}: AP={ap:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1_score']:.4f}")
    
    return evaluation_results


# 使用示例
def main():
    # 加载模型
    model = DetectionOnlyFastRCNN(num_regions=29)
    model.load_state_dict(torch.load("detection_model.pth"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 创建测试数据加载器 (假设已有)
    # test_loader = create_test_data_loader()
    
    # 执行测试
    # test_detection_model(model, test_loader, device, output_dir='./test_results')