import pandas as pd
import argparse
import os
import logging
from datetime import datetime
import json
import torch
import sys

# 导入项目中已有的CheXbert评估器
sys.path.append('.')
from tools.metrics_clinical import CheXbertMetrics

def setup_logger():
    """设置简单的日志记录器"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def calculate_ce_metrics(csv_path, output_dir=None, checkpoint_path=None):
    """
    计算CE指标
    
    Args:
        csv_path: CSV文件路径
        output_dir: 输出目录
        checkpoint_path: CheXbert模型检查点路径
    """
    logger = setup_logger()
    logger.info(f"开始处理CSV文件: {csv_path}")
    
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.dirname(csv_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"成功读取CSV文件，共{len(df)}行")
    except Exception as e:
        logger.error(f"读取CSV文件时出错: {e}")
        return
    
    # 检查列名
    column_pairs = [
        ["ground_truth", "generated_report"],
        ["findings_gt", "findings_pred"],
        ["gt", "pred"],
        ["reference", "hypothesis"]
    ]
    
    # 尝试找到正确的列名
    found_columns = None
    for cols in column_pairs:
        if all(col in df.columns for col in cols):
            found_columns = cols
            break
    
    if found_columns is None:
        logger.error(f"CSV文件缺少必要的列。支持的列名对: {column_pairs}")
        logger.info(f"当前CSV文件的列: {list(df.columns)}")
        return
    
    # 提取真实报告和生成报告
    gt_col, pred_col = found_columns
    ground_truths = df[gt_col].tolist()
    generated_reports = df[pred_col].tolist()
    
    logger.info(f"使用列 '{gt_col}' 作为真实报告，'{pred_col}' 作为生成报告")
    
    # 初始化CheXbert评估器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if checkpoint_path is None:
        # 使用默认路径
        checkpoint_path = "tools/chexbert.pth"
        if not os.path.exists(checkpoint_path):
            logger.error(f"未找到CheXbert检查点: {checkpoint_path}")
            logger.error("请提供正确的检查点路径")
            return
    
    try:
        chexbert_metrics = CheXbertMetrics(
            checkpoint_path=checkpoint_path,
            mbatch_size=16,
            device=device
        )
        logger.info("CheXbert评估器初始化成功")
    except Exception as e:
        logger.error(f"初始化CheXbert评估器失败: {e}")
        return
    
    # 计算CE指标
    logger.info("开始计算CE指标...")
    try:
        ce_metrics = chexbert_metrics.compute(ground_truths, generated_reports)
        logger.info("CE指标计算完成")
        
        # 保存结果到JSON文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"ce_metrics_{timestamp}.json")
        
        with open(output_file, "w") as f:
            json.dump(ce_metrics, f, indent=2)
        
        logger.info(f"结果已保存到: {output_file}")
        
        # 打印主要指标
        logger.info("\n主要CE指标:")
        for metric_name, value in ce_metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")
        
    except Exception as e:
        logger.error(f"计算CE指标时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description="计算放射学报告的临床评估(CE)指标")
    parser.add_argument("csv_path", default="/home/chenlb/xray_moe/test_predictions_epoch_16.csv", help="包含真实报告和生成报告的CSV文件路径")
    parser.add_argument("--output_dir", default="ce_metrics/", help="输出目录，默认为CSV文件所在目录")
    parser.add_argument("--checkpoint", default="", help="CheXbert模型检查点路径")
    
    args = parser.parse_args()
    
    calculate_ce_metrics(args.csv_path, args.output_dir, args.checkpoint)

if __name__ == "__main__":
    main() 