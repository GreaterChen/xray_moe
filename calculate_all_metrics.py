 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算CSV文件中的NLG和CE指标
支持从CSV文件读取数据并计算所有评估指标
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from metrics import compute_scores
from tools.metrics_clinical import CheXbertMetrics
from utils import setup_logger


class MetricsCalculator:
    """指标计算器类"""
    
    def __init__(self, chexbert_checkpoint_path: Optional[str] = None, device: str = "cuda"):
        """
        初始化指标计算器
        
        Args:
            chexbert_checkpoint_path: CheXbert模型检查点路径
            device: 计算设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.logger = setup_logger()
        
        # 初始化CheXbert评估器
        self.chexbert_metrics = None
        if chexbert_checkpoint_path and os.path.exists(chexbert_checkpoint_path):
            try:
                self.chexbert_metrics = CheXbertMetrics(
                    checkpoint_path=chexbert_checkpoint_path,
                    mbatch_size=16,
                    device=self.device
                )
                self.logger.info(f"成功初始化CheXbert评估器，使用设备: {self.device}")
            except Exception as e:
                self.logger.warning(f"初始化CheXbert评估器失败: {e}")
        else:
            self.logger.warning("未提供CheXbert检查点路径或路径不存在，将跳过CE指标计算")
    
    def load_csv_data(self, csv_path: str) -> Tuple[List[str], List[str], Optional[List]]:
        """
        从CSV文件加载数据
        
        Args:
            csv_path: CSV文件路径
            
        Returns:
            ground_truths: 真实报告列表
            predictions: 预测报告列表
            labels: 标签列表（可选）
        """
        try:
            df = pd.read_csv(csv_path)
            self.logger.info(f"成功读取CSV文件: {csv_path}，共{len(df)}行")
            
            # 检查必要的列
            required_columns = ["findings_gt", "findings_pred"]
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"CSV文件缺少必要的列: {required_columns}")
            
            # 提取数据
            ground_truths = df["findings_gt"].tolist()
            predictions = df["findings_pred"].tolist()
            
            # 检查是否有标签列
            labels = None
            if "labels" in df.columns:
                labels = df["labels"].tolist()
                self.logger.info("检测到标签列，将用于额外分析")
            
            # 数据清理和验证
            ground_truths = [str(gt).strip() for gt in ground_truths if pd.notna(gt)]
            predictions = [str(pred).strip() for pred in predictions if pd.notna(pred)]
            
            if len(ground_truths) != len(predictions):
                raise ValueError("真实报告和预测报告数量不匹配")
            
            self.logger.info(f"有效数据样本数: {len(ground_truths)}")
            return ground_truths, predictions, labels
            
        except Exception as e:
            self.logger.error(f"读取CSV文件时出错: {e}")
            raise
    
    def calculate_nlg_metrics(self, ground_truths: List[str], predictions: List[str]) -> Dict:
        """
        计算NLG指标（BLEU, METEOR, ROUGE_L, CIDEr）
        
        Args:
            ground_truths: 真实报告列表
            predictions: 预测报告列表
            
        Returns:
            NLG指标字典
        """
        try:
            self.logger.info("开始计算NLG指标...")
            
            # 准备数据格式
            gts = {i: [gt] for i, gt in enumerate(ground_truths)}
            res = {i: [pred] for i, pred in enumerate(predictions)}
            
            # 计算指标
            nlg_metrics = compute_scores(gts, res)
            
            self.logger.info("NLG指标计算完成")
            return nlg_metrics
            
        except Exception as e:
            self.logger.error(f"计算NLG指标时出错: {e}")
            return {}
    
    def calculate_ce_metrics(self, ground_truths: List[str], predictions: List[str]) -> Dict:
        """
        计算CE指标（CheXbert临床评估指标）
        
        Args:
            ground_truths: 真实报告列表
            predictions: 预测报告列表
            
        Returns:
            CE指标字典
        """
        if self.chexbert_metrics is None:
            self.logger.warning("CheXbert评估器未初始化，跳过CE指标计算")
            return {}
        
        try:
            self.logger.info("开始计算CE指标...")
            
            # 计算CheXbert指标
            ce_metrics = self.chexbert_metrics.compute(ground_truths, predictions)
            
            self.logger.info("CE指标计算完成")
            return ce_metrics
            
        except Exception as e:
            self.logger.error(f"计算CE指标时出错: {e}")
            return {}
    
    def calculate_all_metrics(self, csv_path: str, output_dir: Optional[str] = None) -> Dict:
        """
        计算所有指标
        
        Args:
            csv_path: CSV文件路径
            output_dir: 输出目录
            
        Returns:
            包含所有指标的字典
        """
        # 设置输出目录
        if output_dir is None:
            output_dir = os.path.dirname(csv_path) or "."
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载数据
        ground_truths, predictions, labels = self.load_csv_data(csv_path)
        
        # 计算NLG指标
        nlg_metrics = self.calculate_nlg_metrics(ground_truths, predictions)
        
        # 计算CE指标
        ce_metrics = self.calculate_ce_metrics(ground_truths, predictions)
        
        # 汇总所有指标
        all_metrics = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "csv_file": os.path.basename(csv_path),
            "num_samples": len(ground_truths),
            "nlg_metrics": nlg_metrics,
            "ce_metrics": ce_metrics
        }
        
        # 保存结果
        self.save_results(all_metrics, output_dir, csv_path)
        
        return all_metrics
    
    def save_results(self, metrics: Dict, output_dir: str, csv_path: str):
        """
        保存计算结果
        
        Args:
            metrics: 指标字典
            output_dir: 输出目录
            csv_path: 原始CSV文件路径
        """
        try:
            # 生成输出文件名
            base_name = os.path.splitext(os.path.basename(csv_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存详细指标到JSON文件
            import json
            json_path = os.path.join(output_dir, f"{base_name}_metrics_{timestamp}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            
            # 保存指标摘要到CSV文件
            summary_data = {
                "metric_type": [],
                "metric_name": [],
                "metric_value": []
            }
            
            # 添加NLG指标
            for metric_name, value in metrics["nlg_metrics"].items():
                summary_data["metric_type"].append("NLG")
                summary_data["metric_name"].append(metric_name)
                summary_data["metric_value"].append(value)
            
            # 添加CE指标
            for metric_name, value in metrics["ce_metrics"].items():
                summary_data["metric_type"].append("CE")
                summary_data["metric_name"].append(metric_name)
                summary_data["metric_value"].append(value)
            
            summary_df = pd.DataFrame(summary_data)
            csv_path = os.path.join(output_dir, f"{base_name}_summary_{timestamp}.csv")
            summary_df.to_csv(csv_path, index=False)
            
            self.logger.info(f"结果已保存到:")
            self.logger.info(f"  - 详细指标: {json_path}")
            self.logger.info(f"  - 指标摘要: {csv_path}")
            
        except Exception as e:
            self.logger.error(f"保存结果时出错: {e}")
    
    def print_results(self, metrics: Dict):
        """
        打印计算结果
        
        Args:
            metrics: 指标字典
        """
        print("\n" + "="*60)
        print("评估指标计算结果")
        print("="*60)
        print(f"文件: {metrics['csv_file']}")
        print(f"样本数: {metrics['num_samples']}")
        print(f"时间: {metrics['timestamp']}")
        
        # 打印NLG指标
        if metrics['nlg_metrics']:
            print("\nNLG指标:")
            print("-" * 30)
            for metric_name, value in metrics['nlg_metrics'].items():
                print(f"{metric_name:12}: {value:.4f}")
        
        # 打印CE指标
        if metrics['ce_metrics']:
            print("\nCE指标 (CheXbert临床评估):")
            print("-" * 30)
            for metric_name, value in metrics['ce_metrics'].items():
                print(f"{metric_name:12}: {value:.4f}")
        
        print("="*60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="计算CSV文件中的NLG和CE指标")
    parser.add_argument("--csv_path", help="输入CSV文件路径", default="/mnt/chenlb/xray_moe/results/finetune_bert_vit_instruction_moe_odd_extra_2itc/test_results/val_results_epoch_49.csv")
    parser.add_argument("--chexbert_path", help="CheXbert模型检查点路径", default="/home/chenlb/xray_moe/tools/chexbert.pth")
    parser.add_argument("--output_dir", help="输出目录", default="/mnt/chenlb/xray_moe/results/finetune_bert_vit/")
    parser.add_argument("--device", default="cuda", help="计算设备 (cuda/cpu)")
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.csv_path):
        print(f"错误: CSV文件不存在: {args.csv_path}")
        sys.exit(1)
    
    try:
        # 创建指标计算器
        calculator = MetricsCalculator(
            chexbert_checkpoint_path=args.chexbert_path,
            device=args.device
        )
        
        # 计算所有指标
        metrics = calculator.calculate_all_metrics(args.csv_path, args.output_dir)
        
        # 打印结果
        calculator.print_results(metrics)
        
    except Exception as e:
        print(f"计算指标时出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()