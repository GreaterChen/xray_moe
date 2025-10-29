import os
import pandas as pd
import argparse
from utils import analyze_results_from_csv, setup_logger
from metrics import compute_scores


def parse_args():
    parser = argparse.ArgumentParser(description="分析实验结果")

    parser.add_argument(
        "--csv_path",
        default="/home/chenlb/xray_report_generation/results/stage3/baseline/test_results/test_results_epoch_1.csv",
        type=str,
        help="结果CSV文件的路径",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 初始化logger
    logger, log_file = setup_logger(log_dir="logs", is_main_process=True)
    logger.info(f"日志文件: {log_file}")

    # 检查文件是否存在
    if not os.path.exists(args.csv_path):
        logger.error(f"CSV文件不存在: {args.csv_path}")
        raise ValueError(f"CSV文件不存在: {args.csv_path}")

    logger.info(f"\n分析文件: {os.path.basename(args.csv_path)}")

    results = analyze_results_from_csv(args.csv_path, metric_ftns=compute_scores)

    # 打印Findings指标
    logger.info("\nFindings Metrics:")
    for metric_name, value in results["findings_metrics"].items():
        logger.info(f"{metric_name}: {value:.4f}")

    # 打印Impression指标（如果存在）
    if results["impression_metrics"]:
        logger.info("\nImpression Metrics:")
        for metric_name, value in results["impression_metrics"].items():
            logger.info(f"{metric_name}: {value:.4f}")

    # 打印Combined指标（如果存在）
    if results["combined_metrics"]:
        logger.info("\nCombined Metrics:")
        for metric_name, value in results["combined_metrics"].items():
            logger.info(f"{metric_name}: {value:.4f}")

    logger.info("-" * 50)


if __name__ == "__main__":
    main()
