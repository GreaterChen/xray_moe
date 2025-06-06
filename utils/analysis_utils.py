"""结果分析工具函数"""
import re
import pandas as pd
from utils.logging_utils import clean_report_mimic_cxr


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

