#!/usr/bin/env python3
"""
统计在 generation_target = "all" (findings + impression 合并) 情况下的文本长度分布。

- 直接读取 MIMIC 注释，不加载图像，复用数据集的清洗逻辑
- 输出按 split（train/validate/test）与 overall 的长度统计
  - 字符数、词数（空格分词）、BERT token 数（可选，如本地可加载 tokenizer）
- 结果打印到控制台，并可选保存 CSV
"""

import os
import sys
import json
import math
import argparse
from typing import Dict, List

import numpy as np

# 项目内导入
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs import config
from datasets import MIMIC

try:
    from transformers import BertTokenizer
    _HAS_TOKENIZER = True
except Exception:
    _HAS_TOKENIZER = False


def build_target_text(findings: str, impression: str) -> str:
    """对齐 MIMIC.__getitem__ 在 generation_target = "all" 时的拼接逻辑。"""
    findings = findings or ""
    impression = impression or ""
    if findings and impression:
        return f"{findings} {impression}".strip()
    if impression:
        return impression.strip()
    return findings.strip()


def compute_basic_stats(values: List[int]) -> Dict[str, float]:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.int32)
    percentiles = [25, 50, 75, 90, 95, 99]
    stats = {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": int(arr.min()),
        "max": int(arr.max()),
    }
    for p in percentiles:
        stats[f"p{p}"] = float(np.percentile(arr, p))
    return stats


def print_stats(title: str, word_lens: List[int], char_lens: List[int], token_lens: List[int] = None):
    print(f"\n=== {title} ===")
    ws = compute_basic_stats(word_lens)
    cs = compute_basic_stats(char_lens)
    print("[words]   count={count} mean={mean:.2f} std={std:.2f} min={min} p50={p50:.0f} p75={p75:.0f} p90={p90:.0f} p95={p95:.0f} p99={p99:.0f} max={max}".format(**ws))
    print("[chars]   count={count} mean={mean:.2f} std={std:.2f} min={min} p50={p50:.0f} p75={p75:.0f} p90={p90:.0f} p95={p95:.0f} p99={p99:.0f} max={max}".format(**cs))
    if token_lens is not None:
        ts = compute_basic_stats(token_lens)
        print("[tokens]  count={count} mean={mean:.2f} std={std:.2f} min={min} p50={p50:.0f} p75={p75:.0f} p90={p90:.0f} p95={p95:.0f} p99={p99:.0f} max={max}".format(**ts))


def main():
    parser = argparse.ArgumentParser(description="Analyze text length distribution for generation_target=all")
    parser.add_argument("--save_csv", action="store_true", help="是否将统计结果保存到CSV")
    parser.add_argument("--ann_path", type=str, default=None, help="覆盖默认注释路径")
    args = parser.parse_args()

    ann_path = args.ann_path or config.ANN_DIR
    split_csv_path = getattr(config, "SPLIT_CSV_PATH", None)

    if not os.path.exists(ann_path):
        print(f"[Error] 注释文件不存在: {ann_path}")
        sys.exit(1)

    # 加载共享注释，不加载图像
    MIMIC.load_shared_data(
        directory=config.DATA_DIR,
        ann_dir=ann_path,
        mode=getattr(config, "MODE", "TRAIN"),
        binary_mode=True,
        split_csv_path=split_csv_path,
    )

    annotation = MIMIC._shared_data.get("annotation", {})
    if not annotation:
        print("[Error] 未加载到注释数据。")
        sys.exit(1)

    # 可选 tokenizer
    tokenizer = None
    if _HAS_TOKENIZER:
        try:
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
        except Exception:
            tokenizer = None

    # 统计容器
    results = {}
    overall_words, overall_chars, overall_tokens = [], [], []

    for split_name in ["train", "validate", "test"]:
        split_data = annotation.get(split_name, [])
        word_lens, char_lens, token_lens = [], [], []

        for item in split_data:
            # MIMIC.load_shared_data 已对 findings、impression 做过 _clean_report
            findings = item.get("findings", "")
            impression = item.get("impression", "")
            target_text = build_target_text(findings, impression)

            # 基础长度
            char_lens.append(len(target_text))
            word_lens.append(len(target_text.split()))

            # 可选：BERT token 长度
            if tokenizer is not None:
                encoded = tokenizer(
                    target_text,
                    max_length=getattr(config, "GEN_MAX_NEW_TOKENS", 150) + getattr(config, "MAX_LEN_HISTORY", 50),
                    truncation=True,
                )
                token_lens.append(len(encoded["input_ids"]))

        # 保存分布
        results[split_name] = {
            "word_lens": word_lens,
            "char_lens": char_lens,
            "token_lens": token_lens if tokenizer is not None else None,
        }

        # 打印统计
        print_stats(split_name, word_lens, char_lens, token_lens if tokenizer is not None else None)

        # 汇总 overall
        overall_words.extend(word_lens)
        overall_chars.extend(char_lens)
        if tokenizer is not None:
            overall_tokens.extend(token_lens)

    # Overall 统计
    print_stats("overall", overall_words, overall_chars, overall_tokens if overall_tokens else None)

    # 可选保存CSV
    if args.save_csv:
        import pandas as pd
        save_dir = getattr(config, "CHECKPOINT_PATH_TO", os.path.join(os.getcwd(), "results"))
        os.makedirs(save_dir, exist_ok=True)
        def dump(split_name: str, wl: List[int], cl: List[int], tl: List[int] = None):
            df = {
                "split": [split_name] * len(wl),
                "words": wl,
                "chars": cl,
            }
            if tl is not None:
                df["tokens"] = tl
            return pd.DataFrame(df)

        frames = []
        for split_name, d in results.items():
            frames.append(dump(split_name, d["word_lens"], d["char_lens"], d["token_lens"]))
        frames.append(dump("overall", overall_words, overall_chars, overall_tokens if overall_tokens else None))

        out_path = os.path.join(save_dir, "length_distribution_all.csv")
        pd.concat(frames, ignore_index=True).to_csv(out_path, index=False)
        print(f"\nCSV 已保存: {out_path}")


if __name__ == "__main__":
    main()


