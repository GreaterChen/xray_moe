"""Standalone utility to count positive/negative pairs for all contrastive losses.

Usage (example):
    python tools/contrastive_pair_counter.py --config-module configs.config \
        --batch-size 32 --num-workers 4 --mode train

The script only iterates over the dataset; it does not run any networks or update
parameters. It relies on anatomical embeddings / NLP status stored in the dataset.
"""
from __future__ import annotations

import argparse
import importlib
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import MIMIC, mimic_collate_fn
from utils.contrastive_stats import (
    summarize_clip_pairs,
    summarize_region_itc_pairs,
    summarize_simple_region_clip_pairs,
)

STATUS_TO_CODE = {"normal": 0, "abnormal": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count positive/negative pairs for pretrain strategies")
    parser.add_argument(
        "--config-module",
        default="configs.config",
        help="Python module that exposes a `config` object (default: configs.config)",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for traversal (defaults to config)")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader worker count (defaults to config)")
    parser.add_argument("--subset-size", type=int, default=None, help="Optional number of samples to scan")
    parser.add_argument(
        "--mode",
        default="train",
        choices=["train", "validate", "test"],
        help="Dataset split to traverse",
    )
    parser.add_argument(
        "--generation-target",
        default=None,
        choices=["findings", "all"],
        help="Override config.GENERATION_TARGET if provided",
    )
    parser.add_argument(
        "--anatomical-db",
        default=None,
        help="Override config.ANATOMICAL_DATABASE_PATH for region statistics",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show tqdm progress bar",
    )
    return parser.parse_args()


def load_config(module_name: str):
    module = importlib.import_module(module_name)
    if not hasattr(module, "config"):
        raise AttributeError(f"Module {module_name} does not expose a `config` object")
    return module.config


@dataclass
class RegionPairLists:
    batch_indices: List[int]
    region_indices: List[int]
    status_codes: List[int]
    group_batch_indices: List[int]
    group_ids: List[int]

    @property
    def num_pairs(self) -> int:
        return len(self.batch_indices)


def collect_region_pair_lists(batch: Dict[str, List]) -> RegionPairLists:
    batch_indices: List[int] = []
    region_indices: List[int] = []
    status_codes: List[int] = []
    group_batch_indices: List[int] = []
    group_ids: List[int] = []

    embeddings_batch = batch.get("anatomical_embeddings", [])
    nlp_status_batch = batch.get("anatomical_nlp_status", [])
    same_text_groups_batch = batch.get("same_text_region_groups", [])

    for b_idx, embeddings in enumerate(embeddings_batch):
        if not embeddings:
            continue

        status_dict = nlp_status_batch[b_idx] or {}
        same_text_groups = same_text_groups_batch[b_idx] or []

        region_to_group: Dict[int, int] = {}
        for group_id, region_group in enumerate(same_text_groups):
            for region_idx in region_group:
                region_to_group[int(region_idx)] = group_id

        for raw_region_idx in embeddings.keys():
            region_idx = int(raw_region_idx)
            batch_indices.append(b_idx)
            region_indices.append(region_idx - 1)  # 原实现使用 0-based

            status = status_dict.get(region_idx, None)
            status_codes.append(STATUS_TO_CODE.get(status, -1))

            group_batch_indices.append(b_idx)
            group_ids.append(region_to_group.get(region_idx, -1))

    return RegionPairLists(
        batch_indices=batch_indices,
        region_indices=region_indices,
        status_codes=status_codes,
        group_batch_indices=group_batch_indices,
        group_ids=group_ids,
    )


def build_region_masks(region_pairs: RegionPairLists) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if region_pairs.num_pairs < 2:
        return None

    device = torch.device("cpu")
    batch_indices = torch.tensor(region_pairs.batch_indices, device=device, dtype=torch.long)
    region_indices = torch.tensor(region_pairs.region_indices, device=device, dtype=torch.long)
    status_tensor = torch.tensor(region_pairs.status_codes, device=device, dtype=torch.long)
    group_batch_indices = torch.tensor(region_pairs.group_batch_indices, device=device, dtype=torch.long)
    group_ids = torch.tensor(region_pairs.group_ids, device=device, dtype=torch.long)

    same_region_mask = region_indices.unsqueeze(1) == region_indices.unsqueeze(0)
    diff_region_mask = ~same_region_mask

    same_sample_mask = batch_indices.unsqueeze(1) == batch_indices.unsqueeze(0)
    diff_sample_mask = ~same_sample_mask

    same_status_mask = status_tensor.unsqueeze(1) == status_tensor.unsqueeze(0)
    diff_status_mask = ~same_status_mask

    valid_status_mask = (status_tensor >= 0)
    valid_pair_mask = valid_status_mask.unsqueeze(1) & valid_status_mask.unsqueeze(0)

    valid_group_mask = group_ids >= 0
    valid_group_pair_mask = valid_group_mask.unsqueeze(1) & valid_group_mask.unsqueeze(0)
    same_group_mask = group_ids.unsqueeze(1) == group_ids.unsqueeze(0)
    same_text_mask = same_sample_mask & same_group_mask & valid_group_pair_mask

    positive_mask = same_region_mask & same_status_mask & valid_pair_mask

    negative_mask_same_region_diff_status = same_region_mask & diff_status_mask & valid_pair_mask
    negative_mask_same_sample_diff_region = same_sample_mask & diff_region_mask & (~same_text_mask)
    negative_mask_diff_sample_diff_region = diff_sample_mask & diff_region_mask
    negative_mask = (
        negative_mask_same_region_diff_status
        | negative_mask_same_sample_diff_region
        | negative_mask_diff_sample_diff_region
    )

    return positive_mask, negative_mask


class StrategyStatTracker:
    def __init__(self, name: str):
        self.name = name
        self.total_pairs = 0
        self.positives = 0
        self.negatives = 0
        self.valid_batches = 0

    def update(self, stats: Optional[Dict[str, int]]):
        if not stats:
            return
        self.total_pairs += stats.get("total_pairs", 0)
        self.positives += stats.get("positives", 0)
        self.negatives += stats.get("negatives", 0)
        self.valid_batches += 1

    def summary(self) -> Dict[str, float]:
        avg_pos = (self.positives / self.valid_batches) if self.valid_batches else 0.0
        avg_neg = (self.negatives / self.valid_batches) if self.valid_batches else 0.0
        return {
            "strategy": self.name,
            "valid_batches": self.valid_batches,
            "total_pairs": self.total_pairs,
            "total_positives": self.positives,
            "total_negatives": self.negatives,
            "avg_positives_per_batch": avg_pos,
            "avg_negatives_per_batch": avg_neg,
        }


def prepare_dataloader(cfg, args: argparse.Namespace) -> DataLoader:
    batch_size = args.batch_size or getattr(cfg, "TRAIN_BATCH_SIZE", 32)
    num_workers = args.num_workers if args.num_workers is not None else getattr(cfg, "NUM_WORKERS", 4)

    generation_target = args.generation_target or getattr(cfg, "GENERATION_TARGET", "findings")
    dataset = MIMIC(
        directory=cfg.DATA_DIR,
        ann_dir=cfg.ANN_DIR,
        images_dir=cfg.IMAGES_DIR,
        input_size=(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
        random_transform=False,
        tokenizer=None,
        mode=args.mode,
        subset_size=args.subset_size,
        generation_target=generation_target,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=mimic_collate_fn,
    )
    return loader


def ensure_anatomical_db(cfg, args: argparse.Namespace):
    db_path = args.anatomical_db or getattr(cfg, "ANATOMICAL_DATABASE_PATH", None)
    if db_path and os.path.exists(db_path):
        MIMIC.load_anatomical_embeddings(db_path)
    else:
        print("⚠️  Anatomical database path not provided or does not exist; region statistics may be empty.")


def main():
    args = parse_args()
    cfg = load_config(args.config_module)

    ensure_anatomical_db(cfg, args)
    dataloader = prepare_dataloader(cfg, args)

    clip_tracker = StrategyStatTracker("clip")
    simple_region_tracker = StrategyStatTracker("simple_region_clip")
    region_itc_tracker = StrategyStatTracker("region_itc")

    iterator = tqdm(dataloader, desc="Scanning batches") if args.progress else dataloader

    for batch in iterator:
        batch_size = batch["image"].shape[0]
        clip_tracker.update(summarize_clip_pairs(batch_size))

        region_pairs = collect_region_pair_lists(batch)
        if region_pairs.num_pairs > 0:
            simple_region_tracker.update(
                summarize_simple_region_clip_pairs(region_pairs.num_pairs)
            )

            masks = build_region_masks(region_pairs)
            if masks is not None:
                region_itc_tracker.update(summarize_region_itc_pairs(*masks))
        else:
            simple_region_tracker.update(None)
            region_itc_tracker.update(None)

    print("\n=== Contrastive Pair Statistics ===")
    for tracker in (clip_tracker, simple_region_tracker, region_itc_tracker):
        stats = tracker.summary()
        print(f"\n[{stats['strategy']}] batches: {stats['valid_batches']}")
        print(f"  Total positives: {stats['total_positives']} | Total negatives: {stats['total_negatives']}")
        print(
            f"  Avg positives/batch: {stats['avg_positives_per_batch']:.2f} | Avg negatives/batch: {stats['avg_negatives_per_batch']:.2f}"
        )


if __name__ == "__main__":
    main()
