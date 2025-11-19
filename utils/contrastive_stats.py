"""Utilities for counting positive/negative pairs in contrastive objectives."""
from __future__ import annotations

from typing import Dict, Optional

import torch


def summarize_clip_pairs(batch_size: int) -> Dict[str, int]:
    """Return positive/negative counts for global CLIP-style loss.

    Args:
        batch_size: Number of (image, text) pairs in the batch.
    """
    positives = max(batch_size, 0)
    total = batch_size * batch_size
    negatives = max(total - positives, 0)
    return {
        "strategy": "clip",
        "batch_size": batch_size,
        "total_pairs": total,
        "positives": positives,
        "negatives": negatives,
    }


def summarize_simple_region_clip_pairs(num_pairs: int) -> Dict[str, int]:
    """Return stats for simplified region-level CLIP loss.

    Args:
        num_pairs: Number of valid region-text pairs (N).
    """
    positives = max(num_pairs, 0)
    total = num_pairs * num_pairs
    negatives = max(total - positives, 0)
    return {
        "strategy": "simple_region_clip",
        "num_pairs": num_pairs,
        "total_pairs": total,
        "positives": positives,
        "negatives": negatives,
    }


def summarize_region_itc_pairs(
    positive_mask: torch.Tensor, negative_mask: torch.Tensor
) -> Optional[Dict[str, int]]:
    """Return stats for region ITC loss using the actual masks."""
    if positive_mask is None or negative_mask is None:
        return None
    if positive_mask.dim() != 2 or negative_mask.dim() != 2:
        return None
    if positive_mask.shape != negative_mask.shape:
        return None

    total_pairs = positive_mask.numel()
    positives = int(positive_mask.sum().item())
    negatives = int(negative_mask.sum().item())
    return {
        "strategy": "region_itc",
        "total_pairs": total_pairs,
        "positives": positives,
        "negatives": negatives,
    }
