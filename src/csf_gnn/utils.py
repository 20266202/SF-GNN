"""
Shared utilities for CSF-GNN experiments.

This repo was refactored from a single notebook into importable modules so that:
- each dataset can be run from a small script under `scripts/`
- core logic is reused and unit-testable
"""
from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import coalesce, degree, remove_self_loops, to_undirected


# ---------------------------- Repro ----------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Best-effort determinism (still depends on ops/CUDA)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------- Printing ----------------------------

def print_header(msg: str) -> None:
    bar = "=" * 80
    print(f"\n{bar}\n{msg}\n{bar}")


# ---------------------------- Graph IO ----------------------------

def undirected_unique_edge_count(edge_index: torch.Tensor, num_nodes: int) -> int:
    ei = to_undirected(edge_index, num_nodes=num_nodes)
    ei, _ = remove_self_loops(ei)
    ei = coalesce(ei, None, num_nodes, num_nodes)[0]
    return ei.size(1)


def symmetrize_adj(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    ei = to_undirected(edge_index, num_nodes=num_nodes)
    ei, _ = remove_self_loops(ei)
    ei = coalesce(ei, None, num_nodes, num_nodes)[0]
    return ei


def read_and_validate_edges(edge_path: str, num_nodes: int) -> torch.Tensor:
    if not os.path.exists(edge_path):
        raise FileNotFoundError(f"Missing file: {edge_path}")

    edges: List[Tuple[int, int]] = []
    with open(edge_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a, b = line.split()
            u, v = int(a), int(b)
            if u < 0 or v < 0 or u >= num_nodes or v >= num_nodes:
                raise ValueError(f"Edge ({u},{v}) out of range for num_nodes={num_nodes}")
            edges.append((u, v))

    if len(edges) == 0:
        raise ValueError(f"No edges found in: {edge_path}")

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


# ---------------------------- Splits ----------------------------

def stratified_masks(
    y: torch.Tensor,
    train_ratio: float = 0.5,
    val_ratio: float = 0.25,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    y_np = y.cpu().numpy()
    n = len(y_np)
    tv_ratio = train_ratio + val_ratio

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=1 - tv_ratio, random_state=seed)
    tv_idx, te_idx = next(sss1.split(np.zeros(n), y_np))

    y_tv = y_np[tv_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio / tv_ratio, random_state=seed)
    tr_rel, va_rel = next(sss2.split(np.zeros(len(tv_idx)), y_tv))

    tr_idx = tv_idx[tr_rel]
    va_idx = tv_idx[va_rel]

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    train_mask[tr_idx] = True
    val_mask[va_idx] = True
    test_mask[te_idx] = True

    return {"train": train_mask, "val": val_mask, "test": test_mask}


def capped_class_masks(
    labels: torch.Tensor,
    train_ratio: float = 0.5,
    val_ratio: float = 0.25,
    max_per_class: int = 100,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    y = labels.cpu().numpy()
    rng = np.random.RandomState(seed)

    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    p1 = train_ratio
    p2 = train_ratio + val_ratio

    train0 = idx0[: min(int(p1 * len(idx0)), max_per_class)]
    train1 = idx1[: min(int(p1 * len(idx1)), max_per_class)]

    val0 = idx0[int(p1 * len(idx0)) : int(p2 * len(idx0))]
    val1 = idx1[int(p1 * len(idx1)) : int(p2 * len(idx1))]

    test0 = idx0[int(p2 * len(idx0)) :]
    test1 = idx1[int(p2 * len(idx1)) :]

    train_mask = np.zeros(len(y), dtype=bool)
    val_mask = np.zeros(len(y), dtype=bool)
    test_mask = np.zeros(len(y), dtype=bool)

    train_mask[np.concatenate([train0, train1])] = True
    val_mask[np.concatenate([val0, val1])] = True
    test_mask[np.concatenate([test0, test1])] = True

    return {
        "train": torch.from_numpy(train_mask),
        "val": torch.from_numpy(val_mask),
        "test": torch.from_numpy(test_mask),
    }


# ---------------------------- Feature preprocess ----------------------------

def standardize_features(x: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(x)


# ---------------------------- Dataset stats ----------------------------

def dataset_stats(edge_index: torch.Tensor, x: torch.Tensor) -> Dict[str, int]:
    n = x.size(0)
    return {
        "n_nodes": n,
        "n_edges": int(undirected_unique_edge_count(edge_index, n)),
        "n_features": int(x.size(1)),
    }
