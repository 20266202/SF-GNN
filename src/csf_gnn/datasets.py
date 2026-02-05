"""
Dataset loaders.

Each loader reads:
- a node CSV (features + label + sensitive attribute)
- an edge list text file "u v" per line

Returns:
edge_index (LongTensor [2,E]), X (FloatTensor [N,F]), y (LongTensor [N]),
sensitive (LongTensor [N]), masks dict(train/val/test)
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .utils import (
    capped_class_masks,
    print_header,
    read_and_validate_edges,
    standardize_features,
    stratified_masks,
    symmetrize_adj,
)


@dataclass(frozen=True)
class LoadedDataset:
    edge_index: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    s: torch.Tensor
    masks: Dict[str, torch.Tensor]


def _finalize(
    df: pd.DataFrame,
    predict_attr: str,
    sens_attr: str,
    edge_path: str,
    drop_cols: Sequence[str],
    standardize: bool,
    masks: Dict[str, torch.Tensor],
) -> LoadedDataset:
    labels = df[predict_attr].to_numpy(copy=True)

    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError(f"Labels in {predict_attr} must be binary 0/1 after preprocessing.")

    sens = df[sens_attr].to_numpy(copy=True)
    if not set(np.unique(sens)).issubset({0, 1}):
        raise ValueError(f"Sensitive attr {sens_attr} must be binary 0/1 after preprocessing.")

    keep_cols = [c for c in df.columns if c not in set(drop_cols)]
    x_np = df[keep_cols].to_numpy(dtype=np.float32, copy=True)

    if standardize:
        x_np = standardize_features(x_np)

    x = torch.from_numpy(x_np).float()
    y = torch.from_numpy(labels).long()
    s = torch.from_numpy(sens).long()

    edge_index = read_and_validate_edges(edge_path, num_nodes=len(df))
    edge_index = symmetrize_adj(edge_index, num_nodes=len(df))

    # sanity: masks length matches
    for k, m in masks.items():
        if len(m) != len(df):
            raise ValueError(f"Mask {k} has length {len(m)} but N={len(df)}")

    return LoadedDataset(edge_index=edge_index, x=x, y=y, s=s, masks=masks)


# ---------------------------- German ----------------------------

def load_german(
    csv_path: str = "german.csv",
    edge_path: str = "german_edges.txt",
    standardize: bool = True,
    drop_cols: Sequence[str] = ("GoodCustomer", "OtherLoansAtStore", "PurposeOfLoan"),
    seed: int = 42,
) -> LoadedDataset:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing file: {csv_path}")
    if not os.path.exists(edge_path):
        raise FileNotFoundError(f"Missing file: {edge_path}")

    print_header("Loading german dataset")
    df = pd.read_csv(csv_path)

    predict_attr = "GoodCustomer"
    sens_attr = "Gender"

    labels = df[predict_attr].to_numpy(copy=True)
    # notebook maps -1 -> 0
    labels = np.where(labels == -1, 0, labels).astype(int)
    df[predict_attr] = labels

    mapping = {"Female": 1, "Male": 0, "female": 1, "male": 0, "F": 1, "M": 0}
    df[sens_attr] = df[sens_attr].astype(str).map(mapping)
    if df[sens_attr].isna().any():
        raise ValueError("German: could not map some Gender values to {0,1}.")
    df[sens_attr] = df[sens_attr].astype(int)

    masks = stratified_masks(torch.from_numpy(df[predict_attr].to_numpy()).long(), seed=seed)
    return _finalize(df, predict_attr, sens_attr, edge_path, drop_cols, standardize, masks)


# ---------------------------- NBA ----------------------------

def load_nba(
    csv_path: str = "nba.csv",
    edge_path: str = "nba_edges.txt",
    standardize: bool = True,
    drop_cols: Sequence[str] = ("SALARY",),
    seed: int = 42,
    train_ratio: float = 0.5,
    val_ratio: float = 0.25,
    max_per_class: int = 50,
) -> LoadedDataset:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing file: {csv_path}")
    if not os.path.exists(edge_path):
        raise FileNotFoundError(f"Missing file: {edge_path}")

    print_header("Loading nba dataset")
    df = pd.read_csv(csv_path)

    predict_attr = "SALARY"
    sens_attr = "country"

    # notebook removes -1 labels rows
    labels = df[predict_attr].to_numpy(copy=True)
    valid_mask = labels >= 0
    df = df[valid_mask].reset_index(drop=True)

    # label conversion: (SALARY > median)?? Let's follow notebook: it binarizes using median.
    labels = df[predict_attr].to_numpy(copy=True)
    median = np.median(labels)
    df[predict_attr] = (labels > median).astype(int)

    # sensitive mapping: USA vs others?
    df[sens_attr] = df[sens_attr].astype(str).str.lower()
    df[sens_attr] = (df[sens_attr] != "usa").astype(int)  # matches notebook: USA=0, non-USA=1
    df[sens_attr] = df[sens_attr].astype(int)

    masks = capped_class_masks(
        torch.from_numpy(df[predict_attr].to_numpy()).long(),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        max_per_class=max_per_class,
        seed=seed,
    )
    return _finalize(df, predict_attr, sens_attr, edge_path, drop_cols, standardize, masks)


# ---------------------------- Bail ----------------------------

def load_bail(
    csv_path: str = "bail.csv",
    edge_path: str = "bail_edges.txt",
    standardize: bool = True,
    drop_cols: Sequence[str] = ("RECID",),
    seed: int = 42,
    train_ratio: float = 0.5,
    val_ratio: float = 0.25,
    max_per_class: int = 100,
) -> LoadedDataset:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing file: {csv_path}")
    if not os.path.exists(edge_path):
        raise FileNotFoundError(f"Missing file: {edge_path}")

    print_header("Loading bail dataset")
    df = pd.read_csv(csv_path)

    predict_attr = "RECID"
    sens_attr = "WHITE"

    # ensure ints
    df[predict_attr] = pd.to_numeric(df[predict_attr], errors="coerce").fillna(0).astype(int)
    df[sens_attr] = pd.to_numeric(df[sens_attr], errors="coerce").fillna(0).astype(int)
    df[sens_attr] = (df[sens_attr] > 0).astype(int)

    masks = capped_class_masks(
        torch.from_numpy(df[predict_attr].to_numpy()).long(),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        max_per_class=max_per_class,
        seed=seed,
    )
    return _finalize(df, predict_attr, sens_attr, edge_path, drop_cols, standardize, masks)


# ---------------------------- Credit ----------------------------

def load_credit(
    csv_path: str = "credit.csv",
    edge_path: str = "credit_edges.txt",
    standardize: bool = True,
    drop_cols: Sequence[str] = ("NoDefaultNextMonth", "Single"),
    seed: int = 42,
    train_ratio: float = 0.5,
    val_ratio: float = 0.25,
    max_per_class: int = 500,
) -> LoadedDataset:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing file: {csv_path}")
    if not os.path.exists(edge_path):
        raise FileNotFoundError(f"Missing file: {edge_path}")

    print_header("Loading credit dataset")
    df = pd.read_csv(csv_path)

    predict_attr = "NoDefaultNextMonth"
    sens_attr = "Age"

    df[predict_attr] = pd.to_numeric(df[predict_attr], errors="coerce").fillna(0).astype(int)

    # Age -> binary (>= median) to match notebook logic: it bins by median
    df[sens_attr] = pd.to_numeric(df[sens_attr], errors="coerce").fillna(0).astype(int)
    med = int(np.median(df[sens_attr].to_numpy()))
    df[sens_attr] = (df[sens_attr] >= med).astype(int)

    masks = capped_class_masks(
        torch.from_numpy(df[predict_attr].to_numpy()).long(),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        max_per_class=max_per_class,
        seed=seed,
    )
    return _finalize(df, predict_attr, sens_attr, edge_path, drop_cols, standardize, masks)


# ---------------------------- Income ----------------------------

def load_income(
    csv_path: str = "income.csv",
    edge_path: str = "income_edges.txt",
    standardize: bool = True,
    drop_cols: Sequence[str] = ("income",),
    seed: int = 42,
    train_ratio: float = 0.5,
    val_ratio: float = 0.25,
    max_per_class: int = 500,
) -> LoadedDataset:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing file: {csv_path}")
    if not os.path.exists(edge_path):
        raise FileNotFoundError(f"Missing file: {edge_path}")

    print_header("Loading income dataset")
    df = pd.read_csv(csv_path)

    predict_attr = "income"
    sens_attr = "race"

    df[predict_attr] = pd.to_numeric(df[predict_attr], errors="coerce").fillna(0).astype(int)
    df[sens_attr] = pd.to_numeric(df[sens_attr], errors="coerce").fillna(0).astype(int)
    df[sens_attr] = (df[sens_attr] > 0).astype(int)

    masks = capped_class_masks(
        torch.from_numpy(df[predict_attr].to_numpy()).long(),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        max_per_class=max_per_class,
        seed=seed,
    )
    return _finalize(df, predict_attr, sens_attr, edge_path, drop_cols, standardize, masks)
