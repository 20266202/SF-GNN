import os
import time
import random
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from .logger import print_header


def to_long_bool(x):
    if isinstance(x, torch.Tensor):
        return x.long(), x.bool()
    return torch.tensor(x, dtype=torch.long), torch.tensor(x, dtype=torch.bool)



def undirected_unique_edge_count(edge_index, num_nodes):
    u = torch.minimum(edge_index[0], edge_index[1])
    v = torch.maximum(edge_index[0], edge_index[1])
    uv = torch.stack([u, v], dim=0)
    uv = coalesce(uv, None, num_nodes, num_nodes)[0]
    return uv.size(1)

# =========================================================
# Loaders
# =========================================================



def read_and_validate_edges(edge_file, n, make_undirected=True, drop_self_loops=True):
    edges = np.genfromtxt(edge_file).astype(int)
    if edges.ndim == 1 and edges.size == 2:
        edges = edges.reshape(1, 2)
    mask = (edges[:, 0] >= 0) & (edges[:, 0] < n) & (edges[:, 1] >= 0) & (edges[:, 1] < n)
    if (~mask).any():
        print(f"Edge bounds violations removed: {np.sum(~mask)}")
    edges = edges[mask]
    edge_index = torch.tensor(edges.T, dtype=torch.long)
    if drop_self_loops:
        edge_index, _ = remove_self_loops(edge_index)
    if make_undirected:
        edge_index = to_undirected(edge_index, num_nodes=n)
    edge_index = coalesce(edge_index, None, n, n)[0]
    return edge_index


def symmetrize_adj(adj_coo):
    A = adj_coo.tocoo()
    AT = A.transpose().tocoo()
    A_sym = A + AT.multiply(AT > A) - A.multiply(AT > A)
    return A_sym.tocoo()



def stratified_masks(y, train_ratio=0.5, val_ratio=0.25, seed=42):
    y_np = y.cpu().numpy()
    N = len(y_np)
    tv_ratio = train_ratio + val_ratio
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=1 - tv_ratio, random_state=seed)
    (tv_idx, te_idx) = next(sss1.split(np.zeros(N), y_np))
    y_tv = y_np[tv_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio / tv_ratio, random_state=seed)
    (tr_rel, va_rel) = next(sss2.split(np.zeros(len(tv_idx)), y_tv))
    tr_idx = tv_idx[tr_rel]; va_idx = tv_idx[va_rel]
    train_mask = torch.zeros(N, dtype=torch.bool); train_mask[tr_idx] = True
    val_mask   = torch.zeros(N, dtype=torch.bool); val_mask[va_idx]   = True
    test_mask  = torch.zeros(N, dtype=torch.bool); test_mask[te_idx]  = True
    return {"train": train_mask, "val": val_mask, "test": test_mask}


def dataset_stats(edge_index, features):
    n_nodes = features.size(0)
    n_features = features.size(1)
    n_edges_undirected = undirected_unique_edge_count(edge_index, n_nodes)
    n_edges_directed = edge_index.size(1)
    return {"n_nodes": int(n_nodes), "n_edges": int(n_edges_undirected), "n_features": int(n_features),
            "n_edges_directed": int(n_edges_directed)}

@torch.no_grad()



def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # deterministic-ish
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
