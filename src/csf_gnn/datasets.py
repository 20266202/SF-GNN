import os
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from .logger import print_header
from .utils import read_and_validate_edges, symmetrize_adj, stratified_masks, dataset_stats


def load_german_dataset(
    csv_path="german.csv",
    edge_path="german_edges.txt",
    standardize=True,
    drop_cols=("GoodCustomer", "OtherLoansAtStore", "PurposeOfLoan"),
):
    if not os.path.exists(csv_path): raise FileNotFoundError(f"Missing file: {csv_path}")
    if not os.path.exists(edge_path): raise FileNotFoundError(f"Missing file: {edge_path}")

    print_header("Loading german dataset")
    t0 = time.time()
    df = pd.read_csv(csv_path)

    predict_attr = "GoodCustomer"
    sens_attr = "Gender"

    unique_lbls = pd.unique(df[predict_attr])
    print("Unique labels in CSV:", unique_lbls)
    labels = df[predict_attr].to_numpy(copy=True)
    labels = np.where(labels == -1, 0, labels).astype(int)
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Labels must map to {0,1} after conversion.")

    mapping = {"Female": 1, "Male": 0, "female": 1, "male": 0, "F": 1, "M": 0}
    df[sens_attr] = df[sens_attr].astype(str).map(mapping)
    df[sens_attr] = pd.to_numeric(df[sens_attr], errors="coerce").fillna(0).astype(int)
    sens = df[sens_attr].values.astype(int)

    keep_cols = [c for c in df.columns if c not in drop_cols]
    X_df = df[keep_cols].copy()
    cat_cols = [c for c in X_df.columns if X_df[c].dtype == "object"]
    if cat_cols:
        X_df = pd.get_dummies(X_df, columns=cat_cols, drop_first=True)

    X_np = X_df.to_numpy(dtype=np.float32)
    if standardize:
        scaler = StandardScaler()
        X_np = scaler.fit_transform(X_np).astype(np.float32)

    features = torch.tensor(X_np, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    sens_t   = torch.tensor(sens, dtype=torch.long)

    n = features.size(0)
    edge_index = read_and_validate_edges(edge_path, n, make_undirected=True, drop_self_loops=True)

    print("Num nodes:", n, " Num features:", features.size(1), " Num edges:", edge_index.size(1))
    deg = torch.bincount(edge_index[0], minlength=n).cpu().numpy()
    print("Degree: min/med/max:", int(deg.min()), float(np.median(deg)), int(deg.max()))
    print("Isolated nodes:", int((deg == 0).sum()))
    u_lbl, c_lbl = np.unique(labels, return_counts=True)
    print("Label distribution:", dict(zip(u_lbl.tolist(), c_lbl.tolist())))
    u_s, c_s = np.unique(sens, return_counts=True)
    print("Sensitive distribution (0=Male,1=Female):", dict(zip(u_s.tolist(), c_s.tolist())))

    masks = stratified_masks(labels_t, train_ratio=0.5, val_ratio=0.25, seed=TORCH_SEED)
    overlap = {
        "train∩val": int((masks["train"] & masks["val"]).sum()),
        "train∩test": int((masks["train"] & masks["test"]).sum()),
        "val∩test": int((masks["val"] & masks["test"]).sum()),
    }
    print("Mask overlaps (should be 0s):", overlap)

    print(f"Done in {time.time()-t0:.3f}s")

    num_nodes = features.size(0)
    m_dir = edge_index.size(1)
    m_undir = undirected_unique_edge_count(edge_index, num_nodes)
    print(f"Edges (directed columns)       : {m_dir}")
    print(f"Edges (undirected unique pairs): {m_undir}")
    return edge_index, features, labels_t, sens_t, masks




def load_nba_dataset(
    csv_path="nba.csv",
    edge_path="nba_edges.txt",
    standardize=True,
    drop_cols=["SALARY"],
):
    if not os.path.exists(csv_path): raise FileNotFoundError(f"Missing file: {csv_path}")
    if not os.path.exists(edge_path): raise FileNotFoundError(f"Missing file: {edge_path}")

    print_header("Loading nba dataset")
    t0 = time.time()
    df = pd.read_csv(csv_path)

    predict_attr = "SALARY"
    sens_attr = "country"

    labels = df[predict_attr].to_numpy(copy=True)
    valid_mask = labels >= 0
    df = df[valid_mask].reset_index(drop=True)
    labels = labels[valid_mask]
    print(f"Removed {-1} labels: remaining nodes = {len(labels)}")

    unique_lbls = pd.unique(df[predict_attr])
    print("Unique labels in CSV:", unique_lbls)
    labels = df[predict_attr].to_numpy(copy=True)
    labels[labels > 1] = 1
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Labels must map to {0,1} after conversion.")

    df[sens_attr] = pd.to_numeric(df[sens_attr], errors="coerce").fillna(0).astype(int)
    sens = df[sens_attr].values.astype(int)

    keep_cols = [c for c in df.columns if c not in drop_cols]
    X_df = df[keep_cols].copy()
    cat_cols = [c for c in X_df.columns if X_df[c].dtype == "object"]
    if cat_cols:
        X_df = pd.get_dummies(X_df, columns=cat_cols, drop_first=True)

    X_np = X_df.to_numpy(dtype=np.float32)
    if standardize:
        scaler = StandardScaler()
        X_np = scaler.fit_transform(X_np).astype(np.float32)

    features = torch.tensor(X_np, dtype=torch.float32)

    old_ids = df["user_id"].values
    new_index_map = dict(zip(old_ids, range(len(df))))

    edges_raw = np.genfromtxt(edge_path).astype(int)
    edges_filtered = []
    for u, v in edges_raw:
        if u in new_index_map and v in new_index_map:
            edges_filtered.append([new_index_map[u], new_index_map[v]])
    edges_filtered = np.array(edges_filtered)

    edge_index = torch.tensor(edges_filtered.T, dtype=torch.long)
    edge_index = to_undirected(edge_index, num_nodes=len(df))
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = coalesce(edge_index, None, len(df), len(df))[0]

    labels_t = torch.tensor(labels, dtype=torch.long)
    sens_t   = torch.tensor(sens, dtype=torch.long)

    n = features.size(0)
    print("Num nodes:", n, " Num features:", features.size(1), " Num edges:", edge_index.size(1))
    deg = torch.bincount(edge_index[0], minlength=n).cpu().numpy()
    print("Degree: min/med/max:", int(deg.min()), float(np.median(deg)), int(deg.max()))
    print("Isolated nodes:", int((deg == 0).sum()))
    u_lbl, c_lbl = np.unique(labels, return_counts=True)
    print("Label distribution:", dict(zip(u_lbl.tolist(), c_lbl.tolist())))
    u_s, c_s = np.unique(sens, return_counts=True)
    print("Sensitive distribution (country):", dict(zip(u_s.tolist(), c_s.tolist())))

    masks = nba_masks(labels_t, train_ratio=0.5, val_ratio=0.25,
                  max_per_class=50, seed=TORCH_SEED)

    overlap = {
        "train∩val": int((masks["train"] & masks["val"]).sum()),
        "train∩test": int((masks["train"] & masks["test"]).sum()),
        "val∩test": int((masks["val"] & masks["test"]).sum()),
    }
    print("Mask overlaps (should be 0s):", overlap)

    print(f"Done in {time.time()-t0:.3f}s")

    num_nodes = features.size(0)
    m_dir = edge_index.size(1)
    m_undir = undirected_unique_edge_count(edge_index, num_nodes)
    print(f"Edges (directed columns)       : {m_dir}")
    print(f"Edges (undirected unique pairs): {m_undir}")
    return edge_index, features, labels_t, sens_t, masks

# =========================================================
# Bias-prone edge detector
# =========================================================
@torch.no_grad()




def load_bail_dataset(
    csv_path="bail.csv",
    edge_path="bail_edges.txt",
    standardize=True,
    drop_cols=["RECID"],
):
    if not os.path.exists(csv_path): raise FileNotFoundError(f"Missing file: {csv_path}")
    if not os.path.exists(edge_path): raise FileNotFoundError(f"Missing file: {edge_path}")

    print_header("Loading bail dataset")
    t0 = time.time()
    df = pd.read_csv(csv_path)

    predict_attr = "RECID"
    sens_attr = "WHITE"

    labels = df[predict_attr].to_numpy(copy=True)

    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Labels must map to {0,1} after conversion.")

    df[sens_attr] = pd.to_numeric(df[sens_attr], errors="coerce").fillna(0).astype(int)
    sens = df[sens_attr].values.astype(int)

    keep_cols = [c for c in df.columns if c not in drop_cols]
    X_df = df[keep_cols].copy()
    cat_cols = [c for c in X_df.columns if X_df[c].dtype == "object"]
    if cat_cols:
        X_df = pd.get_dummies(X_df, columns=cat_cols, drop_first=True)

    X_np = X_df.to_numpy(dtype=np.float32)
    if standardize:
        scaler = StandardScaler()
        X_np = scaler.fit_transform(X_np).astype(np.float32)

    features = torch.tensor(X_np, dtype=torch.float32)

    edges_raw = np.genfromtxt(edge_path).astype(int)


    if edges_raw.ndim == 1 and edges_raw.size == 2:
        edges_raw = edges_raw.reshape(1, 2)

    edge_index = torch.tensor(edges_raw.T, dtype=torch.long)
    edge_index = to_undirected(edge_index, num_nodes=len(df))
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = coalesce(edge_index, None, len(df), len(df))[0]

    labels_t = torch.tensor(labels, dtype=torch.long)
    sens_t   = torch.tensor(sens, dtype=torch.long)

    n = features.size(0)
    print("Num nodes:", n, " Num features:", features.size(1), " Num edges:", edge_index.size(1))
    deg = torch.bincount(edge_index[0], minlength=n).cpu().numpy()
    print("Degree: min/med/max:", int(deg.min()), float(np.median(deg)), int(deg.max()))
    print("Isolated nodes:", int((deg == 0).sum()))
    u_lbl, c_lbl = np.unique(labels, return_counts=True)
    print("Label distribution:", dict(zip(u_lbl.tolist(), c_lbl.tolist())))
    u_s, c_s = np.unique(sens, return_counts=True)
    print("Sensitive distribution (WHITE):", dict(zip(u_s.tolist(), c_s.tolist())))

    masks = bail_masks(labels_t, train_ratio=0.5, val_ratio=0.25,
                  max_per_class=100, seed=TORCH_SEED)

    overlap = {
        "train∩val": int((masks["train"] & masks["val"]).sum()),
        "train∩test": int((masks["train"] & masks["test"]).sum()),
        "val∩test": int((masks["val"] & masks["test"]).sum()),
    }
    print("Mask overlaps (should be 0s):", overlap)

    print(f"Done in {time.time()-t0:.3f}s")

    num_nodes = features.size(0)
    m_dir = edge_index.size(1)
    m_undir = undirected_unique_edge_count(edge_index, num_nodes)
    print(f"Edges (directed columns)       : {m_dir}")
    print(f"Edges (undirected unique pairs): {m_undir}")
    return edge_index, features, labels_t, sens_t, masks

# =========================================================
# Bias-prone edge detector
# =========================================================
@torch.no_grad()




def load_credit_dataset(
    csv_path="credit.csv",
    edge_path="credit_edges.txt",
    standardize=True,
    drop_cols=["NoDefaultNextMonth",'Single']
):
    if not os.path.exists(csv_path): raise FileNotFoundError(f"Missing file: {csv_path}")
    if not os.path.exists(edge_path): raise FileNotFoundError(f"Missing file: {edge_path}")

    print_header("Loading credit dataset")
    t0 = time.time()
    df = pd.read_csv(csv_path)

    predict_attr = "NoDefaultNextMonth"
    sens_attr = "Age"

    labels = df[predict_attr].to_numpy(copy=True)

    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Labels must map to {0,1} after conversion.")

    df[sens_attr] = pd.to_numeric(df[sens_attr], errors="coerce").fillna(0).astype(int)
    sens = df[sens_attr].values.astype(int)

    keep_cols = [c for c in df.columns if c not in drop_cols]
    X_df = df[keep_cols].copy()
    cat_cols = [c for c in X_df.columns if X_df[c].dtype == "object"]
    if cat_cols:
        X_df = pd.get_dummies(X_df, columns=cat_cols, drop_first=True)

    X_np = X_df.to_numpy(dtype=np.float32)
    if standardize:
        scaler = StandardScaler()
        X_np = scaler.fit_transform(X_np).astype(np.float32)

    features = torch.tensor(X_np, dtype=torch.float32)

    edges_raw = np.genfromtxt(edge_path).astype(int)


    if edges_raw.ndim == 1 and edges_raw.size == 2:
        edges_raw = edges_raw.reshape(1, 2)

    edge_index = torch.tensor(edges_raw.T, dtype=torch.long)
    edge_index = to_undirected(edge_index, num_nodes=len(df))
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = coalesce(edge_index, None, len(df), len(df))[0]

    labels_t = torch.tensor(labels, dtype=torch.long)
    sens_t   = torch.tensor(sens, dtype=torch.long)

    n = features.size(0)
    print("Num nodes:", n, " Num features:", features.size(1), " Num edges:", edge_index.size(1))
    deg = torch.bincount(edge_index[0], minlength=n).cpu().numpy()
    print("Degree: min/med/max:", int(deg.min()), float(np.median(deg)), int(deg.max()))
    print("Isolated nodes:", int((deg == 0).sum()))
    u_lbl, c_lbl = np.unique(labels, return_counts=True)
    print("Label distribution:", dict(zip(u_lbl.tolist(), c_lbl.tolist())))
    u_s, c_s = np.unique(sens, return_counts=True)
    print(f"Sensitive distribution ({sens_attr}):", dict(zip(u_s.tolist(), c_s.tolist())))

    masks = credit_masks(labels_t, train_ratio=0.5, val_ratio=0.25,
                  max_per_class=500, seed=TORCH_SEED)

    overlap = {
        "train∩val": int((masks["train"] & masks["val"]).sum()),
        "train∩test": int((masks["train"] & masks["test"]).sum()),
        "val∩test": int((masks["val"] & masks["test"]).sum()),
    }
    print("Mask overlaps (should be 0s):", overlap)

    print(f"Done in {time.time()-t0:.3f}s")

    num_nodes = features.size(0)
    m_dir = edge_index.size(1)
    m_undir = undirected_unique_edge_count(edge_index, num_nodes)
    print(f"Edges (directed columns)       : {m_dir}")
    print(f"Edges (undirected unique pairs): {m_undir}")
    return edge_index, features, labels_t, sens_t, masks

# =========================================================
# Bias-prone edge detector
# =========================================================
@torch.no_grad()




def load_income_dataset(
    csv_path="income.csv",
    edge_path="income_edges.txt",
    standardize=True,
    drop_cols=["income"]
):
    if not os.path.exists(csv_path): raise FileNotFoundError(f"Missing file: {csv_path}")
    if not os.path.exists(edge_path): raise FileNotFoundError(f"Missing file: {edge_path}")

    print_header("Loading income dataset")
    t0 = time.time()
    df = pd.read_csv(csv_path)

    predict_attr = "income"
    sens_attr = "race"

    labels = df[predict_attr].to_numpy(copy=True)

    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Labels must map to {0,1} after conversion.")

    df[sens_attr] = pd.to_numeric(df[sens_attr], errors="coerce").fillna(0).astype(int)
    sens = df[sens_attr].values.astype(int)

    keep_cols = [c for c in df.columns if c not in drop_cols]
    X_df = df[keep_cols].copy()
    cat_cols = [c for c in X_df.columns if X_df[c].dtype == "object"]
    if cat_cols:
        X_df = pd.get_dummies(X_df, columns=cat_cols, drop_first=True)

    X_np = X_df.to_numpy(dtype=np.float32)
    if standardize:
        scaler = StandardScaler()
        X_np = scaler.fit_transform(X_np).astype(np.float32)

    features = torch.tensor(X_np, dtype=torch.float32)

    edges_raw = np.genfromtxt(edge_path).astype(int)


    if edges_raw.ndim == 1 and edges_raw.size == 2:
        edges_raw = edges_raw.reshape(1, 2)

    edge_index = torch.tensor(edges_raw.T, dtype=torch.long)
    edge_index = to_undirected(edge_index, num_nodes=len(df))
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = coalesce(edge_index, None, len(df), len(df))[0]

    labels_t = torch.tensor(labels, dtype=torch.long)
    sens_t   = torch.tensor(sens, dtype=torch.long)

    n = features.size(0)
    print("Num nodes:", n, " Num features:", features.size(1), " Num edges:", edge_index.size(1))
    deg = torch.bincount(edge_index[0], minlength=n).cpu().numpy()
    print("Degree: min/med/max:", int(deg.min()), float(np.median(deg)), int(deg.max()))
    print("Isolated nodes:", int((deg == 0).sum()))
    u_lbl, c_lbl = np.unique(labels, return_counts=True)
    print("Label distribution:", dict(zip(u_lbl.tolist(), c_lbl.tolist())))
    u_s, c_s = np.unique(sens, return_counts=True)
    print(f"Sensitive distribution ({sens_attr}):", dict(zip(u_s.tolist(), c_s.tolist())))

    masks = income_masks(labels_t, train_ratio=0.5, val_ratio=0.25,
                  max_per_class=500, seed=TORCH_SEED)

    overlap = {
        "train∩val": int((masks["train"] & masks["val"]).sum()),
        "train∩test": int((masks["train"] & masks["test"]).sum()),
        "val∩test": int((masks["val"] & masks["test"]).sum()),
    }
    print("Mask overlaps (should be 0s):", overlap)

    print(f"Done in {time.time()-t0:.3f}s")

    num_nodes = features.size(0)
    m_dir = edge_index.size(1)
    m_undir = undirected_unique_edge_count(edge_index, num_nodes)
    print(f"Edges (directed columns)       : {m_dir}")
    print(f"Edges (undirected unique pairs): {m_undir}")
    return edge_index, features, labels_t, sens_t, masks

# =========================================================
# Bias-prone edge detector
# =========================================================
@torch.no_grad()


def load_dataset(name: str, data_dir: str | None = None):
    """Convenience wrapper to load one of: german, nba, bail, credit, income.

    If data_dir is provided, csv/edge paths are resolved relative to it.
    """
    name = name.lower().strip()
    if name not in {"german","nba","bail","credit","income"}:
        raise ValueError(f"Unknown dataset: {name}")
    if data_dir:
        def p(x): return os.path.join(data_dir, x)
    else:
        def p(x): return x

    if name == "german":
        return load_german_dataset(csv_path=p("german.csv"), edge_path=p("german_edges.txt"))
    if name == "nba":
        return load_nba_dataset(csv_path=p("nba.csv"), edge_path=p("nba_edges.txt"))
    if name == "bail":
        return load_bail_dataset(csv_path=p("bail.csv"), edge_path=p("bail_edges.txt"))
    if name == "credit":
        return load_credit_dataset(csv_path=p("credit.csv"), edge_path=p("credit_edges.txt"))
    return load_income_dataset(csv_path=p("income.csv"), edge_path=p("income_edges.txt"))
