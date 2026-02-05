from __future__ import annotations

import argparse
import uuid
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

import torch

from csf_gnn.datasets import load_bail, load_credit, load_german, load_income, load_nba
from csf_gnn.logging import append_row, ensure_csv
from csf_gnn.train import baseline_metrics_logreg, baseline_logreg, split_metrics, train_csfgnn
from csf_gnn.bias_edges import detect_bias_prone_edges
from csf_gnn.utils import dataset_stats, print_header, set_seed, device


HEADER: List[str] = [
    "run_id","timestamp","dataset","seed","split_seed",
    "n_nodes","n_edges","n_features","layers","hidden","epochs",
    "optimizer","lr","weight_decay","lambda_fair","fairness_warmup",
    "drop_prob","downweight","hub_q","recompute_biasy_each_epoch",
    "class_weighting",
    "baseline_acc","baseline_auc","baseline_spd","baseline_eod",
    "train_acc","train_auc","train_spd","train_eod",
    "val_acc","val_auc","val_spd","val_eod",
    "test_acc","test_auc","test_spd","test_eod",
    "notes"
]

LOADERS = {
    "german": load_german,
    "nba": load_nba,
    "bail": load_bail,
    "credit": load_credit,
    "income": load_income,
}

DEFAULT_GRID = {
    "german": dict(lrs=[0.005], drop_probs=[0.1,0.2,0.3], downweights=[0.3,0.4,0.5],
                  lambda_fairs=[0.5,0.6,0.7,0.8], hub_qs=[0.80,0.90], hidden=256, epochs=500),
    "nba":    dict(lrs=[0.005], drop_probs=[0.1,0.2,0.3], downweights=[0.3,0.4,0.5],
                  lambda_fairs=[0.5,0.6,0.7,0.8], hub_qs=[0.80,0.90], hidden=256, epochs=500),
    "bail":   dict(lrs=[0.005], drop_probs=[0.1,0.2,0.3], downweights=[0.3,0.4,0.5],
                  lambda_fairs=[0.5,0.6,0.7,0.8], hub_qs=[0.80,0.90], hidden=256, epochs=500),
    "credit": dict(lrs=[0.0001], drop_probs=[0.1,0.2,0.3], downweights=[0.3,0.4,0.5],
                  lambda_fairs=[0.5,0.6,0.7,0.8], hub_qs=[0.80,0.90], hidden=256, epochs=200),
    "income": dict(lrs=[0.005], drop_probs=[0.1,0.2,0.3], downweights=[0.3,0.4,0.5],
                  lambda_fairs=[0.5,0.6,0.7,0.8], hub_qs=[0.80,0.90], hidden=256, epochs=500),
}


def run(dataset: str, log_path: str, seed: int = 42, progress: bool = True) -> None:
    if dataset not in LOADERS:
        raise ValueError(f"Unknown dataset={dataset}. Choose from {list(LOADERS)}")

    set_seed(seed)
    ensure_csv(log_path, HEADER)

    # Load once
    loaded = LOADERS[dataset]()
    edge_index, X, y, s, masks = loaded.edge_index, loaded.x, loaded.y, loaded.s, loaded.masks

    # Optional baseline print (features-only)
    _ = baseline_logreg(X, y, masks)

    ds_stats = dataset_stats(edge_index, X)
    base = baseline_metrics_logreg(X, y, s, masks)

    grid = DEFAULT_GRID[dataset]
    weight_decay = 1e-4
    fairness_warmup = 100
    recompute_biasy_each_epoch = False
    class_weighting = "balanced"
    layers = 2

    combos = list(product(grid["lrs"], grid["drop_probs"], grid["downweights"], grid["lambda_fairs"], grid["hub_qs"]))
    iterator = tqdm(combos, desc=f"{dataset} grid", total=len(combos)) if progress else combos

    total_runs = 0
    for lr, drop_prob, downweight, lambda_fair, hub_q in iterator:
        run_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print_header(
            f"RUN {run_id} | dataset={dataset} lr={lr} drop={drop_prob} down={downweight} "
            f"lambda_fair={lambda_fair} hub_q={hub_q}"
        )

        model, _, _ = train_csfgnn(
            X, edge_index, y, s, masks,
            hidden=grid["hidden"], epochs=grid["epochs"], lr=lr,
            lambda_fair=lambda_fair, drop_prob=drop_prob,
            downweight=downweight, hub_q=hub_q,
            fairness_warmup=fairness_warmup,
            recompute_biasy_each_epoch=recompute_biasy_each_epoch,
            weight_decay=weight_decay, verbose_every=10
        )

        # Metrics on splits
        with torch.no_grad():
            dev = device()
            Xd, Ei, yd, sd = X.to(dev), edge_index.to(dev), y.to(dev), s.to(dev)
            edge_masks = detect_bias_prone_edges(Ei.cpu(), sd.cpu(), Xd.size(0), hub_q=hub_q)
            biasy_mask = edge_masks["biasy"].to(dev)
            bm = split_metrics(model, Xd, Ei, sd, yd, masks, biasy_mask)

        row = {
            "run_id": run_id,
            "timestamp": timestamp,
            "dataset": dataset,
            "seed": seed,
            "split_seed": seed,
            "n_nodes": ds_stats["n_nodes"],
            "n_edges": ds_stats["n_edges"],
            "n_features": ds_stats["n_features"],
            "layers": layers,
            "hidden": grid["hidden"],
            "epochs": grid["epochs"],
            "optimizer": "Adam",
            "lr": lr,
            "weight_decay": weight_decay,
            "lambda_fair": lambda_fair,
            "fairness_warmup": fairness_warmup,
            "drop_prob": drop_prob,
            "downweight": downweight,
            "hub_q": hub_q,
            "recompute_biasy_each_epoch": int(recompute_biasy_each_epoch),
            "class_weighting": class_weighting,
            "baseline_acc": base["acc"],
            "baseline_auc": base["auc"],
            "baseline_spd": base["spd"],
            "baseline_eod": base["eod"],
            "train_acc": bm["train"]["acc"],
            "train_auc": bm["train"]["auc"],
            "train_spd": bm["train"]["spd"],
            "train_eod": bm["train"]["eod"],
            "val_acc": bm["val"]["acc"],
            "val_auc": bm["val"]["auc"],
            "val_spd": bm["val"]["spd"],
            "val_eod": bm["val"]["eod"],
            "test_acc": bm["test"]["acc"],
            "test_auc": bm["test"]["auc"],
            "test_spd": bm["test"]["spd"],
            "test_eod": bm["test"]["eod"],
            "notes": ""
        }
        append_row(log_path, row, HEADER)
        total_runs += 1

    print_header(f"ALL DONE — logged {total_runs} runs to {log_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(LOADERS.keys()))
    ap.add_argument("--log", default=None, help="CSV log path. Default: experiments_<dataset>.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    log_path = args.log or f"experiments_{args.dataset}.csv"
    run(args.dataset, log_path=log_path, seed=args.seed, progress=(not args.no_progress))


if __name__ == "__main__":
    main()
