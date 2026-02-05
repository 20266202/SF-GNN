import os
import sys
import uuid
from datetime import datetime
import argparse

# Make imports work when running "python scripts/xxx.py" from repo root.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from csf_gnn.datasets import load_dataset
from csf_gnn.train_eval import train_csfgnn, baseline_metrics_logreg
from csf_gnn.logger import append_row, print_header


DEFAULTS = {
    "german": dict(lrs=[0.005], drop_probs=[0.1,0.2,0.3], downweights=[0.3,0.4,0.5], lambda_fairs=[0.5,0.6,0.7,0.8], hub_qs=[0.80,0.90],
                  hidden=256, epochs=500, weight_decay=1e-4, fairness_warmup=100, recompute_biasy_each_epoch=False, class_weighting="balanced", layers=2),
    "nba":    dict(lrs=[0.005], drop_probs=[0.1,0.2,0.3], downweights=[0.3,0.4,0.5], lambda_fairs=[0.5,0.6,0.7,0.8], hub_qs=[0.80,0.90],
                  hidden=256, epochs=500, weight_decay=1e-4, fairness_warmup=100, recompute_biasy_each_epoch=False, class_weighting="balanced", layers=2),
    "bail":   dict(lrs=[0.005], drop_probs=[0.1,0.2,0.3], downweights=[0.3,0.4,0.5], lambda_fairs=[0.5,0.6,0.7,0.8], hub_qs=[0.80,0.90],
                  hidden=256, epochs=500, weight_decay=1e-4, fairness_warmup=100, recompute_biasy_each_epoch=False, class_weighting="balanced", layers=2),
    "income": dict(lrs=[0.005], drop_probs=[0.1,0.2,0.3], downweights=[0.3,0.4,0.5], lambda_fairs=[0.5,0.6,0.7,0.8], hub_qs=[0.80,0.90],
                  hidden=256, epochs=500, weight_decay=1e-4, fairness_warmup=100, recompute_biasy_each_epoch=False, class_weighting="balanced", layers=2),
    "credit": dict(lrs=[0.0001], drop_probs=[0.1,0.2,0.3], downweights=[0.3,0.4,0.5], lambda_fairs=[0.5,0.6,0.7,0.8], hub_qs=[0.80,0.90],
                  hidden=256, epochs=200, weight_decay=1e-4, fairness_warmup=200, recompute_biasy_each_epoch=False, class_weighting="balanced", layers=2),
}

CSV_HEADERS = [
    "run_id","timestamp","dataset",
    "lr","drop_prob","downweight","lambda_fair","hub_q",
    "hidden","epochs","weight_decay","fairness_warmup","recompute_biasy_each_epoch","class_weighting","layers",
    "baseline_acc","baseline_auc","baseline_spd","baseline_eod",
    "train_acc","train_auc","train_spd","train_eod",
    "val_acc","val_auc","val_spd","val_eod",
    "test_acc","test_auc","test_spd","test_eod",
    "notes"
]


def run(dataset: str, data_dir: str | None = None, log_path: str | None = None) -> str:
    dataset = dataset.lower().strip()
    if dataset not in DEFAULTS:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from {sorted(DEFAULTS)}")

    if log_path is None:
        log_path = f"experiments_{dataset}.csv"

    cfg = DEFAULTS[dataset]

    x, edge_index, y, sensitive, masks = load_dataset(dataset, data_dir=data_dir)
    base = baseline_metrics_logreg(x, y, sensitive, masks)

    total_runs = 0
    for lr in cfg["lrs"]:
        for drop_prob in cfg["drop_probs"]:
            for downweight in cfg["downweights"]:
                for lambda_fair in cfg["lambda_fairs"]:
                    for hub_q in cfg["hub_qs"]:
                        run_id = str(uuid.uuid4())
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print_header(
                            f"RUN {run_id} | {dataset} | lr={lr} drop={drop_prob} down={downweight} "
                            f"lambda={lambda_fair} hub_q={hub_q}"
                        )

                        bm = train_csfgnn(
                            x, edge_index, y, sensitive, masks,
                            hidden=cfg["hidden"], epochs=cfg["epochs"], lr=lr, lambda_fair=lambda_fair,
                            drop_prob=drop_prob, downweight=downweight, hub_q=hub_q,
                            fairness_warmup=cfg["fairness_warmup"],
                            recompute_biasy_each_epoch=cfg["recompute_biasy_each_epoch"],
                            weight_decay=cfg["weight_decay"],
                            class_weighting=cfg["class_weighting"],
                            layers=cfg["layers"],
                        )

                        row = {
                            "run_id": run_id,
                            "timestamp": timestamp,
                            "dataset": dataset,
                            "lr": lr,
                            "drop_prob": drop_prob,
                            "downweight": downweight,
                            "lambda_fair": lambda_fair,
                            "hub_q": hub_q,
                            "hidden": cfg["hidden"],
                            "epochs": cfg["epochs"],
                            "weight_decay": cfg["weight_decay"],
                            "fairness_warmup": cfg["fairness_warmup"],
                            "recompute_biasy_each_epoch": cfg["recompute_biasy_each_epoch"],
                            "class_weighting": cfg["class_weighting"],
                            "layers": cfg["layers"],
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
                            "notes": "",
                        }
                        append_row(log_path, row, CSV_HEADERS)
                        total_runs += 1

    print_header(f"ALL DONE — logged {total_runs} runs to {log_path}")
    return log_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DEFAULTS.keys()))
    ap.add_argument("--data_dir", default=None, help="Optional directory containing csv + edge files")
    ap.add_argument("--log", default=None, help="CSV log path (default: experiments_<dataset>.csv)")
    args = ap.parse_args()
    run(args.dataset, data_dir=args.data_dir, log_path=args.log)

if __name__ == "__main__":
    main()
