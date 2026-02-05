"""
Training and evaluation.

This keeps the logic as close as possible to the original notebook, but
wrapped into reusable functions.
"""
from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from .bias_edges import detect_bias_prone_edges
from .models import CSFGNN
from .utils import device, print_header


def fairness_penalty_stat_parity(logits, sensitive, temp=1.0):
    if logits.size(-1) == 2:
        p1 = F.softmax(logits / temp, dim=-1)[:, 1]
    else:
        p1 = torch.sigmoid(logits.view(-1))
    s0 = sensitive == 0
    s1 = sensitive == 1
    if s0.sum() == 0 or s1.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    return torch.abs(p1[s0].mean() - p1[s1].mean())



def loss_with_fairness(logits, y, sensitive, lambda_fair=0.5, class_weight=None):
    ce = F.cross_entropy(logits, y, weight=class_weight)
    sp = fairness_penalty_stat_parity(logits, sensitive)
    return ce + lambda_fair * sp, {"ce": ce.item(), "sp": sp.item()}

@torch.no_grad()


def metrics(logits, y, sensitive):
    pred = logits.argmax(dim=-1)
    acc = (pred == y).float().mean().item()
    if logits.size(-1) == 2:
        p1 = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    else:
        p1 = torch.sigmoid(logits.view(-1)).cpu().numpy()
    y_np = y.cpu().numpy()
    s_np = sensitive.cpu().numpy()
    try:
        auc = roc_auc_score(y_np, p1)
    except ValueError:
        auc = float("nan")
    s0 = s_np == 0; s1 = s_np == 1
    p1_mean_s0 = p1[s0].mean() if s0.any() else np.nan
    p1_mean_s1 = p1[s1].mean() if s1.any() else np.nan
    spd = abs(p1_mean_s0 - p1_mean_s1)
    pred_pos = (pred.cpu().numpy() == 1); y_pos = (y_np == 1)
    def tpr(mask):
        denom = (y_pos & mask).sum()
        return (pred_pos & y_pos & mask).sum() / denom if denom > 0 else np.nan
    eod = abs(tpr(s0) - tpr(s1)) if np.isfinite(tpr(s0)) and np.isfinite(tpr(s1)) else np.nan
    return {"acc": acc, "auc": auc, "spd": spd, "eod": eod}

# =========================================================
# Baseline (features-only) sanity check
# =========================================================


def baseline_logreg(features, labels, masks):
    print_header("Baseline: Logistic Regression (features-only)")
    X = features.cpu().numpy(); y = labels.cpu().numpy()
    tr = masks["train"].cpu().numpy(); va = masks["val"].cpu().numpy(); te = masks["test"].cpu().numpy()
    # global s is expected in original script; but we won't rely on global here

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X[tr], y[tr])

    for name, msk in [("Train", tr), ("Val", va), ("Test", te)]:
        prob = clf.predict_proba(X[msk])[:, 1]
        pred = (prob >= 0.5).astype(int)
        acc = accuracy_score(y[msk], pred)
        try:
            auc = roc_auc_score(y[msk], prob)
        except ValueError:
            auc = float("nan")
        # placeholders: fairness needs sensitive externally
        print(f"{name}: acc={acc:.3f} auc={auc:.3f} (SPD/EOD need sensitive)")

    return clf

# =========================================================
# Train loop
# =========================================================


def baseline_metrics_logreg(features, labels, sensitive, masks):
    X = features.cpu().numpy()
    y = labels.cpu().numpy()
    s_np = sensitive.cpu().numpy()
    tr = masks["train"].cpu().numpy()
    te = masks["test"].cpu().numpy()

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X[tr], y[tr])

    prob = clf.predict_proba(X[te])[:, 1]
    pred = (prob >= 0.5).astype(int)
    acc = accuracy_score(y[te], pred)
    try:
        auc = roc_auc_score(y[te], prob)
    except ValueError:
        auc = float("nan")

    s0, s1 = (s_np[te] == 0), (s_np[te] == 1)
    p1_s0 = prob[s0].mean() if s0.any() else float("nan")
    p1_s1 = prob[s1].mean() if s1.any() else float("nan")
    spd = abs(p1_s0 - p1_s1)

    y_pos = (y[te] == 1)
    def tpr(mask):
        denom = (y_pos & mask).sum()
        return (pred & y_pos & mask).sum() / denom if denom > 0 else float("nan")
    eod = abs(tpr(s0) - tpr(s1)) if np.isfinite(tpr(s0)) and np.isfinite(tpr(s1)) else float("nan")

    return {"acc": acc, "auc": auc, "spd": spd, "eod": eod}



def split_metrics(model, x, edge_index, sensitive, y, masks, biasy_mask):
    device = next(model.parameters()).device
    model.eval()
    logits = model(x.to(device), edge_index.to(device), biasy_mask.to(device))
    out = {}
    for split_name in ["train", "val", "test"]:
        msk = masks[split_name].to(device)
        mm = metrics(logits[msk], y[msk].to(device), sensitive[msk].to(device))
        out[split_name] = mm
    return out



def train_csfgnn(
    x, edge_index, y, sensitive, masks,
    hidden=256, epochs=300, lr=0.01, lambda_fair=0.6,
    drop_prob=0.1, downweight=0.3, hub_q=0.95,
    fairness_warmup=100, recompute_biasy_each_epoch=False,
    weight_decay=1e-4, verbose_every=10
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x, edge_index, y, sensitive = x.to(device), edge_index.to(device), y.to(device), sensitive.to(device)
    train_mask = masks["train"].to(device); val_mask = masks["val"].to(device); test_mask = masks["test"].to(device)

    pos = (y == 1).sum().item(); neg = (y == 0).sum().item()
    cw = torch.tensor([pos/(pos+neg), neg/(pos+neg)], device=device, dtype=torch.float32)

    masks_edges = detect_bias_prone_edges(edge_index.cpu(), sensitive.cpu(), x.size(0), hub_q=hub_q)
    biasy_mask = masks_edges["biasy"].to(device)

    model = CSFGNN(x.size(1), hid=hidden, out_dim=2, drop_prob=drop_prob, downweight=downweight).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for ep in range(1, epochs + 1):
        print(f"Epoch {ep}/{epochs}", end="\r", file=sys.stdout)
        if recompute_biasy_each_epoch:
            masks_edges = detect_bias_prone_edges(edge_index.cpu(), sensitive.cpu(), x.size(0), hub_q=hub_q)
            biasy_mask = masks_edges["biasy"].to(device)

        model.train()
        opt.zero_grad()
        logits = model(x, edge_index, biasy_mask)

        lam = (min(1.0, ep / float(fairness_warmup)) * lambda_fair) if fairness_warmup > 0 else lambda_fair
        loss, parts = loss_with_fairness(logits[train_mask], y[train_mask], sensitive[train_mask], lambda_fair=lam, class_weight=cw)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step()

        if verbose_every and (ep % verbose_every == 0 or ep == 1):
            model.eval()
            with torc
