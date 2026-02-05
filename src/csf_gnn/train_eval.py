import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .models import CSFGNN
from .bias_utils import detect_bias_prone_edges
from .metrics import loss_with_fairness, metrics
from .logger import print_header

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
            with torch.no_grad():
                trm = metrics(logits[train_mask], y[train_mask], sensitive[train_mask])
                valm = metrics(logits[val_mask],   y[val_mask],   sensitive[val_mask])
            print(f"[{ep:03d}] loss={loss.item():.4f} "
                  f"train(acc={trm['acc']:.3f}, auc={trm['auc']:.3f}, spd={trm['spd']:.3f}, eod={trm['eod']:.3f}) "
                  f"val(acc={valm['acc']:.3f}, auc={valm['auc']:.3f}, spd={valm['spd']:.3f}, eod={valm['eod']:.3f})")

    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index, biasy_mask)
        test = metrics(logits[test_mask], y[test_mask], sensitive[test_mask])
    return model, test, masks_edges

# =========================================================
# Experiment logging helpers
# =========================================================

