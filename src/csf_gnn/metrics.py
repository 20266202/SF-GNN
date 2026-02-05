import numpy as np
import torch

    def tpr(mask):
        denom = (y_pos & mask).sum()
        return (pred_pos & y_pos & mask).sum() / denom if denom > 0 else np.nan
    eod = abs(tpr(s0) - tpr(s1)) if np.isfinite(tpr(s0)) and np.isfinite(tpr(s1)) else np.nan
    return {"acc": acc, "auc": auc, "spd": spd, "eod": eod}

# =========================================================
# Baseline (features-only) sanity check
# =========================================================


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

