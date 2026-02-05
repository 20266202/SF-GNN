from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import coalesce, remove_self_loops

class CSFConv(nn.Module):
    def __init__(self, in_dim, out_dim, aggr='mean', drop_prob=0.3, downweight=0.5):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.bn  = nn.BatchNorm1d(out_dim)
        self.do  = nn.Dropout(p=0.2)
        self.aggr = aggr
        self.drop_prob = drop_prob
        self.downweight = downweight
        self.res_proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else None

    def _aggregate(self, edge_index, x, edge_weight):
        N, _ = x.size()
        x_j = x[edge_index[1]]
        msgs = edge_weight.unsqueeze(-1) * x_j
        out = torch.zeros_like(x)
        out.index_add_(0, edge_index[0], msgs)
        if self.aggr == 'mean':
            deg = torch.bincount(edge_index[0], minlength=N).clamp(min=1).float().to(x.device)
            out = out / deg.unsqueeze(-1)
        return out

    def forward(self, x, edge_index, biasy_mask):
        h_in = x
        x = self.lin(x)
        device = x.device
        M = edge_index.size(1)
        ew = torch.ones(M, device=device)
        if biasy_mask.any():
            drops = (torch.rand(M, device=device) < self.drop_prob) & biasy_mask
            keep  = ~drops
            down  = biasy_mask & keep
            ew[down] = self.downweight
            edge_index = edge_index[:, keep]
            ew = ew[keep]
        h = self._aggregate(edge_index, x, ew)
        h = h + (self.res_proj(h_in) if self.res_proj is not None else h_in)
        h = self.bn(h)
        return self.do(F.relu(h))



class CSFGNN(nn.Module):
    def __init__(self, in_dim, hid=128, out_dim=2, drop_prob=0.1, downweight=0.3):
        super().__init__()
        self.conv1 = CSFConv(in_dim, hid, drop_prob=drop_prob, downweight=downweight)
        self.conv2 = CSFConv(hid, hid, drop_prob=drop_prob, downweight=downweight)
        self.cls   = nn.Sequential(
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hid, out_dim)
        )

    def forward(self, x, edge_index, biasy_mask):
        x = self.conv1(x, edge_index, biasy_mask)
        x = self.conv2(x, edge_index, biasy_mask)
        return self.cls(x)

# =========================================================
# Fairness loss + metrics
# =========================================================
