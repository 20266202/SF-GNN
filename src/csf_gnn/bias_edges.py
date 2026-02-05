from __future__ import annotations

def detect_bias_prone_edges(edge_index, sensitive, num_nodes, hub_q=0.90):
    m = edge_index.size(1)
    s = sensitive
    homophilic = (s[edge_index[0]] == s[edge_index[1]])

    deg0 = degree(edge_index[0], num_nodes=num_nodes)
    deg  = deg0.clone()
    thresh = torch.quantile(deg.float(), torch.tensor(hub_q, device=deg.device))
    hub_node = deg >= thresh
    hubs = hub_node[edge_index[0]] | hub_node[edge_index[1]]

    adj = [[] for _ in range(num_nodes)]
    e_np = edge_index.detach().cpu().numpy()
    for u, v in zip(e_np[0], e_np[1]):
        adj[u].append(v); adj[v].append(u)
    adj_sets = [set(nbrs) for nbrs in adj]
    tri = torch.zeros(m, dtype=torch.bool)
    for k in range(m):
        u = int(edge_index[0, k]); v = int(edge_index[1, k])
        a, b = (adj_sets[u], adj_sets[v]) if len(adj_sets[u]) < len(adj_sets[v]) else (adj_sets[v], adj_sets[u])
        if not a:
            continue
        for w in a:
            if w in b:
                tri[k] = True
                break

    biasy = homophilic & (tri | hubs)
    return {"homophilic": homophilic, "triangles": tri, "hubs": hubs, "biasy": biasy}

# =========================================================
# CSF-GNN
# =========================================================
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

