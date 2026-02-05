import torch
import numpy as np

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

