"""MetapathModel — HAN-style heterogeneous attention network.

Architecture:
  1. Node-level attention (1-layer GAT per metapath graph):
       For each metapath, apply GAT on precomputed metapath adjacency.
       Drug/disease nodes get a representation h_mp from each metapath.
  2. Semantic attention:
       Learned weight vector q computes importance of each metapath.
       alpha_mp = softmax(q · tanh(W * h_mp))
       Final repr = weighted sum across metapaths.
  3. Link prediction:
       MLP([drug_repr || disease_repr])

Metapath adjacency matrices are precomputed once by graph_builder.py
and passed in at init via the metapath_graphs dict.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from .base import BaseModel


class _GATConvSparse(nn.Module):
    """1-hop GAT on a sparse (precomputed) adjacency.

    Parameters
    ----------
    in_dim  : input feature dimension
    out_dim : output feature dimension
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(out_dim * 2, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        """
        h          : (N, in_dim)
        edge_index : (2, E) — row=src, col=dst
        edge_weight: (E,) optional
        Returns    : (N, out_dim)
        """
        Wh = self.W(h)                         # (N, out_dim)
        src, dst = edge_index[0], edge_index[1]

        e = torch.cat([Wh[src], Wh[dst]], dim=-1)  # (E, 2*out_dim)
        alpha = self.a(e).squeeze(-1)              # (E,)
        if edge_weight is not None:
            alpha = alpha * edge_weight
        alpha = torch.zeros(h.size(0), dtype=h.dtype, device=h.device).scatter_add(
            0, dst,
            torch.softmax_like_scatter(alpha, dst, h.size(0))
            if hasattr(torch, 'softmax_like_scatter') else alpha
        )
        # Use sparse softmax per destination node
        alpha_exp = torch.exp(alpha - alpha.max())
        denom = torch.zeros(h.size(0), device=h.device).scatter_add(0, dst, alpha_exp[src])
        alpha_norm = alpha_exp[src] / (denom[dst] + 1e-9)
        alpha_norm = self.dropout(alpha_norm)

        out = torch.zeros_like(Wh)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(Wh[src]),
                         alpha_norm.unsqueeze(-1) * Wh[src])
        return F.elu(out)


class _MetapathAttentionLayer(nn.Module):
    """Single metapath: GAT aggregation over the precomputed adjacency."""

    def __init__(self, emb_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

    def forward(self, src_emb: torch.Tensor, dst_emb: torch.Tensor,
                edge_index: torch.Tensor, edge_weight: torch.Tensor | None,
                query_idx: torch.Tensor) -> torch.Tensor:
        """Aggregate src embeddings to query (dst) nodes via the metapath adjacency.

        Returns (len(query_idx), hidden_dim) representations.
        """
        # Only aggregate over edges whose dst is in query_idx
        # For simplicity: mean-aggregate src embeddings weighted by edge_weight
        q_set = set(query_idx.tolist())
        mask = torch.tensor(
            [int(edge_index[1, i].item()) in q_set for i in range(edge_index.size(1))],
            dtype=torch.bool, device=edge_index.device,
        )
        if mask.sum() == 0:
            return torch.zeros(len(query_idx), src_emb.size(-1), device=src_emb.device)

        ei = edge_index[:, mask]
        ew = edge_weight[mask] if edge_weight is not None else torch.ones(
            mask.sum(), device=edge_index.device
        )

        # Remap dst indices to query_idx positions
        dst_global = ei[1]
        idx_map = {v: k for k, v in enumerate(query_idx.tolist())}
        dst_local = torch.tensor(
            [idx_map[int(d)] for d in dst_global.tolist()],
            dtype=torch.long, device=edge_index.device,
        )

        aggr = torch.zeros(len(query_idx), src_emb.size(-1), device=src_emb.device)
        weight_sum = torch.zeros(len(query_idx), 1, device=src_emb.device)
        aggr.scatter_add_(0, dst_local.unsqueeze(-1).expand(-1, src_emb.size(-1)),
                          ew.unsqueeze(-1) * src_emb[ei[0]])
        weight_sum.scatter_add_(0, dst_local.unsqueeze(-1), ew.unsqueeze(-1))
        aggr = aggr / (weight_sum + 1e-9)
        return self.conv(aggr)


class _MetapathHANModule(nn.Module):
    """HAN-style model with precomputed metapath adjacency matrices."""

    def __init__(self, n_drugs: int, n_diseases: int, metapath_keys: list[str],
                 emb_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.drug_emb = nn.Embedding(n_drugs, emb_dim)
        self.disease_emb = nn.Embedding(n_diseases, emb_dim)

        self.mp_keys = metapath_keys
        n_mp = len(metapath_keys)

        # One attention layer per metapath
        self.mp_layers = nn.ModuleList([
            _MetapathAttentionLayer(emb_dim, hidden_dim, dropout)
            for _ in range(n_mp)
        ])

        # Semantic-level attention
        self.sem_q = nn.Parameter(torch.empty(hidden_dim))
        nn.init.xavier_uniform_(self.sem_q.unsqueeze(0))
        self.sem_W = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Link prediction head
        self.link_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def _semantic_aggregate(self, mp_repr_list: list[torch.Tensor]) -> torch.Tensor:
        """Weighted sum of metapath representations via semantic attention."""
        # mp_repr_list: list of (N, H) tensors
        stacked = torch.stack(mp_repr_list, dim=1)          # (N, n_mp, H)
        e = torch.tanh(self.sem_W(stacked))                  # (N, n_mp, H)
        w = torch.matmul(e, self.sem_q)                      # (N, n_mp)
        alpha = torch.softmax(w, dim=-1).unsqueeze(-1)       # (N, n_mp, 1)
        return (alpha * stacked).sum(dim=1)                  # (N, H)

    def forward(self, metapath_graphs: dict, drug_idx: torch.Tensor,
                disease_idx: torch.Tensor) -> torch.Tensor:
        drug_h = self.drug_emb.weight      # (n_drugs, emb_dim)
        disease_h = self.disease_emb.weight  # (n_diseases, emb_dim)

        drug_repr_list = []
        disease_repr_list = []

        for mp_key, layer in zip(self.mp_keys, self.mp_layers):
            if mp_key not in metapath_graphs:
                # Fallback: zero representation
                drug_repr_list.append(
                    torch.zeros(len(drug_idx), drug_h.size(-1), device=drug_h.device)
                )
                disease_repr_list.append(
                    torch.zeros(len(disease_idx), drug_h.size(-1), device=drug_h.device)
                )
                continue

            edge_index, edge_weight = metapath_graphs[mp_key]
            edge_index = edge_index.to(drug_h.device)
            edge_weight = edge_weight.to(drug_h.device)

            # drug-side: aggregate diseases that drugs connect to
            dr = layer(disease_h, drug_h, edge_index, edge_weight, drug_idx)
            drug_repr_list.append(dr)

            # disease-side: aggregate drugs that diseases connect to (transpose adj)
            ei_rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
            dis = layer(drug_h, disease_h, ei_rev, edge_weight, disease_idx)
            disease_repr_list.append(dis)

        drug_repr = self._semantic_aggregate(drug_repr_list)       # (B, H)
        disease_repr = self._semantic_aggregate(disease_repr_list)  # (B, H)

        return self.link_mlp(
            torch.cat([drug_repr, disease_repr], dim=-1)
        ).squeeze(-1)


class MetapathModel(BaseModel):
    """Wraps _MetapathHANModule as a BaseModel.

    Requires metapath_graphs kwarg at construction.
    """

    def __init__(self, cfg, data, drug_map, disease_map, metapath_graphs: dict | None = None):
        super().__init__(cfg, data, drug_map, disease_map)
        self.metapath_graphs = metapath_graphs or {}

    def train_and_eval(
        self,
        train_idx,
        test_idx,
        samples_df,
        drug_local,
        dis_local,
        y,
        label='',
    ) -> dict:
        device = self._device()

        n_drugs = len(self.drug_map)
        n_diseases = len(self.disease_map)
        mp_keys = list(self.metapath_graphs.keys())

        emb_dim = self.cfg.get('emb_dim', 64)
        hidden_dim = self.cfg.get('hidden_dim', 64)
        dropout = self.cfg.get('dropout', 0.3)
        lr = self.cfg.get('lr', 1e-3)
        wd = self.cfg.get('weight_decay', 1e-5)
        epochs = self.cfg.get('epochs', 50)

        # Move metapath graphs to device
        mp_graphs_dev = {
            k: (ei.to(device), ew.to(device))
            for k, (ei, ew) in self.metapath_graphs.items()
        }

        model = _MetapathHANModule(
            n_drugs, n_diseases, mp_keys, emb_dim, hidden_dim, dropout
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-5
        )
        criterion = nn.BCEWithLogitsLoss()

        tr_drug = torch.tensor(drug_local[train_idx], dtype=torch.long, device=device)
        tr_dis = torch.tensor(dis_local[train_idx], dtype=torch.long, device=device)
        tr_y = torch.tensor(y[train_idx], dtype=torch.float, device=device)

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(mp_graphs_dev, tr_drug, tr_dis)
            loss = criterion(logits, tr_y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if label and (epoch + 1) % 10 == 0:
                print(
                    f"    {label} epoch {epoch+1:2d}/{epochs}  loss={loss.item():.4f}  "
                    f"lr={scheduler.get_last_lr()[0]:.2e}",
                    flush=True,
                )

        te_drug = torch.tensor(drug_local[test_idx], dtype=torch.long, device=device)
        te_dis = torch.tensor(dis_local[test_idx], dtype=torch.long, device=device)
        te_y = y[test_idx]

        model.eval()
        with torch.no_grad():
            te_probs = torch.sigmoid(
                model(mp_graphs_dev, te_drug, te_dis)
            ).cpu().numpy()

        auc = roc_auc_score(te_y, te_probs)
        pred = (te_probs >= 0.5).astype(int)
        acc = accuracy_score(te_y, pred)
        f1 = f1_score(te_y, pred)
        return {'auc': auc, 'acc': acc, 'f1': f1}

    @staticmethod
    def _device():
        if torch.cuda.is_available():
            return 'cuda'
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
