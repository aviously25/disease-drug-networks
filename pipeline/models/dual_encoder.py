"""DualEncoder model — two HeteroConv stacks with cross-attention link prediction."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from .base import BaseModel


class _DualEncoderModule(nn.Module):
    """Two independent HeteroConv stacks (drug / disease encoders) + cross-attention.

    Architecture:
        drug_emb    (B, H)  → queries disease
        disease_emb (B, H)  → queries drug

        drug_attended    = CrossAttn(Q=drug_emb,    K=disease_emb, V=disease_emb)
        disease_attended = CrossAttn(Q=disease_emb, K=drug_emb,   V=drug_emb)

        link_input = cat([drug_emb, disease_emb, drug_attended, disease_attended])  # (B, 4H)
        logit = MLP(link_input)
    """

    def __init__(self, node_type_sizes, edge_types, emb_dim, hidden_dim,
                 num_layers, num_heads, dropout):
        super().__init__()

        # Shared embeddings (both encoders start from the same lookup)
        self.embeddings = nn.ModuleDict({
            nt: nn.Embedding(n, emb_dim)
            for nt, n in node_type_sizes.items()
        })

        # Drug encoder
        self.drug_convs = nn.ModuleList([
            HeteroConv({et: SAGEConv((-1, -1), hidden_dim) for et in edge_types}, aggr='mean')
            for _ in range(num_layers)
        ])

        # Disease encoder (separate parameters)
        self.disease_convs = nn.ModuleList([
            HeteroConv({et: SAGEConv((-1, -1), hidden_dim) for et in edge_types}, aggr='mean')
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # Cross-attention: drug queries disease, and disease queries drug
        self.drug_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=dropout
        )
        self.disease_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=dropout
        )

        # Link prediction head: 4H → 1
        self.link_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _encode(self, convs, data) -> dict:
        x_dict = {nt: self.embeddings[nt].weight for nt in self.embeddings}
        for conv in convs:
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {k: self.dropout(F.relu(v)) for k, v in x_dict.items()}
        return x_dict

    def forward(self, data, drug_idx, disease_idx):
        # Encode with each stack
        drug_x = self._encode(self.drug_convs, data)
        disease_x = self._encode(self.disease_convs, data)

        drug_emb = drug_x['drug'][drug_idx]         # (B, H)
        disease_emb = disease_x['disease'][disease_idx]  # (B, H)

        # Cross-attention: reshape to (B, 1, H) for MultiheadAttention
        dru_q = drug_emb.unsqueeze(1)     # (B, 1, H)
        dis_k = disease_emb.unsqueeze(1)  # (B, 1, H)

        drug_attended, _ = self.drug_attn(dru_q, dis_k, dis_k)   # (B, 1, H)
        disease_attended, _ = self.disease_attn(dis_k, dru_q, dru_q)  # (B, 1, H)

        drug_attended = drug_attended.squeeze(1)      # (B, H)
        disease_attended = disease_attended.squeeze(1)  # (B, H)

        link_input = torch.cat(
            [drug_emb, disease_emb, drug_attended, disease_attended], dim=-1
        )  # (B, 4H)
        return self.link_mlp(link_input).squeeze(-1)


class DualEncoder(BaseModel):
    """Wraps _DualEncoderModule as a BaseModel."""

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
        data_dev = self.data.to(device)

        node_type_sizes = {nt: data_dev[nt].num_nodes for nt in data_dev.node_types}
        edge_types = list(data_dev.edge_index_dict.keys())

        emb_dim = self.cfg.get('emb_dim', 64)
        hidden_dim = self.cfg.get('hidden_dim', 64)
        num_layers = self.cfg.get('num_layers', 2)
        num_heads = self.cfg.get('num_heads', 4)
        dropout = self.cfg.get('dropout', 0.3)
        lr = self.cfg.get('lr', 1e-3)
        wd = self.cfg.get('weight_decay', 1e-5)
        epochs = self.cfg.get('epochs', 50)

        model = _DualEncoderModule(
            node_type_sizes, edge_types, emb_dim, hidden_dim,
            num_layers, num_heads, dropout
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
            logits = model(data_dev, tr_drug, tr_dis)
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
            te_probs = torch.sigmoid(model(data_dev, te_drug, te_dis)).cpu().numpy()

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
