"""
src/models/synergy_model.py

DrugSynergy3D: Full model combining:
  1. GVP-GNN branches for each drug's protein target pocket (SE3-equivariant)
  2. GATv2 branches for each drug's molecular graph
  3. Cross-attention fusion (drug A pocket ↔ drug B pocket)
  4. PPI interaction features (STRING DB)
  5. Synergy prediction head

Architecture overview:
  Drug A molecular graph  →  DrugEncoder  →  drug_a_emb [B, d]
  Drug A protein pocket   →  GVPEncoder   →  prot_a_emb [B, d]
  Drug B molecular graph  →  DrugEncoder  →  drug_b_emb [B, d]
  Drug B protein pocket   →  GVPEncoder   →  prot_b_emb [B, d]
  PPI features            →               →  ppi_emb    [B, 6]
  
  Fusion:
    prot_a_emb, prot_b_emb → cross_attention → fused_prot [B, d]
    drug_a_emb, drug_b_emb → element-wise ops → fused_drug [B, d]
    [fused_prot | fused_drug | ppi_emb] → MLP → synergy_score [B, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .gvp_gnn import GVPEncoder
from .drug_encoder import DrugEncoder


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention between two protein pocket embeddings.
    Drug A's pocket attends to Drug B's pocket and vice versa.
    This captures geometric complementarity between binding sites.
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn_ab = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_ba = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        emb_a: torch.Tensor,  # [B, d]
        emb_b: torch.Tensor,  # [B, d]
    ) -> torch.Tensor:
        """Returns fused embedding [B, d]."""
        # Unsqueeze to [B, 1, d] for attention
        a = emb_a.unsqueeze(1)
        b = emb_b.unsqueeze(1)

        # A attends to B
        a_cross, _ = self.attn_ab(a, b, b)
        a_cross = self.norm1(a + a_cross)

        # B attends to A
        b_cross, _ = self.attn_ba(b, a, a)
        b_cross = self.norm2(b + b_cross)

        # Squeeze and concatenate
        fused = torch.cat([a_cross.squeeze(1), b_cross.squeeze(1)], dim=-1)  # [B, 2d]
        fused = self.norm3(emb_a + self.ff(fused))  # residual on a's embedding

        return fused  # [B, d]


class SynergyHead(nn.Module):
    """MLP synergy prediction head."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list = [512, 256, 128],
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        cur_dim = in_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(cur_dim, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            cur_dim = h
        layers.append(nn.Linear(cur_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # [B]


class DrugSynergy3D(nn.Module):
    """
    Full DrugSynergy3D model.
    
    Novel contributions:
    1. First model to use SE(3)-equivariant GNN on AF2 binding pocket graphs
       for drug synergy prediction
    2. Cross-attention between pocket geometries captures structural complementarity
    3. PPI network features provide biological context beyond molecular structure
    """

    def __init__(
        self,
        # GVP encoder config
        gvp_node_dims=(27, 1),
        gvp_edge_dims=(5, 1),
        gvp_hidden_dims=(128, 16),
        gvp_out_dim=256,
        gvp_num_layers=5,
        gvp_drop_rate=0.1,
        # Drug encoder config
        drug_node_dim=23,
        drug_edge_dim=6,
        drug_hidden_dim=128,
        drug_out_dim=256,
        drug_num_layers=4,
        drug_heads=4,
        # Fusion config
        fusion_d_model=256,
        fusion_heads=8,
        fusion_dropout=0.1,
        # Head config
        head_hidden=[512, 256, 128],
        head_dropout=0.2,
        # PPI features
        ppi_dim=6,
    ):
        super().__init__()

        d = gvp_out_dim  # unified embedding dim

        # Protein pocket encoders (shared weights — drug A and B use same encoder)
        self.pocket_encoder = GVPEncoder(
            node_in_dims=gvp_node_dims,
            edge_in_dims=gvp_edge_dims,
            hidden_dims=gvp_hidden_dims,
            out_dim=d,
            num_layers=gvp_num_layers,
            drop_rate=gvp_drop_rate,
        )

        # Drug molecular graph encoders (shared weights)
        self.drug_encoder = DrugEncoder(
            node_in_dim=drug_node_dim,
            edge_in_dim=drug_edge_dim,
            hidden_dim=drug_hidden_dim,
            out_dim=d,
            num_layers=drug_num_layers,
            heads=drug_heads,
        )

        # Cross-attention fusion for pocket geometries
        self.pocket_fusion = CrossAttentionFusion(
            d_model=d,
            n_heads=fusion_heads,
            dropout=fusion_dropout,
        )

        # Drug pair interaction features
        # Element-wise: concat(a+b, a*b, |a-b|) = 3 drug pair features
        self.drug_pair_proj = nn.Sequential(
            nn.Linear(d * 3, d),
            nn.GELU(),
        )

        # PPI feature projection
        self.ppi_proj = nn.Sequential(
            nn.Linear(ppi_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )

        # Final head: pocket_fused + drug_pair + ppi → synergy
        head_in = d + d + 64
        self.head = SynergyHead(
            in_dim=head_in,
            hidden_dims=head_hidden,
            dropout=head_dropout,
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode_pocket(
        self,
        x_s, x_v, edge_index, edge_s, edge_v, batch
    ) -> torch.Tensor:
        """Encode a protein pocket graph → embedding [B, d]."""
        return self.pocket_encoder(x_s, x_v, edge_index, edge_s, edge_v, batch)

    def encode_drug(
        self,
        x, edge_index, edge_attr, batch
    ) -> torch.Tensor:
        """Encode a drug molecular graph → embedding [B, d]."""
        return self.drug_encoder(x, edge_index, edge_attr, batch)

    def forward(
        self,
        # Drug A protein pocket
        pa_xs, pa_xv, pa_ei, pa_es, pa_ev, pa_batch,
        # Drug B protein pocket
        pb_xs, pb_xv, pb_ei, pb_es, pb_ev, pb_batch,
        # Drug A molecular graph
        da_x, da_ei, da_ea, da_batch,
        # Drug B molecular graph
        db_x, db_ei, db_ea, db_batch,
        # PPI features between targets
        ppi_feats,                          # [B, 6]
    ) -> torch.Tensor:
        """
        Returns synergy score prediction [B].
        Higher = more synergistic.
        """

        # Encode protein pockets (SE3-equivariant)
        prot_a = self.encode_pocket(pa_xs, pa_xv, pa_ei, pa_es, pa_ev, pa_batch)
        prot_b = self.encode_pocket(pb_xs, pb_xv, pb_ei, pb_es, pb_ev, pb_batch)

        # Encode drug molecules
        drug_a = self.encode_drug(da_x, da_ei, da_ea, da_batch)
        drug_b = self.encode_drug(db_x, db_ei, db_ea, db_batch)

        # Cross-attention between protein pockets
        pocket_fused = self.pocket_fusion(prot_a, prot_b)  # [B, d]

        # Drug pair features: capture interaction between the two molecules
        drug_pair = self.drug_pair_proj(
            torch.cat([
                drug_a + drug_b,                    # additive
                drug_a * drug_b,                    # multiplicative (interaction)
                (drug_a - drug_b).abs(),            # difference (asymmetry)
            ], dim=-1)
        )  # [B, d]

        # PPI features
        ppi_emb = self.ppi_proj(ppi_feats)  # [B, 64]

        # Concatenate all features and predict
        combined = torch.cat([pocket_fused, drug_pair, ppi_emb], dim=-1)
        synergy = self.head(combined)  # [B]

        return synergy

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        pocket_enc = sum(p.numel() for p in self.pocket_encoder.parameters())
        drug_enc = sum(p.numel() for p in self.drug_encoder.parameters())
        fusion = sum(p.numel() for p in self.pocket_fusion.parameters())
        head = sum(p.numel() for p in self.head.parameters())
        return {
            "total": total,
            "pocket_encoder": pocket_enc,
            "drug_encoder": drug_enc,
            "fusion": fusion,
            "head": head,
        }


if __name__ == "__main__":
    model = DrugSynergy3D()
    counts = model.count_parameters()
    print("DrugSynergy3D parameter counts:")
    for k, v in counts.items():
        print(f"  {k}: {v:,}")
