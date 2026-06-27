"""
src/data/dataset.py

PyTorch Dataset for DrugSynergy3D.
Each sample = (drug_pair, cell_line) → synergy_score.

Handles:
- Loading precomputed protein pocket graphs
- Loading drug molecular graphs
- PPI features for the target pair
- Train/val/test splits by cell line (no leakage)
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from typing import Optional, Dict, List, Tuple
from .drug_encoder import get_drug_graph, mol_to_graph, DRUG_SMILES
from .ppi_network import get_ppi_edge_features


class SynergyDataset(Dataset):
    """
    Dataset of drug pair synergy observations.
    
    Each item returns a dict with:
        prot_a_graph: PyG Data (protein A pocket)
        prot_b_graph: PyG Data (protein B pocket)
        drug_a_graph: PyG Data (drug A molecule)
        drug_b_graph: PyG Data (drug B molecule)
        ppi_feats:    Tensor [6] PPI features between targets
        css_score:    float synergy score
        synergy_label: int 0/1
        drug_a_name:  str
        drug_b_name:  str
        cell_line:    str
    """

    def __init__(
        self,
        almanac_df: pd.DataFrame,
        target_map: pd.DataFrame,
        protein_graphs: Dict[str, Data],
        ppi_df: Optional[pd.DataFrame] = None,
        mode: str = "train",  # train / val / test
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        seed: int = 42,
    ):
        self.protein_graphs = protein_graphs
        self.ppi_df = ppi_df

        # Build drug name → graph lookup
        self.drug_graphs = {}
        for name in DRUG_SMILES:
            g = get_drug_graph(name)
            if g is not None:
                self.drug_graphs[name] = g

        # Build drug name → [uniprot_ids] lookup
        self.drug_to_uniprots = {}
        for _, row in target_map.iterrows():
            name = str(row["name"])
            if pd.notna(row["uniprot_ids"]) and row["uniprot_ids"]:
                self.drug_to_uniprots[name] = str(row["uniprot_ids"]).split(";")

        # Split by cell line to avoid leakage
        all_cell_lines = almanac_df["cell_line"].unique()
        rng = np.random.default_rng(seed)
        rng.shuffle(all_cell_lines)

        n = len(all_cell_lines)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        if mode == "train":
            selected_lines = set(all_cell_lines[:n_train])
        elif mode == "val":
            selected_lines = set(all_cell_lines[n_train:n_train + n_val])
        else:  # test
            selected_lines = set(all_cell_lines[n_train + n_val:])

        df = almanac_df[almanac_df["cell_line"].isin(selected_lines)].copy()

        # Filter to pairs where we have both drug graphs and protein graphs
        valid_rows = []
        for _, row in df.iterrows():
            d1, d2 = str(row["drug1_name"]), str(row["drug2_name"])
            if d1 not in self.drug_graphs or d2 not in self.drug_graphs:
                continue
            if d1 not in self.drug_to_uniprots or d2 not in self.drug_to_uniprots:
                continue
            u1 = self.drug_to_uniprots[d1][0]
            u2 = self.drug_to_uniprots[d2][0]
            if u1 not in self.protein_graphs or u2 not in self.protein_graphs:
                continue
            valid_rows.append(row)

        self.data = pd.DataFrame(valid_rows).reset_index(drop=True)
        print(f"[Dataset] {mode}: {len(self.data)} valid pairs "
              f"({len(selected_lines)} cell lines)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        d1, d2 = str(row["drug1_name"]), str(row["drug2_name"])

        u1 = self.drug_to_uniprots[d1][0]
        u2 = self.drug_to_uniprots[d2][0]

        prot_a = self.protein_graphs[u1]
        prot_b = self.protein_graphs[u2]
        drug_a = self.drug_graphs[d1]
        drug_b = self.drug_graphs[d2]

        # PPI features
        if self.ppi_df is not None:
            ppi_feats = get_ppi_edge_features(u1, u2, self.ppi_df)
        else:
            ppi_feats = torch.zeros(6, dtype=torch.float32)

        css_score = float(row["css_score"])
        synergy_label = int(row["synergy_label"])

        return {
            "prot_a": prot_a,
            "prot_b": prot_b,
            "drug_a": drug_a,
            "drug_b": drug_b,
            "ppi_feats": ppi_feats,
            "css_score": torch.tensor(css_score, dtype=torch.float32),
            "synergy_label": torch.tensor(synergy_label, dtype=torch.long),
            "drug_a_name": d1,
            "drug_b_name": d2,
            "cell_line": str(row["cell_line"]),
            "uniprot_a": u1,
            "uniprot_b": u2,
        }


def collate_fn(batch: List[dict]) -> dict:
    """
    Custom collate: batch PyG graphs and regular tensors separately.
    """
    prot_a_batch = Batch.from_data_list([b["prot_a"] for b in batch])
    prot_b_batch = Batch.from_data_list([b["prot_b"] for b in batch])
    drug_a_batch = Batch.from_data_list([b["drug_a"] for b in batch])
    drug_b_batch = Batch.from_data_list([b["drug_b"] for b in batch])

    return {
        # Protein A pocket
        "pa_xs": prot_a_batch.x,
        "pa_xv": prot_a_batch.v,
        "pa_ei": prot_a_batch.edge_index,
        "pa_es": prot_a_batch.edge_attr,
        "pa_ev": prot_a_batch.edge_vec,
        "pa_batch": prot_a_batch.batch,

        # Protein B pocket
        "pb_xs": prot_b_batch.x,
        "pb_xv": prot_b_batch.v,
        "pb_ei": prot_b_batch.edge_index,
        "pb_es": prot_b_batch.edge_attr,
        "pb_ev": prot_b_batch.edge_vec,
        "pb_batch": prot_b_batch.batch,

        # Drug A molecule
        "da_x": drug_a_batch.x,
        "da_ei": drug_a_batch.edge_index,
        "da_ea": drug_a_batch.edge_attr,
        "da_batch": drug_a_batch.batch,

        # Drug B molecule
        "db_x": drug_b_batch.x,
        "db_ei": drug_b_batch.edge_index,
        "db_ea": drug_b_batch.edge_attr,
        "db_batch": drug_b_batch.batch,

        # Scalar features
        "ppi_feats": torch.stack([b["ppi_feats"] for b in batch]),
        "css_score": torch.stack([b["css_score"] for b in batch]),
        "synergy_label": torch.stack([b["synergy_label"] for b in batch]),

        # Metadata
        "drug_a_name": [b["drug_a_name"] for b in batch],
        "drug_b_name": [b["drug_b_name"] for b in batch],
        "cell_line": [b["cell_line"] for b in batch],
    }


def get_dataloaders(
    almanac_df: pd.DataFrame,
    target_map: pd.DataFrame,
    protein_graphs: Dict[str, Data],
    ppi_df: Optional[pd.DataFrame] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders."""

    train_ds = SynergyDataset(
        almanac_df, target_map, protein_graphs, ppi_df,
        mode="train", seed=seed
    )
    val_ds = SynergyDataset(
        almanac_df, target_map, protein_graphs, ppi_df,
        mode="val", seed=seed
    )
    test_ds = SynergyDataset(
        almanac_df, target_map, protein_graphs, ppi_df,
        mode="test", seed=seed
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader
