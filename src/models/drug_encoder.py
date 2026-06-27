"""
src/models/drug_encoder.py

Encodes drug molecules as 2D molecular graphs using Graph Attention Networks.
Uses RDKit to featurize atoms and bonds.

Node features (atoms):
  - Atomic number one-hot (common elements: C,N,O,S,P,F,Cl,Br,I + other)
  - Degree (number of bonds)
  - Formal charge
  - Hybridization (sp, sp2, sp3, sp3d, sp3d2)
  - Aromaticity
  - Ring membership
  - Hydrogen count
  Total: 74 features

Edge features (bonds):
  - Bond type (single, double, triple, aromatic)
  - In ring
  - Conjugated
  Total: 6 features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_add_pool
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import numpy as np
from typing import Optional


# Atom feature vocabularies
ATOM_TYPES = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "other"]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def one_hot(value, vocab: list) -> list:
    """One-hot encode value against vocab, with an 'other' bin at the end."""
    encoding = [0] * (len(vocab) + 1)
    if value in vocab:
        encoding[vocab.index(value)] = 1
    else:
        encoding[-1] = 1
    return encoding


def atom_features(atom) -> list:
    """Compute atom feature vector (74 dims)."""
    symbol = atom.GetSymbol()
    feats = (
        one_hot(symbol, ATOM_TYPES[:-1])                        # 10
        + [atom.GetDegree() / 10.0]                             # 1
        + [atom.GetFormalCharge() / 5.0]                        # 1
        + one_hot(atom.GetHybridization(), HYBRIDIZATIONS)      # 6
        + [int(atom.GetIsAromatic())]                           # 1
        + [int(atom.IsInRing())]                                # 1
        + [atom.GetTotalNumHs() / 8.0]                          # 1
        + [atom.GetMass() / 200.0]                              # 1
        + [atom.GetImplicitValence() / 8.0]                     # 1
    )
    return feats  # total: 23 features (compact)


def bond_features(bond) -> list:
    """Compute bond feature vector (6 dims)."""
    feats = (
        one_hot(bond.GetBondType(), BOND_TYPES)                 # 5
        + [int(bond.IsInRing())]                                # 1
    )
    return feats  # total: 6 features


def mol_to_graph(smiles: str, drug_name: str = "") -> Optional[Data]:
    """
    Convert SMILES string to PyG Data object.
    Returns None if SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    
    # Node features
    node_feats = []
    for atom in mol.GetAtoms():
        node_feats.append(atom_features(atom))
    x = torch.tensor(node_feats, dtype=torch.float32)

    # Edge features (bidirectional)
    edge_indices = []
    edge_feats = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feats = bond_features(bond)
        edge_indices += [[i, j], [j, i]]
        edge_feats += [feats, feats]

    if len(edge_indices) == 0:
        return None

    edge_index = torch.tensor(edge_indices, dtype=torch.long).T  # [2, E]
    edge_attr = torch.tensor(edge_feats, dtype=torch.float32)    # [E, 6]

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        smiles=smiles,
        name=drug_name,
    )


class DrugEncoder(nn.Module):
    """
    GATv2-based drug molecular graph encoder.
    
    Input: Drug molecular graph (atoms as nodes, bonds as edges)
    Output: Drug embedding [B, out_dim]
    """

    def __init__(
        self,
        node_in_dim: int = 23,
        edge_in_dim: int = 6,
        hidden_dim: int = 128,
        out_dim: int = 256,
        num_layers: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.node_embed = nn.Linear(node_in_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_in_dim, hidden_dim)

        self.conv_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim
            out_channels = hidden_dim // heads
            self.conv_layers.append(
                GATv2Conv(
                    in_channels=in_dim,
                    out_channels=out_channels,
                    heads=heads,
                    edge_dim=hidden_dim,
                    dropout=dropout,
                    concat=True,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        # Dual pooling readout
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, out_dim),  # mean + sum pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(
        self,
        x: torch.Tensor,           # [N_atoms, node_in_dim]
        edge_index: torch.Tensor,  # [2, E]
        edge_attr: torch.Tensor,   # [E, edge_in_dim]
        batch: torch.Tensor,       # [N_atoms]
    ) -> torch.Tensor:
        """Returns drug embedding [B, out_dim]."""
        x = self.node_embed(x)
        edge_attr = self.edge_embed(edge_attr)

        for conv, norm in zip(self.conv_layers, self.norms):
            x_new = conv(x, edge_index, edge_attr)
            x_new = self.dropout(norm(x_new))
            x = x + x_new  # residual

        # Dual pooling: concatenate mean and sum
        x_mean = global_mean_pool(x, batch)  # [B, hidden_dim]
        x_sum = global_add_pool(x, batch)    # [B, hidden_dim]
        x_pool = torch.cat([x_mean, x_sum], dim=-1)  # [B, 2*hidden_dim]

        return self.readout(x_pool)  # [B, out_dim]


# SMILES for known drugs (curated)
DRUG_SMILES = {
    "Cisplatin":        "[Pt](Cl)(Cl)(N)N",
    "Doxorubicin":      "COc1cccc2C(=O)c3c(O)c4c(c(O)c3C(=O)c12)C[C@@](O)(C(=O)CO)C[C@H]4O[C@H]1C[C@@H](N)[C@H](O)[C@H](C)O1",
    "Paclitaxel":       "CC1=C2[C@@]([C@H](C(=O)[C@@H]3[C@@]2(CC[C@H]4[C@]3(C(=O)O[C@H]([C@@H]4OC(=O)c5ccccc5)OC(=O)c6ccccc6)C)OC(=O)C)(C[C@@H]1OC(=O)[C@H](O)[C@@H](NC(=O)c7ccccc7)c8ccccc8)O)(C)C",
    "Erlotinib":        "C#Cc1cccc(Nc2ncnc3cc(OCCO)c(OCCO)cc23)c1",
    "Sorafenib":        "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1",
    "Dasatinib":        "Cc1nc(Nc2ncc(C(=O)Nc3c(C)cccc3Cl)s2)cc(N2CCN(CCO)CC2)n1",
    "Gemcitabine":      "NC(=O)[C@@H]1CC(F)(F)O[C@@H]1CO",
    "Irinotecan":       "CCc1c(C2=CC3=C(CN(CC3)C(=O)OCC)C(=O)c4c(cc5cc(OC)c6c(c5n4)CCCC6=O)C2)c(O)cc(C(=O)N1CCCCC1)c1",
    "Vemurafenib":      "CCCS(=O)(=O)Nc1ccc(F)c(C(=O)c2c[nH]c3cc(Cl)ccc23)c1",
    "Trametinib":       "CC(=O)Nc1ccc(-c2cc3c(cc2F)nc(Nc2ccc(I)cc2F)nc3-c2coc(=O)[nH]2)cc1",
    "Palbociclib":      "CC1=C(C(=O)Nc2ncnc3[nH]ccc23)C=CN1c1cccc(=O)[nH]1",
    "Venetoclax":       "CC1(C)CCC(=C1)c1ccc(-n2c(=O)ccc2-c2ccc(N3CC[C@@H](CNc4ccc(C(F)(F)F)cc4Cl)CC3)cc2)cc1",
    "Nilotinib":        "Cc1ccc(-c2cc(NC(=O)c3ccc(C)c(Nc4nccc(-c5cccnc5)n4)c3)ccn2)cc1C(F)(F)F",
    "Vincristine":      "CCC1(CC2CC(C3=C(CCN(C2)C1)C4=CC=CC=C4N3)(C(=O)OC)O)OC(=O)C",
    "Hydroxyurea":      "NC(=O)NO",
    "Methotrexate":     "CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(C(=O)N[C@@H](CCC(=O)O)C(=O)O)cc1",
    "Fluorouracil":     "O=c1[nH]cc(F)c(=O)[nH]1",
    "Cytarabine":       "Nc1ccn([C@@H]2O[C@H](CO)[C@@H](O)[C@@H]2O)c(=O)n1",
}


def get_drug_graph(drug_name: str) -> Optional[Data]:
    """Get molecular graph for a drug by name."""
    smiles = DRUG_SMILES.get(drug_name)
    if smiles is None:
        return None
    return mol_to_graph(smiles, drug_name)


if __name__ == "__main__":
    # Test drug encoding
    graph = get_drug_graph("Erlotinib")
    if graph:
        print(f"Erlotinib: {graph.x.shape[0]} atoms, {graph.edge_index.shape[1]//2} bonds")
        print(f"  Node features: {graph.x.shape}")
        print(f"  Edge features: {graph.edge_attr.shape}")

    encoder = DrugEncoder()
    print(f"\nDrugEncoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")
