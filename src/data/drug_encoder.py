import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_add_pool
from rdkit import Chem
import numpy as np
from typing import Optional

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

def one_hot(value, vocab):
    enc = [0]*(len(vocab)+1)
    enc[vocab.index(value) if value in vocab else -1] = 1
    return enc

def atom_features(atom):
    return (
        one_hot(atom.GetSymbol(), ["C","N","O","S","P","F","Cl","Br","I"])
        + [atom.GetDegree()/10., atom.GetFormalCharge()/5.,
           int(atom.GetIsAromatic()), int(atom.IsInRing()),
           atom.GetTotalNumHs()/8., atom.GetMass()/200.]
    )

def bond_features(bond):
    return one_hot(bond.GetBondType(), BOND_TYPES) + [int(bond.IsInRing())]

def mol_to_graph(smiles, drug_name=""):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    node_feats = [atom_features(a) for a in mol.GetAtoms()]
    x = torch.tensor(node_feats, dtype=torch.float32)
    edge_indices, edge_feats = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        f = bond_features(bond)
        edge_indices += [[i,j],[j,i]]
        edge_feats += [f, f]
    if not edge_indices:
        return None
    edge_index = torch.tensor(edge_indices, dtype=torch.long).T
    edge_attr = torch.tensor(edge_feats, dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles, name=drug_name)

DRUG_SMILES = {
    "Cisplatin":      "[Pt](Cl)(Cl)(N)N",
    "Doxorubicin":    "COc1cccc2C(=O)c3c(O)c4c(c(O)c3C(=O)c12)CC(O)(C(=O)CO)CC4O",
    "Paclitaxel":     "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(=C2OC1=O)C)O)OC(=O)C5=CC=CC=C5)(CO4)OC(=O)C)O)OC(=O)C6=CC=CC=C6",
    "Erlotinib":      "C#Cc1cccc(Nc2ncnc3cc(OCCO)c(OCCO)cc23)c1",
    "Sorafenib":      "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1",
    "Dasatinib":      "Cc1nc(Nc2ncc(C(=O)Nc3c(C)cccc3Cl)s2)cc(N2CCN(CCO)CC2)n1",
    "Gemcitabine":    "NC(=O)C1CC(F)(F)OC1CO",
    "Irinotecan":     "CCC1=C2CN3CCC4=C(C3CC2=NC1=O)C(=O)OCC4=O",
    "Vemurafenib":    "CCCS(=O)(=O)Nc1ccc(F)c(C(=O)c2c[nH]c3cc(Cl)ccc23)c1",
    "Trametinib":     "CC(=O)Nc1ccc(-c2cc3c(cc2F)nc(Nc2ccc(I)cc2F)nc3N2CCOCC2)cc1",
    "Palbociclib":    "CC1=C(C(=O)Nc2ncnc3[nH]ccc23)C=CN1c1cccc(=O)[nH]1",
    "Venetoclax":     "CC1(CCC(=C1)c1ccc(-n2c(=O)ccc2-c2ccc(N3CCCC3CNc3ccc(C(F)(F)F)cc3Cl)cc2)cc1)C",
    "Nilotinib":      "Cc1ccc(-c2cc(NC(=O)c3ccc(C)c(Nc4nccc(-c5cccnc5)n4)c3)ccn2)cc1C(F)(F)F",
    "Vincristine":    "CCC1(CC2CC(C3=C(CCN(C2)C1)C4=CC=CC=C4N3)(C(=O)OC)O)OC(=O)C",
    "Hydroxyurea":    "NC(=O)NO",
    "Methotrexate":   "CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(C(=O)NC(CCC(=O)O)C(=O)O)cc1",
    "Fluorouracil":   "O=c1[nH]cc(F)c(=O)[nH]1",
    "Cytarabine":     "Nc1ccn(C2OC(CO)C(O)C2O)c(=O)n1",
}

def get_drug_graph(drug_name):
    smiles = DRUG_SMILES.get(drug_name)
    if smiles is None:
        return None
    return mol_to_graph(smiles, drug_name)

class DrugEncoder(nn.Module):
    def __init__(self, node_in_dim=16, edge_in_dim=5, hidden_dim=128,
                 out_dim=256, num_layers=4, heads=4, dropout=0.1):
        super().__init__()
        self.node_embed = nn.Linear(node_in_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_in_dim, hidden_dim)
        self.convs = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim//heads, heads=heads,
                      edge_dim=hidden_dim, dropout=dropout, concat=True)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim*2, out_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(out_dim, out_dim)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_embed(x)
        edge_attr = self.edge_embed(edge_attr)
        for conv, norm in zip(self.convs, self.norms):
            x = x + self.dropout(norm(conv(x, edge_index, edge_attr)))
        return self.readout(torch.cat([global_mean_pool(x, batch),
                                       global_add_pool(x, batch)], dim=-1))
