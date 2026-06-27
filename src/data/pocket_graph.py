"""
src/data/pocket_graph.py

Takes an AlphaFold2 PDB file and builds a geometric residue graph
of the predicted binding pocket.

Node features per residue:
  - One-hot amino acid type (20 dim)
  - AlphaFold pLDDT confidence score (1 dim, normalized)
  - Secondary structure (3 dim: helix/sheet/coil via DSSP-lite heuristic)
  - Solvent-accessible surface area proxy (1 dim)
  - Backbone torsion angles sin/cos φ,ψ (4 dim)
  Total: 29 scalar + 1 vector (Cα→Cβ direction) per node

Edge features per residue pair:
  - Euclidean distance (1 dim)
  - Unit displacement vector (3 dim)
  - Sequence separation (1 dim, clamped+normalized)
  Total: 5 scalar + 1 vector per edge

Graph structure:
  - k-NN graph on Cα positions (k=10)
  - Only residues within <cutoff> Å of the pocket centroid
"""

import os
import math
import numpy as np
import torch
from torch_geometric.data import Data
from typing import Optional, Tuple
from Bio.PDB import PDBParser, DSSP
from Bio.PDB.Polypeptide import is_aa
from Bio.Data.IUPACData import protein_letters_3to1
import warnings
import pandas as pd

warnings.filterwarnings("ignore")


# Standard amino acid ordering
AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}
UNK_IDX = len(AA_ORDER)  # 20 for unknown


def parse_pdb_residues(pdb_path: str) -> dict:
    """
    Parse PDB file → extract per-residue info.
    Returns dict keyed by (chain_id, res_seq):
        {
          'aa': single-letter code,
          'ca': Cα position (3,),
          'cb': Cβ position (3,) or None for Gly,
          'plddt': B-factor (AlphaFold pLDDT),
          'phi': torsion or 0.0,
          'psi': torsion or 0.0,
        }
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model = structure[0]

    residues = {}
    for chain in model:
        for res in chain:
            if not is_aa(res, standard=True):
                continue
            resname = res.get_resname()
            try:
                aa = protein_letters_3to1.get(resname.capitalize(), "X")
            except Exception:
                aa = "X"

            ca = None
            cb = None
            plddt = 100.0

            if "CA" in res:
                ca = np.array(res["CA"].get_coord(), dtype=np.float32)
                plddt = float(res["CA"].get_bfactor())  # pLDDT stored in B-factor

            if "CB" in res:
                cb = np.array(res["CB"].get_coord(), dtype=np.float32)
            elif ca is not None:
                # Approximate Cβ for Gly
                cb = ca + np.array([0.0, 0.0, 1.5], dtype=np.float32)

            if ca is None:
                continue

            key = (chain.id, res.get_id()[1])
            residues[key] = {
                "aa": aa,
                "ca": ca,
                "cb": cb,
                "plddt": plddt / 100.0,  # normalize to [0,1]
                "phi": 0.0,
                "psi": 0.0,
            }

    # Compute backbone torsions (simplified — from sequential CA positions)
    keys = sorted(residues.keys())
    for i in range(1, len(keys) - 1):
        prev_ca = residues[keys[i-1]]["ca"]
        curr_ca = residues[keys[i]]["ca"]
        next_ca = residues[keys[i+1]]["ca"]
        # Pseudo-phi: angle between prev→curr and curr→next
        v1 = curr_ca - prev_ca
        v2 = next_ca - curr_ca
        v1 /= (np.linalg.norm(v1) + 1e-8)
        v2 /= (np.linalg.norm(v2) + 1e-8)
        angle = math.acos(float(np.clip(np.dot(v1, v2), -1, 1)))
        residues[keys[i]]["phi"] = angle
        residues[keys[i]]["psi"] = -angle  # pseudo-psi

    return residues


def get_pocket_residues(
    residues: dict,
    pocket_centroid: Optional[np.ndarray] = None,
    cutoff: float = 15.0,
    min_plddt: float = 0.5,
    top_n: int = 80,
) -> list:
    """
    Select residues that form the binding pocket.
    
    Strategy:
    1. If pocket_centroid given: select residues within cutoff Å
    2. Otherwise: use high-pLDDT residues in the geometric center
    3. Always cap at top_n residues for compute efficiency
    """
    keys = list(residues.keys())
    all_ca = np.stack([residues[k]["ca"] for k in keys])

    if pocket_centroid is None:
        # Use residues near the protein center (common for globular proteins)
        pocket_centroid = all_ca.mean(axis=0)

    dists = np.linalg.norm(all_ca - pocket_centroid, axis=1)
    mask = dists < cutoff

    pocket_keys = [k for k, m in zip(keys, mask) if m]

    # Filter by pLDDT confidence
    pocket_keys = [k for k in pocket_keys if residues[k]["plddt"] >= min_plddt]

    # Sort by distance, take top_n
    pocket_keys.sort(key=lambda k: np.linalg.norm(residues[k]["ca"] - pocket_centroid))
    pocket_keys = pocket_keys[:top_n]

    return pocket_keys


def build_node_features(residues: dict, pocket_keys: list) -> torch.Tensor:
    """
    Build node feature matrix [N, 29].
    """
    features = []
    for key in pocket_keys:
        res = residues[key]
        aa = res["aa"]

        # One-hot amino acid (20 dims + 1 unknown = 21)
        aa_feat = [0.0] * 21
        idx = AA_TO_IDX.get(aa, UNK_IDX)
        aa_feat[idx] = 1.0

        # pLDDT (1 dim)
        plddt_feat = [res["plddt"]]

        # Torsion angles sin/cos (4 dims)
        phi, psi = res["phi"], res["psi"]
        torsion_feat = [math.sin(phi), math.cos(phi), math.sin(psi), math.cos(psi)]

        # Normalized sequence position (1 dim)
        seq_pos = [key[1] / 2000.0]  # normalize by typical max length

        # Combine: 21 + 1 + 4 + 1 = 27 features per node
        feat = aa_feat + plddt_feat + torsion_feat + seq_pos
        features.append(feat)

    return torch.tensor(features, dtype=torch.float32)


def build_vector_features(residues: dict, pocket_keys: list) -> torch.Tensor:
    """
    Build per-node vector features [N, 1, 3] — Cα→Cβ direction.
    These are equivariant under rotation (used by GVP-GNN).
    """
    vectors = []
    for key in pocket_keys:
        res = residues[key]
        ca = res["ca"]
        cb = res["cb"] if res["cb"] is not None else ca + np.array([0, 0, 1.5])
        direction = cb - ca
        norm = np.linalg.norm(direction) + 1e-8
        direction = direction / norm
        vectors.append(direction)
    # Shape: [N, 1, 3] (1 vector per node)
    return torch.tensor(np.stack(vectors)[:, None, :], dtype=torch.float32)


def build_knn_edges(
    residues: dict,
    pocket_keys: list,
    k: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build k-NN edge index on Cα positions.
    Returns: edge_index [2, E], edge_scalar [E, 5], edge_vector [E, 1, 3]
    """
    n = len(pocket_keys)
    ca_coords = np.stack([residues[k]["ca"] for k in pocket_keys])  # [N, 3]

    # Pairwise distances
    diff = ca_coords[:, None, :] - ca_coords[None, :, :]  # [N, N, 3]
    dists = np.linalg.norm(diff, axis=-1)  # [N, N]
    np.fill_diagonal(dists, np.inf)

    # k-NN
    src_list, dst_list = [], []
    for i in range(n):
        neighbors = np.argsort(dists[i])[:k]
        for j in neighbors:
            if dists[i, j] < np.inf:
                src_list.append(i)
                dst_list.append(j)

    src = np.array(src_list)
    dst = np.array(dst_list)
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)

    # Edge scalar features: [dist, sin_dist, seq_sep_norm, same_chain, placeholder]
    edge_scalars = []
    edge_vectors = []
    for s, d in zip(src_list, dst_list):
        dist = float(dists[s, d])
        seq_sep = abs(pocket_keys[s][1] - pocket_keys[d][1])
        same_chain = float(pocket_keys[s][0] == pocket_keys[d][0])

        direction = ca_coords[d] - ca_coords[s]
        norm = np.linalg.norm(direction) + 1e-8
        unit = direction / norm

        scalar = [
            dist / 20.0,            # normalized distance
            math.sin(dist / 5.0),   # oscillating distance encoding
            math.cos(dist / 5.0),
            min(seq_sep, 50) / 50.0, # normalized sequence separation
            same_chain,
        ]
        edge_scalars.append(scalar)
        edge_vectors.append(unit)

    edge_scalar_tensor = torch.tensor(edge_scalars, dtype=torch.float32)
    edge_vector_tensor = torch.tensor(
        np.stack(edge_vectors)[:, None, :], dtype=torch.float32
    )

    return edge_index, edge_scalar_tensor, edge_vector_tensor


def pdb_to_graph(
    pdb_path: str,
    uniprot_id: str,
    pocket_centroid: Optional[np.ndarray] = None,
    k: int = 10,
    cutoff: float = 15.0,
    top_n: int = 80,
    min_plddt: float = 0.4,
) -> Optional[Data]:
    """
    Full pipeline: PDB file → PyG Data object.
    
    Returns PyG Data with:
        x:              [N, 27] node scalar features
        v:              [N, 1, 3] node vector features (Cα→Cβ)
        edge_index:     [2, E]
        edge_attr:      [E, 5] edge scalar features  
        edge_vec:       [E, 1, 3] edge vector features
        pos:            [N, 3] Cα positions
        uniprot_id:     str
        n_residues:     int
    """
    if not os.path.exists(pdb_path):
        return None

    try:
        residues = parse_pdb_residues(pdb_path)
    except Exception as e:
        print(f"  [GRAPH] Failed to parse {pdb_path}: {e}")
        return None

    if len(residues) < 10:
        return None

    pocket_keys = get_pocket_residues(
        residues, pocket_centroid=pocket_centroid,
        cutoff=cutoff, min_plddt=min_plddt, top_n=top_n
    )

    if len(pocket_keys) < 5:
        print(f"  [GRAPH] Too few pocket residues ({len(pocket_keys)}) for {uniprot_id}")
        return None

    x = build_node_features(residues, pocket_keys)           # [N, 27]
    v = build_vector_features(residues, pocket_keys)          # [N, 1, 3]
    edge_index, edge_attr, edge_vec = build_knn_edges(
        residues, pocket_keys, k=k
    )

    pos = torch.tensor(
        np.stack([residues[k]["ca"] for k in pocket_keys]),
        dtype=torch.float32
    )

    graph = Data(
        x=x,
        v=v,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_vec=edge_vec,
        pos=pos,
        uniprot_id=uniprot_id,
        n_residues=len(pocket_keys),
    )
    return graph


def precompute_protein_graphs(
    target_map_df,
    graphs_dir: str = "data/graphs",
    k: int = 10,
    cutoff: float = 15.0,
) -> dict:
    """
    Precompute and cache pocket graphs for all proteins.
    Returns dict: uniprot_id → PyG Data
    """
    os.makedirs(graphs_dir, exist_ok=True)
    protein_graphs = {}

    all_uniprots = set()
    for _, row in target_map_df.iterrows():
        if pd.isna(row["af2_paths"]) or not row["af2_paths"]:
            continue
        for pdb_path, uid in zip(
            str(row["af2_paths"]).split(";"),
            str(row["uniprot_ids"]).split(";")
        ):
            all_uniprots.add((uid.strip(), pdb_path.strip()))

    import pandas as pd
    print(f"[GRAPHS] Building pocket graphs for {len(all_uniprots)} proteins ...")
    for uniprot_id, pdb_path in tqdm(all_uniprots):
        cache_path = os.path.join(graphs_dir, f"{uniprot_id}.pt")
        if os.path.exists(cache_path):
            graph = torch.load(cache_path, weights_only=False)
            protein_graphs[uniprot_id] = graph
            continue

        graph = pdb_to_graph(pdb_path, uniprot_id, k=k, cutoff=cutoff)
        if graph is not None:
            torch.save(graph, cache_path)
            protein_graphs[uniprot_id] = graph

    print(f"[GRAPHS] Built {len(protein_graphs)} protein graphs")
    return protein_graphs


if __name__ == "__main__":
    # Quick test on a single PDB
    import sys
    if len(sys.argv) > 1:
        pdb_path = sys.argv[1]
        graph = pdb_to_graph(pdb_path, "TEST")
        if graph:
            print(f"Graph: {graph.n_residues} nodes, {graph.edge_index.shape[1]} edges")
            print(f"  x: {graph.x.shape}")
            print(f"  v: {graph.v.shape}")
            print(f"  edge_attr: {graph.edge_attr.shape}")
            print(f"  pos: {graph.pos.shape}")
