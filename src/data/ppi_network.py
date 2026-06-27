"""
src/data/ppi_network.py

Downloads STRING DB human PPI network and builds cross-protein interaction features.
These become the "cross-graph edges" between Drug A's target and Drug B's target —
the core novel feature of DrugSynergy3D.

Cross-edge features (when two drug targets interact in STRING DB):
  - STRING combined score (normalized)
  - Co-expression score
  - Experimental evidence score  
  - Text-mining score
  - Neighborhood score
  - Co-occurrence score
"""

import os
import gzip
import requests
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm


STRING_URL = (
    "https://stringdb-downloads.org/download/protein.links.detailed.v12.0/"
    "9606.protein.links.detailed.v12.0.txt.gz"
)
STRING_INFO_URL = (
    "https://stringdb-downloads.org/download/protein.info.v12.0/"
    "9606.protein.info.v12.0.txt.gz"
)

# STRING uses ENSP IDs; we need UniProt → ENSP mapping
UNIPROT_TO_ENSP = {
    # Curated mapping for our drug targets
    "P04637": "9606.ENSP00000269305",   # TP53
    "P11388": "9606.ENSP00000361909",   # TOP2A
    "P00390": "9606.ENSP00000288986",   # GSTP1 (proxy)
    "P07437": "9606.ENSP00000301099",   # TUBB
    "P23921": "9606.ENSP00000265148",   # RRM1
    "P00533": "9606.ENSP00000275493",   # EGFR
    "P15056": "9606.ENSP00000288602",   # BRAF
    "P00519": "9606.ENSP00000361423",   # ABL1
    "P35968": "9606.ENSP00000263923",   # KDR
    "P04626": "9606.ENSP00000269571",   # ERBB2
    "Q02750": "9606.ENSP00000250007",   # MAP2K1
    "Q00534": "9606.ENSP00000265734",   # CDK6
    "Q07812": "9606.ENSP00000313521",   # BCL2
    "P11387": "9606.ENSP00000361337",   # TOP1
    "P31350": "9606.ENSP00000357940",   # RRM2
    "P00374": "9606.ENSP00000228928",   # DHFR
    "P04818": "9606.ENSP00000233655",   # TYMS
    "P08238": "9606.ENSP00000340189",   # HSP90AB1
}


def download_string_db(raw_dir: str = "data/raw") -> str:
    """Download STRING DB detailed links. Returns path to txt file."""
    os.makedirs(raw_dir, exist_ok=True)
    gz_path = os.path.join(raw_dir, "string_links.txt.gz")
    txt_path = os.path.join(raw_dir, "string_links.txt")

    if os.path.exists(txt_path):
        print(f"[STRING] Already downloaded: {txt_path}")
        return txt_path

    print("[STRING] Downloading STRING DB human PPI (may take a few minutes)...")
    r = requests.get(STRING_URL, stream=True, timeout=300)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))

    with open(gz_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc="string_db.gz"
    ) as bar:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)
            bar.update(len(chunk))

    print("[STRING] Decompressing...")
    with gzip.open(gz_path, "rt") as gz, open(txt_path, "w") as out:
        for line in tqdm(gz, desc="decompress"):
            out.write(line)

    return txt_path


def load_ppi_network(
    txt_path: str,
    score_threshold: int = 400,
    processed_dir: str = "data/processed",
) -> pd.DataFrame:
    """
    Load and filter STRING DB PPI network.
    Returns DataFrame: protein1, protein2, combined_score, [channel scores]
    """
    out_path = os.path.join(processed_dir, "ppi_filtered.csv")
    if os.path.exists(out_path):
        print(f"[STRING] Loading cached: {out_path}")
        return pd.read_csv(out_path)

    print(f"[STRING] Loading STRING DB from {txt_path} ...")
    df = pd.read_csv(txt_path, sep=" ")
    print(f"  Raw interactions: {len(df)}")

    # Filter by combined score
    if "combined_score" in df.columns:
        df = df[df["combined_score"] >= score_threshold]
    print(f"  After score≥{score_threshold}: {len(df)}")

    os.makedirs(processed_dir, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[STRING] Saved → {out_path}")
    return df


def get_ppi_edge_features(
    uniprot_a: str,
    uniprot_b: str,
    ppi_df: pd.DataFrame,
) -> torch.Tensor:
    """
    Look up STRING interaction features between two UniProt IDs.
    Returns tensor of shape [6] (one per STRING channel) or zeros if no interaction.
    """
    ensp_a = UNIPROT_TO_ENSP.get(uniprot_a)
    ensp_b = UNIPROT_TO_ENSP.get(uniprot_b)

    if ensp_a is None or ensp_b is None:
        return torch.zeros(6, dtype=torch.float32)

    cols = ["neighborhood", "fusion", "cooccurence",
            "coexpression", "experimental", "combined_score"]

    # Check both orientations
    mask = (
        ((ppi_df["protein1"] == ensp_a) & (ppi_df["protein2"] == ensp_b)) |
        ((ppi_df["protein1"] == ensp_b) & (ppi_df["protein2"] == ensp_a))
    )
    matches = ppi_df[mask]

    if len(matches) == 0:
        return torch.zeros(6, dtype=torch.float32)

    row = matches.iloc[0]
    feats = []
    for col in cols:
        if col in row:
            feats.append(float(row[col]) / 1000.0)  # STRING uses 0-1000 scale
        else:
            feats.append(0.0)

    return torch.tensor(feats, dtype=torch.float32)


def build_ppi_matrix(
    all_uniprot_ids: list,
    ppi_df: pd.DataFrame,
) -> dict:
    """
    Build lookup dict: (uniprot_a, uniprot_b) → interaction_features tensor.
    Pre-computes all pairs for fast lookup during training.
    """
    ppi_cache = {}
    n = len(all_uniprot_ids)
    print(f"[STRING] Building PPI feature cache for {n} proteins ({n*n//2} pairs) ...")

    for i in range(n):
        for j in range(i+1, n):
            ua, ub = all_uniprot_ids[i], all_uniprot_ids[j]
            feats = get_ppi_edge_features(ua, ub, ppi_df)
            ppi_cache[(ua, ub)] = feats
            ppi_cache[(ub, ua)] = feats  # symmetric

    return ppi_cache


if __name__ == "__main__":
    txt_path = download_string_db()
    ppi_df = load_ppi_network(txt_path)

    # Test lookup for EGFR-ERBB2 (known interaction)
    feat = get_ppi_edge_features("P00533", "P04626", ppi_df)
    print(f"EGFR-ERBB2 PPI features: {feat}")
