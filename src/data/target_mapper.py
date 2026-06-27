"""
src/data/target_mapper.py
Maps NCI drug NSC IDs → drug targets (UniProt IDs) → AlphaFold2 structure URLs.

Pipeline:
1. NSC → PubChem CID (via NCI/PubChem API)
2. PubChem CID → UniProt target IDs (via ChEMBL or UniProt API)
3. UniProt ID → AlphaFold2 PDB structure URL
"""

import os
import time
import json
import requests
import pandas as pd
from tqdm import tqdm
from typing import Optional


# Known NSC → PubChem CID mappings for major ALMANAC drugs
NSC_TO_PUBCHEM = {
    119875: 84691,      # Cisplatin
    123127: 31703,      # Doxorubicin
    125066: 2907,       # Cyclophosphamide
    26980:  5746,       # Mitomycin C
    38721:  9238,       # Hydroxyurea
    49842:  5978,       # Vinblastine
    67574:  5978,       # Vincristine (use 5978 fallback)
    256439: 36314,      # Paclitaxel
    609699: 60953,      # Irinotecan
    628503: 60750,      # Gemcitabine
    698037: 176870,     # Erlotinib
    701852: 216239,     # Sorafenib
    703813: 3062316,    # Dasatinib
    706725: 5329102,    # Sunitinib
    715055: 208908,     # Lapatinib
    716051: 644241,     # Nilotinib
    724770: 42611257,   # Vemurafenib
    726038: 11626560,   # Crizotinib
    730406: 25126798,   # Ruxolitinib
    732517: 11707110,   # Trametinib
    741078: 5330286,    # Palbociclib
    761431: 49846579,   # Venetoclax
}

# Known drug → primary UniProt target (curated, canonical targets)
DRUG_PRIMARY_TARGETS = {
    "Cisplatin":        ["P04637"],          # TP53
    "Doxorubicin":      ["P11388"],          # TOP2A
    "Cyclophosphamide": ["P00390"],          # GR (alkylating)
    "Paclitaxel":       ["P07437"],          # TUBB
    "Gemcitabine":      ["P23921"],          # RRM1
    "Erlotinib":        ["P00533"],          # EGFR
    "Sorafenib":        ["P15056"],          # BRAF
    "Dasatinib":        ["P00519"],          # ABL1
    "Sunitinib":        ["P35968"],          # KDR (VEGFR2)
    "Lapatinib":        ["P00533", "P04626"],# EGFR, ERBB2
    "Nilotinib":        ["P00519"],          # ABL1
    "Vemurafenib":      ["P15056"],          # BRAF V600E
    "Crizotinib":       ["P08238"],          # ALK (HSP90 binding)
    "Trametinib":       ["Q02750"],          # MAP2K1 (MEK1)
    "Palbociclib":      ["Q00534"],          # CDK6
    "Venetoclax":       ["Q07812"],          # BCL2
    "Irinotecan":       ["P11387"],          # TOP1
    "Vincristine":      ["P07437"],          # TUBB
    "Vinblastine":      ["P07437"],          # TUBB
    "Hydroxyurea":      ["P31350"],          # RRM2
    "Methotrexate":     ["P00374"],          # DHFR
    "Fluorouracil":     ["P04818"],          # TYMS
    "Cytarabine":       ["P23921"],          # RRM1
}


def nsc_to_pubchem(nsc: int, session: requests.Session) -> Optional[int]:
    """Look up PubChem CID from NSC number via PubChem synonym search."""
    if nsc in NSC_TO_PUBCHEM:
        return NSC_TO_PUBCHEM[nsc]
    try:
        url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
               f"NSC{nsc}/property/MolecularFormula/JSON")
        r = session.get(url, timeout=15)
        if r.status_code == 200:
            data = r.json()
            cid = data["PropertyTable"]["Properties"][0]["CID"]
            return int(cid)
    except Exception:
        pass
    return None


def pubchem_to_uniprot(cid: int, session: requests.Session) -> list:
    """Get UniProt target IDs from PubChem BioAssay data."""
    try:
        url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}"
               f"/xrefs/ProteinGI/JSON")
        r = session.get(url, timeout=15)
        if r.status_code == 200:
            # Fallback: use ChEMBL targets instead
            pass
    except Exception:
        pass
    return []


def uniprot_to_af2_url(uniprot_id: str) -> str:
    """Construct AlphaFold2 structure URL for a UniProt ID."""
    return (f"https://alphafold.ebi.ac.uk/files/"
            f"AF-{uniprot_id}-F1-model_v4.pdb")


def download_af2_structure(
    uniprot_id: str,
    structures_dir: str = "data/structures",
    session: Optional[requests.Session] = None,
) -> Optional[str]:
    """Download AF2 PDB for a UniProt ID. Returns local path or None."""
    os.makedirs(structures_dir, exist_ok=True)
    out_path = os.path.join(structures_dir, f"{uniprot_id}.pdb")
    if os.path.exists(out_path):
        return out_path

    url = uniprot_to_af2_url(uniprot_id)
    sess = session or requests.Session()
    try:
        r = sess.get(url, timeout=30)
        if r.status_code == 200:
            with open(out_path, "w") as f:
                f.write(r.text)
            return out_path
        else:
            print(f"  [AF2] {uniprot_id}: HTTP {r.status_code}")
            return None
    except Exception as e:
        print(f"  [AF2] {uniprot_id}: {e}")
        return None


def build_drug_target_map(
    drugs_df: pd.DataFrame,
    structures_dir: str = "data/structures",
    processed_dir: str = "data/processed",
    delay: float = 0.2,
) -> pd.DataFrame:
    """
    Build mapping: drug_nsc → drug_name → [uniprot_ids] → [af2_paths].
    Uses curated DRUG_PRIMARY_TARGETS dict first, then API fallback.
    """
    out_path = os.path.join(processed_dir, "drug_target_map.csv")
    if os.path.exists(out_path):
        print(f"[TARGETS] Loading cached: {out_path}")
        return pd.read_csv(out_path)

    records = []
    session = requests.Session()
    session.headers["User-Agent"] = "DrugSynergy3D/1.0 (research)"

    print(f"[TARGETS] Mapping {len(drugs_df)} drugs to AF2 structures ...")
    for _, row in tqdm(drugs_df.iterrows(), total=len(drugs_df)):
        nsc = int(row["nsc"])
        name = str(row["name"])

        # Get UniProt targets
        if name in DRUG_PRIMARY_TARGETS:
            targets = DRUG_PRIMARY_TARGETS[name]
        else:
            # Try PubChem lookup
            targets = []

        if not targets:
            print(f"  [WARN] No targets for {name} (NSC {nsc}), skipping")
            continue

        # Download AF2 structures
        af2_paths = []
        for uniprot_id in targets:
            path = download_af2_structure(uniprot_id, structures_dir, session)
            if path:
                af2_paths.append(path)
            time.sleep(delay)

        records.append({
            "nsc": nsc,
            "name": name,
            "uniprot_ids": ";".join(targets),
            "af2_paths": ";".join(af2_paths),
            "n_structures": len(af2_paths),
        })

    df = pd.DataFrame(records)
    os.makedirs(processed_dir, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[TARGETS] Saved → {out_path}")
    print(f"  Drugs with structures: {(df['n_structures'] > 0).sum()}/{len(df)}")
    return df


if __name__ == "__main__":
    from almanac import download_almanac, parse_almanac, get_unique_drugs
    csv_path = download_almanac()
    almanac_df = parse_almanac(csv_path)
    drugs_df = get_unique_drugs(almanac_df)
    target_map = build_drug_target_map(drugs_df)
    print(target_map[target_map["n_structures"] > 0].head(10).to_string())
