"""
src/data/almanac.py
Downloads and parses NCI ALMANAC drug combination synergy dataset.
Outputs a clean DataFrame with columns:
  drug1_name, drug2_name, cell_line, css_score, synergy_label
"""

import os
import io
import zipfile
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm


ALMANAC_URL = (
    "https://wiki.nci.nih.gov/download/attachments/338237347/"
    "ComboDrugGrowth_Nov2017.zip"
)

# NCI drug NSC -> common name mapping for top ALMANAC drugs
# We'll supplement with API lookup for the rest
NSC_TO_NAME = {
    119875:  "Cisplatin",
    123127:  "Doxorubicin",
    125066:  "Cyclophosphamide",
    26980:   "Mitomycin C",
    38721:   "Hydroxyurea",
    45388:   "Dacarbazine",
    49842:   "Vinblastine",
    67574:   "Vincristine",
    71423:   "Melphalan",
    752:     "Chlorambucil",
    757:     "Mechlorethamine",
    762:     "Thioguanine",
    3053:    "Methotrexate",
    8806:    "Fluorouracil",
    13875:   "Mercaptopurine",
    14229:   "Cytarabine",
    740:     "Azathioprine",
    256439:  "Paclitaxel",
    609699:  "Irinotecan",
    628503:  "Gemcitabine",
    698037:  "Erlotinib",
    701852:  "Sorafenib",
    703813:  "Dasatinib",
    706725:  "Sunitinib",
    715055:  "Lapatinib",
    716051:  "Nilotinib",
    718781:  "Temsirolimus",
    719276:  "Vorinostat",
    720568:  "Everolimus",
    724770:  "Vemurafenib",
    726038:  "Crizotinib",
    730406:  "Ruxolitinib",
    732517:  "Trametinib",
    733504:  "Dabrafenib",
    737664:  "Cobimetinib",
    741078:  "Palbociclib",
    743414:  "Osimertinib",
    745750:  "Ribociclib",
    747971:  "Abemaciclib",
    761431:  "Venetoclax",
}


def download_almanac(raw_dir: str = "data/raw") -> str:
    """Download NCI ALMANAC zip if not already present. Returns path to CSV."""
    os.makedirs(raw_dir, exist_ok=True)
    zip_path = os.path.join(raw_dir, "almanac.zip")
    csv_path = os.path.join(raw_dir, "ComboDrugGrowth_Nov2017.csv")

    if os.path.exists(csv_path):
        print(f"[ALMANAC] Already downloaded: {csv_path}")
        return csv_path

    print(f"[ALMANAC] Downloading from NCI...")
    r = requests.get(ALMANAC_URL, stream=True, timeout=120)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(zip_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc="almanac.zip"
    ) as bar:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    print("[ALMANAC] Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            if name.endswith(".csv"):
                z.extract(name, raw_dir)
                extracted = os.path.join(raw_dir, name)
                os.rename(extracted, csv_path)
                break

    return csv_path


def compute_css(row_group: pd.DataFrame) -> float:
    """
    Compute Combination Synergy Score (CSS) from growth inhibition curves.
    CSS = area between combo and single-agent dose-response curves.
    Simplified: use PERCENTGROWTHNOTZ as proxy when full curve unavailable.
    """
    # ALMANAC has PERCENTGROWTHNOTZ for each concentration combo
    # We aggregate: mean combo growth vs expected additive
    if "PERCENTGROWTHNOTZ" in row_group.columns:
        return float(row_group["PERCENTGROWTHNOTZ"].mean())
    return float("nan")


def parse_almanac(
    csv_path: str,
    processed_dir: str = "data/processed",
    min_pairs: int = 5,
    synergy_threshold: float = 10.0,
) -> pd.DataFrame:
    """
    Parse raw ALMANAC CSV into clean synergy DataFrame.
    
    Returns DataFrame with:
        drug1_nsc, drug2_nsc, drug1_name, drug2_name,
        cell_line, tissue, css_score, synergy_label
    """
    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, "almanac_clean.csv")

    if os.path.exists(out_path):
        print(f"[ALMANAC] Loading cached: {out_path}")
        return pd.read_csv(out_path)

    print(f"[ALMANAC] Parsing {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Raw shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    # Standardise column names (ALMANAC uses mixed case)
    df.columns = [c.strip().upper() for c in df.columns]

    # Core columns we need
    required = ["NSC1", "NSC2", "CELLNAME", "PERCENTGROWTHNOTZ"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Expected column {col} not found in ALMANAC. "
                             f"Available: {list(df.columns)}")

    # Aggregate per (drug1, drug2, cell_line) → mean growth inhibition
    print("[ALMANAC] Aggregating synergy scores per drug pair × cell line ...")
    grouped = (
        df.groupby(["NSC1", "NSC2", "CELLNAME"])["PERCENTGROWTHNOTZ"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped.columns = ["drug1_nsc", "drug2_nsc", "cell_line",
                       "css_mean", "css_std", "n_concentrations"]

    # Filter: require at least min_pairs concentration measurements
    grouped = grouped[grouped["n_concentrations"] >= min_pairs]
    print(f"  After min_pairs filter: {len(grouped)} pairs")

    # Add drug names
    grouped["drug1_name"] = grouped["drug1_nsc"].map(NSC_TO_NAME).fillna(
        grouped["drug1_nsc"].astype(str)
    )
    grouped["drug2_name"] = grouped["drug2_nsc"].map(NSC_TO_NAME).fillna(
        grouped["drug2_nsc"].astype(str)
    )

    # CSS score: lower growth inhibition of combo vs expected additive = synergy
    # We use css_mean as the primary signal (higher = more growth inhibition = synergistic)
    grouped["css_score"] = grouped["css_mean"]

    # Binary synergy label: synergy if combo inhibits growth beyond threshold
    grouped["synergy_label"] = (grouped["css_score"] > synergy_threshold).astype(int)

    # Tissue mapping from cell line (ALMANAC encodes this in PANEL column if available)
    if "PANEL" in df.columns:
        panel_map = df[["CELLNAME", "PANEL"]].drop_duplicates().set_index("CELLNAME")["PANEL"]
        grouped["tissue"] = grouped["cell_line"].map(panel_map).fillna("Unknown")
    else:
        grouped["tissue"] = "Unknown"

    # Canonicalise drug pair order (NSC1 < NSC2) to avoid duplicates
    mask = grouped["drug1_nsc"] > grouped["drug2_nsc"]
    grouped.loc[mask, ["drug1_nsc", "drug2_nsc"]] = (
        grouped.loc[mask, ["drug2_nsc", "drug1_nsc"]].values
    )
    grouped.loc[mask, ["drug1_name", "drug2_name"]] = (
        grouped.loc[mask, ["drug2_name", "drug1_name"]].values
    )
    grouped = grouped.drop_duplicates(subset=["drug1_nsc", "drug2_nsc", "cell_line"])

    print(f"  Final pairs: {len(grouped)}")
    print(f"  Unique drugs: {pd.concat([grouped['drug1_nsc'], grouped['drug2_nsc']]).nunique()}")
    print(f"  Synergistic pairs: {grouped['synergy_label'].sum()} "
          f"({100*grouped['synergy_label'].mean():.1f}%)")

    grouped.to_csv(out_path, index=False)
    print(f"[ALMANAC] Saved → {out_path}")
    return grouped


def get_unique_drugs(df: pd.DataFrame) -> pd.DataFrame:
    """Extract unique drugs with their NSC IDs."""
    d1 = df[["drug1_nsc", "drug1_name"]].rename(
        columns={"drug1_nsc": "nsc", "drug1_name": "name"}
    )
    d2 = df[["drug2_nsc", "drug2_name"]].rename(
        columns={"drug2_nsc": "nsc", "drug2_name": "name"}
    )
    drugs = pd.concat([d1, d2]).drop_duplicates(subset=["nsc"]).reset_index(drop=True)
    return drugs


if __name__ == "__main__":
    csv_path = download_almanac()
    df = parse_almanac(csv_path)
    drugs = get_unique_drugs(df)
    print(f"\nUnique drugs: {len(drugs)}")
    print(drugs.head(20).to_string())
