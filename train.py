"""
train.py — DrugSynergy3D main entry point.

Runs the complete pipeline:
  1. Download NCI ALMANAC
  2. Map drugs to UniProt targets → download AF2 structures
  3. Build protein pocket graphs (GVP-ready)
  4. Download STRING DB PPI network
  5. Build drug molecular graphs
  6. Train DrugSynergy3D model
  7. Evaluate and save results

Usage:
  python train.py                          # full run
  python train.py --config configs/default.yaml
  python train.py --epochs 50 --batch_size 16
  python train.py --skip_download          # if data already downloaded
"""

import os
import sys
import argparse
import yaml
import torch
import random
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from src.data.almanac import download_almanac, parse_almanac, get_unique_drugs
from src.data.target_mapper import build_drug_target_map
from src.data.pocket_graph import precompute_protein_graphs
from src.data.ppi_network import download_string_db, load_ppi_network
from src.data.dataset import get_dataloaders
from src.models.synergy_model import DrugSynergy3D
from src.training.trainer import Trainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--skip_download", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    # CLI overrides
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["training"]["lr"] = args.lr

    device = args.device or cfg["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU")
        device = "cpu"

    set_seed(cfg["project"]["seed"])

    print(f"\n{'='*60}")
    print(f" DrugSynergy3D — 3D Structure-Aware Drug Synergy Prediction")
    print(f" Device: {device}")
    if device == "cuda":
        print(f" GPU: {torch.cuda.get_device_name(0)}")
        print(f" VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"{'='*60}\n")

    # ── Step 1: Download & parse ALMANAC ─────────────────────────────
    print("Step 1/6: NCI ALMANAC data")
    csv_path = download_almanac(cfg["data"]["raw_dir"])
    almanac_df = parse_almanac(
        csv_path,
        processed_dir=cfg["data"]["processed_dir"],
        synergy_threshold=cfg["evaluation"]["synergy_threshold"],
    )
    drugs_df = get_unique_drugs(almanac_df)
    print(f"  {len(almanac_df)} drug pair observations, {len(drugs_df)} unique drugs\n")

    # ── Step 2: Map drugs to AF2 structures ──────────────────────────
    print("Step 2/6: Drug target mapping + AF2 structure download")
    target_map = build_drug_target_map(
        drugs_df,
        structures_dir=cfg["data"]["structures_dir"],
        processed_dir=cfg["data"]["processed_dir"],
    )
    print(f"  {(target_map['n_structures'] > 0).sum()} drugs with structures\n")

    # ── Step 3: Build protein pocket graphs ──────────────────────────
    print("Step 3/6: Building protein pocket graphs (GVP-ready)")
    protein_graphs = precompute_protein_graphs(
        target_map,
        graphs_dir=cfg["data"]["graphs_dir"],
        k=cfg["data"]["pocket_k_neighbors"],
        cutoff=cfg["data"]["pocket_distance_cutoff"],
    )
    print(f"  {len(protein_graphs)} protein graphs built\n")

    # ── Step 4: STRING DB PPI network ────────────────────────────────
    print("Step 4/6: Loading STRING DB PPI network")
    try:
        string_path = download_string_db(cfg["data"]["raw_dir"])
        ppi_df = load_ppi_network(
            string_path,
            score_threshold=cfg["data"]["cross_protein_score_thresh"],
            processed_dir=cfg["data"]["processed_dir"],
        )
        print(f"  {len(ppi_df)} high-confidence PPIs loaded\n")
    except Exception as e:
        print(f"  [WARN] STRING DB failed ({e}), running without PPI features\n")
        ppi_df = None

    # ── Step 5: Build DataLoaders ─────────────────────────────────────
    print("Step 5/6: Building DataLoaders")
    train_loader, val_loader, test_loader = get_dataloaders(
        almanac_df=almanac_df,
        target_map=target_map,
        protein_graphs=protein_graphs,
        ppi_df=ppi_df,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
        seed=cfg["project"]["seed"],
    )
    print(f"  Train: {len(train_loader.dataset)} | "
          f"Val: {len(val_loader.dataset)} | "
          f"Test: {len(test_loader.dataset)}\n")

    # ── Step 6: Build Model ───────────────────────────────────────────
    print("Step 6/6: Initializing DrugSynergy3D model")
    model = DrugSynergy3D(
        gvp_node_dims=(27, 1),
        gvp_edge_dims=(5, 1),
        gvp_hidden_dims=tuple(cfg["model"]["gvp"]["node_dims"]),
        gvp_out_dim=cfg["model"]["fusion"]["d_model"],
        gvp_num_layers=cfg["model"]["gvp"]["num_layers"],
        gvp_drop_rate=cfg["model"]["gvp"]["drop_rate"],
        drug_hidden_dim=cfg["model"]["drug_gnn"]["hidden_dim"],
        drug_out_dim=cfg["model"]["fusion"]["d_model"],
        drug_num_layers=cfg["model"]["drug_gnn"]["num_layers"],
        drug_heads=cfg["model"]["drug_gnn"]["heads"],
        fusion_d_model=cfg["model"]["fusion"]["d_model"],
        fusion_heads=cfg["model"]["fusion"]["n_heads"],
        fusion_dropout=cfg["model"]["fusion"]["dropout"],
        head_hidden=cfg["model"]["head"]["hidden_dims"],
        head_dropout=cfg["model"]["head"]["dropout"],
    )

    counts = model.count_parameters()
    print(f"  Total parameters: {counts['total']:,}")
    print(f"  Pocket encoder:   {counts['pocket_encoder']:,}")
    print(f"  Drug encoder:     {counts['drug_encoder']:,}")
    print(f"  Fusion:           {counts['fusion']:,}")
    print(f"  Head:             {counts['head']:,}\n")

    # ── Training ──────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=cfg,
        device=device,
        use_wandb=not args.no_wandb,
    )

    results = trainer.train()

    print("\nDone! Results saved to outputs/results/test_results.json")
    print("Best checkpoint saved to outputs/checkpoints/best_model.pt")


if __name__ == "__main__":
    main()
