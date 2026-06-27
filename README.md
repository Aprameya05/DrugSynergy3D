# DrugSynergy3D

**Multimodal Drug Combination Synergy Prediction via 3D Protein Interaction Geometry**

> *The first model to use SE(3)-equivariant geometric deep learning on AlphaFold2 binding pocket graphs for drug synergy prediction.*

---

## What Makes This Novel

Every existing drug synergy predictor (DeepSynergy, AuDNNsynergy, SynergyX, MatchMaker, TranSynergy) treats drugs as flat molecular fingerprints or SMILES strings. **None of them use the 3D geometry of the protein targets the drugs actually bind to.**

DrugSynergy3D changes this:

| What we do | Why it matters |
|---|---|
| AlphaFold2 binding pocket graphs | Real 3D structure, not sequence |
| SE(3)-equivariant GVP-GNN | Predictions invariant to rotation — physically correct |
| Cross-attention between pocket geometries | Captures structural complementarity between binding sites |
| STRING DB PPI cross-edges | Biological context: do the two targets interact? |
| NCI ALMANAC (300k+ observations) | Large-scale, clinically relevant training signal |

---

## Architecture

```
Drug A SMILES  ──→ DrugEncoder (GATv2)    ──→ drug_a_emb [B, 256]
Drug A Target  ──→ GVPEncoder (SE3-GNN)   ──→ prot_a_emb [B, 256]
Drug B SMILES  ──→ DrugEncoder (GATv2)    ──→ drug_b_emb [B, 256]
Drug B Target  ──→ GVPEncoder (SE3-GNN)   ──→ prot_b_emb [B, 256]
STRING DB PPI  ──→ MLP                    ──→ ppi_emb    [B, 64]

prot_a_emb ──┐
             ├──→ CrossAttention ──→ pocket_fused [B, 256]
prot_b_emb ──┘

drug_a_emb ──┐
             ├──→ (a+b, a*b, |a-b|) → MLP ──→ drug_pair [B, 256]
drug_b_emb ──┘

[pocket_fused | drug_pair | ppi_emb] ──→ SynergyHead (MLP) ──→ score [B]
```

---

## Setup

```bash
# Create environment
conda create -n drugsynergy python=3.10
conda activate drugsynergy

# Install PyTorch with CUDA
pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu118

# Install PyG
pip install torch-geometric torch-scatter torch-sparse torch-cluster \
  -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Install remaining dependencies
pip install -r requirements.txt
```

---

## Run

```bash
# Full training run (downloads data automatically)
python train.py

# With custom config
python train.py --config configs/default.yaml --epochs 100 --batch_size 32

# Without wandb
python train.py --no_wandb

# If data already downloaded
python train.py --skip_download
```

---

## Data

| Dataset | Source | Size |
|---|---|---|
| NCI ALMANAC | NCI Wiki | ~300k drug pair × cell line observations |
| AlphaFold2 structures | EBI AlphaFold DB | Per-target, fetched automatically |
| STRING DB v12 | STRING DB | 11M human PPIs |

---

## Results (Expected)

| Metric | Baseline (DeepSynergy) | DrugSynergy3D |
|---|---|---|
| Pearson r | ~0.73 | >0.82 (target) |
| Spearman r | ~0.70 | >0.79 (target) |
| AUROC | ~0.85 | >0.90 (target) |

---

## Publication Targets

- **Primary**: Journal of Chemical Information and Modeling (ACS)
- **Conference**: ICML 2026 CompBio Workshop / NeurIPS 2026 workshop
- **Stretch**: Nature Cancer (with wet lab validation from biotech friend)

---

## Citation

```bibtex
@article{drugsynergy3d2025,
  title={DrugSynergy3D: Multimodal Drug Combination Synergy Prediction 
         via 3D Protein Interaction Geometry},
  author={Aprameya et al.},
  year={2025}
}
```

---

## Project Structure

```
DrugSynergy3D/
├── configs/
│   └── default.yaml          # Full experiment config
├── src/
│   ├── data/
│   │   ├── almanac.py        # NCI ALMANAC downloader + parser
│   │   ├── target_mapper.py  # Drug → UniProt → AF2 structure
│   │   ├── pocket_graph.py   # AF2 PDB → geometric residue graph
│   │   ├── ppi_network.py    # STRING DB PPI network
│   │   └── dataset.py        # PyTorch Dataset + DataLoader
│   ├── models/
│   │   ├── gvp_gnn.py        # SE(3)-equivariant GVP-GNN encoder
│   │   ├── drug_encoder.py   # GATv2 drug molecular encoder
│   │   └── synergy_model.py  # Full DrugSynergy3D model
│   └── training/
│       └── trainer.py        # Training loop, metrics, checkpointing
├── train.py                  # Main entry point
├── requirements.txt
└── README.md
```
