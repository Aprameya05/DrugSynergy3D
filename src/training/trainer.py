"""
src/training/trainer.py

Full training loop for DrugSynergy3D.
Features:
  - MSE + Pearson correlation loss (Pearson as auxiliary)
  - Cosine annealing with warmup
  - Gradient clipping
  - Best model checkpointing
  - wandb logging
  - Early stopping
"""

import os
import time
import math
import json
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
from typing import Optional

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def pearson_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Differentiable Pearson correlation loss (1 - r).
    Encourages the model to rank synergy scores correctly.
    """
    pred_mean = pred.mean()
    target_mean = target.mean()
    pred_c = pred - pred_mean
    target_c = target - target_mean
    num = (pred_c * target_c).sum()
    den = torch.sqrt((pred_c ** 2).sum() * (target_c ** 2).sum() + 1e-8)
    r = num / den
    return 1.0 - r


class WarmupCosineScheduler:
    """Linear warmup followed by cosine annealing."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        base_lr: float,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr

    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


def compute_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    labels: np.ndarray,
    threshold: float = 10.0,
) -> dict:
    """Compute all evaluation metrics."""
    preds = np.array(preds)
    targets = np.array(targets)
    labels = np.array(labels)

    mse = float(np.mean((preds - targets) ** 2))
    rmse = float(np.sqrt(mse))

    try:
        pr, _ = pearsonr(preds, targets)
    except Exception:
        pr = 0.0

    try:
        sr, _ = spearmanr(preds, targets)
    except Exception:
        sr = 0.0

    # Convert continuous predictions to probabilities for classification metrics
    pred_probs = torch.sigmoid(torch.tensor(preds / threshold)).numpy()

    try:
        auroc = float(roc_auc_score(labels, pred_probs))
    except Exception:
        auroc = 0.5

    try:
        auprc = float(average_precision_score(labels, pred_probs))
    except Exception:
        auprc = 0.0

    return {
        "mse": mse,
        "rmse": rmse,
        "pearson_r": float(pr),
        "spearman_r": float(sr),
        "auroc": auroc,
        "auprc": auprc,
    }


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        config: dict,
        device: str = "cuda",
        use_wandb: bool = True,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"],
        )

        # Scheduler
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=config["training"]["warmup_epochs"],
            max_epochs=config["training"]["epochs"],
            base_lr=config["training"]["lr"],
        )

        self.pearson_weight = config["training"]["pearson_weight"]
        self.grad_clip = config["training"]["gradient_clip"]
        self.checkpoint_dir = config["training"]["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.best_val_pearson = -1.0
        self.best_epoch = 0

        # Wandb
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project="DrugSynergy3D",
                config=config,
                name=f"run_{int(time.time())}",
            )
            wandb.watch(model, log_freq=100)

    def _move_batch(self, batch: dict) -> dict:
        """Move batch tensors to device."""
        tensor_keys = [
            "pa_xs", "pa_xv", "pa_ei", "pa_es", "pa_ev", "pa_batch",
            "pb_xs", "pb_xv", "pb_ei", "pb_es", "pb_ev", "pb_batch",
            "da_x", "da_ei", "da_ea", "da_batch",
            "db_x", "db_ei", "db_ea", "db_batch",
            "ppi_feats", "css_score", "synergy_label",
        ]
        moved = {}
        for k, v in batch.items():
            if k in tensor_keys and isinstance(v, torch.Tensor):
                moved[k] = v.to(self.device)
            else:
                moved[k] = v
        return moved

    def _forward(self, batch: dict) -> torch.Tensor:
        """Forward pass returning predictions [B]."""
        return self.model(
            batch["pa_xs"], batch["pa_xv"], batch["pa_ei"],
            batch["pa_es"], batch["pa_ev"], batch["pa_batch"],
            batch["pb_xs"], batch["pb_xv"], batch["pb_ei"],
            batch["pb_es"], batch["pb_ev"], batch["pb_batch"],
            batch["da_x"], batch["da_ei"], batch["da_ea"], batch["da_batch"],
            batch["db_x"], batch["db_ei"], batch["db_ea"], batch["db_batch"],
            batch["ppi_feats"],
        )

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        lr = self.scheduler.step(epoch)

        total_loss = 0.0
        all_preds, all_targets, all_labels = [], [], []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        for batch in pbar:
            batch = self._move_batch(batch)
            self.optimizer.zero_grad()

            preds = self._forward(batch)           # [B]
            targets = batch["css_score"]           # [B]

            # Loss: MSE + Pearson correlation
            mse = nn.functional.mse_loss(preds, targets)
            p_loss = pearson_loss(preds, targets) if len(preds) > 2 else torch.tensor(0.0)
            loss = mse + self.pearson_weight * p_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            all_preds.extend(preds.detach().cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_labels.extend(batch["synergy_label"].cpu().numpy())

            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.6f}")

        metrics = compute_metrics(all_preds, all_targets, all_labels)
        metrics["loss"] = total_loss / len(self.train_loader)
        metrics["lr"] = lr
        return metrics

    @torch.no_grad()
    def eval_epoch(self, loader, split: str = "val") -> dict:
        self.model.eval()
        all_preds, all_targets, all_labels = [], [], []
        total_loss = 0.0

        for batch in tqdm(loader, desc=f"[{split}]", leave=False):
            batch = self._move_batch(batch)
            preds = self._forward(batch)
            targets = batch["css_score"]

            mse = nn.functional.mse_loss(preds, targets)
            total_loss += mse.item()

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_labels.extend(batch["synergy_label"].cpu().numpy())

        metrics = compute_metrics(all_preds, all_targets, all_labels)
        metrics["loss"] = total_loss / len(loader)
        return metrics

    def save_checkpoint(self, epoch: int, metrics: dict, tag: str = "best"):
        path = os.path.join(self.checkpoint_dir, f"{tag}_model.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }, path)
        print(f"  [Checkpoint] Saved {tag} → {path}")

    def train(self, epochs: Optional[int] = None) -> dict:
        n_epochs = epochs or self.config["training"]["epochs"]
        print(f"\n{'='*60}")
        print(f"DrugSynergy3D Training: {n_epochs} epochs on {self.device}")
        print(f"{'='*60}\n")

        history = {"train": [], "val": []}

        for epoch in range(n_epochs):
            t0 = time.time()

            # Train
            train_metrics = self.train_epoch(epoch)
            history["train"].append(train_metrics)

            # Validate
            val_metrics = self.eval_epoch(self.val_loader, "val")
            history["val"].append(val_metrics)

            elapsed = time.time() - t0

            print(
                f"Epoch {epoch+1:03d}/{n_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train r: {train_metrics['pearson_r']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val r: {val_metrics['pearson_r']:.4f} | "
                f"Val AUROC: {val_metrics['auroc']:.4f} | "
                f"{elapsed:.1f}s"
            )

            # Checkpoint best model
            if val_metrics["pearson_r"] > self.best_val_pearson:
                self.best_val_pearson = val_metrics["pearson_r"]
                self.best_epoch = epoch
                self.save_checkpoint(epoch, val_metrics, "best")

            # Save latest
            self.save_checkpoint(epoch, val_metrics, "latest")

            # Wandb logging
            if self.use_wandb:
                log_dict = {
                    f"train/{k}": v for k, v in train_metrics.items()
                }
                log_dict.update({
                    f"val/{k}": v for k, v in val_metrics.items()
                })
                wandb.log(log_dict, step=epoch)

        # Final test evaluation
        print(f"\nLoading best model (epoch {self.best_epoch+1}) for test evaluation...")
        ckpt = torch.load(
            os.path.join(self.checkpoint_dir, "best_model.pt"),
            map_location=self.device,
            weights_only=False,
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        test_metrics = self.eval_epoch(self.test_loader, "test")

        print(f"\n{'='*60}")
        print("FINAL TEST RESULTS:")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")
        print(f"{'='*60}\n")

        # Save results
        results_dir = self.config["training"]["results_dir"]
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "test_results.json"), "w") as f:
            json.dump(test_metrics, f, indent=2)

        if self.use_wandb:
            wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
            wandb.finish()

        return {"train_history": history, "test_metrics": test_metrics}
