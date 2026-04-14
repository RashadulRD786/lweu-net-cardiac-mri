"""
Trainer for ACDC cardiac MRI segmentation.

Handles:
  - Full train/val loop with tqdm progress bars
  - Per-epoch metrics: Dice, IoU, Precision, Recall (per class + mean foreground)
  - ReduceLROnPlateau on val mean foreground Dice
  - Early stopping (patience=30 on val_mean_dice)
  - Best checkpoint saving
  - CSV training history export
  - Optional MLflow logging

Foreground mean excludes BG — consistent with Yang 2020, Singh 2023, Tesfaye 2023.
"""

import os
import csv
import time
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.evaluation.metrics import compute_segmentation_metrics

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

logger = logging.getLogger(__name__)


class Trainer:
    """
    Manages the full training lifecycle for baseline U-Net (and LWEU-Net).

    Parameters
    ----------
    model          : nn.Module
    criterion      : nn.Module         CombinedLoss (Dice + CE)
    optimizer      : torch.optim       Adam
    scheduler      : ReduceLROnPlateau
    train_loader   : DataLoader
    val_loader     : DataLoader
    device         : torch.device
    config         : dict              Full config loaded from YAML
    checkpoint_dir : str
    log_dir        : str
    use_mlflow     : bool
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: ReduceLROnPlateau,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        config: dict,
        checkpoint_dir: str,
        log_dir: str,
        use_mlflow: bool = False,
    ):
        self.model          = model
        self.criterion      = criterion
        self.optimizer      = optimizer
        self.scheduler      = scheduler
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.device         = device
        self.config         = config
        self.checkpoint_dir = checkpoint_dir
        self.log_dir        = log_dir
        self.use_mlflow     = use_mlflow

        self.max_epochs    = config.get("epochs", 200)
        self.es_patience   = config.get("early_stopping_patience", 30)

        # State
        self.best_val_dice     = -1.0
        self.best_epoch        = 0
        self.epochs_no_improve = 0
        self.history           = []

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir,        exist_ok=True)

        # MLflow (optional)
        self.mlflow = None
        if use_mlflow:
            try:
                import mlflow
                self.mlflow = mlflow
                mlflow.set_tracking_uri("experiments/mlruns")
                mlflow.set_experiment(config.get("experiment_name", "unet_baseline"))
            except ImportError:
                logger.warning("MLflow not installed — skipping.")

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()

        total_loss = total_dice_loss = total_ce_loss = 0.0
        n_batches  = 0

        loader = self.train_loader
        if TQDM_AVAILABLE:
            loader = tqdm(loader, desc=f"Epoch {epoch:03d} [Train]", leave=False)

        for images, masks in loader:
            images = images.to(self.device)
            masks  = masks.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)

            loss, loss_dict = self.criterion(logits, masks)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss      += loss_dict["total"]
            total_dice_loss += loss_dict["dice"]
            total_ce_loss   += loss_dict["ce"]
            n_batches       += 1

            if TQDM_AVAILABLE:
                loader.set_postfix(loss=f"{loss_dict['total']:.4f}")

        return {
            "train_loss":      total_loss      / n_batches,
            "train_dice_loss": total_dice_loss / n_batches,
            "train_ce_loss":   total_ce_loss   / n_batches,
        }

    # ------------------------------------------------------------------
    # Validation epoch
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> dict:
        self.model.eval()

        total_loss = 0.0
        n_batches  = 0

        # All metric keys from compute_segmentation_metrics
        metric_keys = (
            [f"dice_{c}" for c in ["bg", "rv", "myo", "lv"]] +
            [f"iou_{c}"  for c in ["bg", "rv", "myo", "lv"]] +
            [f"prec_{c}" for c in ["bg", "rv", "myo", "lv"]] +
            [f"rec_{c}"  for c in ["bg", "rv", "myo", "lv"]] +
            ["mean_dice", "mean_iou", "mean_precision", "mean_recall"]
        )
        accum = {k: 0.0 for k in metric_keys}

        loader = self.val_loader
        if TQDM_AVAILABLE:
            loader = tqdm(loader, desc=f"Epoch {epoch:03d} [Val  ]", leave=False)

        for images, masks in loader:
            images = images.to(self.device)
            masks  = masks.to(self.device)

            logits = self.model(images)

            loss, loss_dict  = self.criterion(logits, masks)
            batch_metrics    = compute_segmentation_metrics(logits, masks, num_classes=4)

            total_loss += loss_dict["total"]
            for k in metric_keys:
                accum[k] += batch_metrics[k]
            n_batches  += 1

        result = {"val_loss": total_loss / n_batches}
        for k, v in accum.items():
            result[f"val_{k}"] = v / n_batches

        return result

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, val_mean_dice: float):
        path = os.path.join(self.checkpoint_dir, "best_model.pth")
        torch.save({
            "epoch":           epoch,
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "val_mean_dice":   val_mean_dice,
            "config":          self.config,
        }, path)
        logger.info(f"  ✓ Checkpoint saved → {path}  (val_mean_dice={val_mean_dice:.4f})")

    def load_best_checkpoint(self):
        path = os.path.join(self.checkpoint_dir, "best_model.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No checkpoint at {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        logger.info(
            f"Loaded best checkpoint from epoch {ckpt['epoch']} "
            f"(val_mean_dice={ckpt['val_mean_dice']:.4f})"
        )

    # ------------------------------------------------------------------
    # History export
    # ------------------------------------------------------------------

    def _save_history(self):
        if not self.history:
            return
        csv_path = os.path.join(self.log_dir, "training_history.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.history[0].keys()))
            writer.writeheader()
            writer.writerows(self.history)
        logger.info(f"Training history saved → {csv_path}")

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        """
        Run the full training loop.

        Stops when max_epochs is reached OR val_mean_dice has not improved
        for es_patience consecutive epochs (early stopping).
        """
        logger.info("=" * 65)
        logger.info(f"  Run  : {self.config.get('run_name', 'unet_baseline')}")
        logger.info(f"  Device     : {self.device}")
        logger.info(f"  Max epochs : {self.max_epochs}  |  ES patience: {self.es_patience}")
        logger.info("=" * 65)

        if self.mlflow:
            self.mlflow.start_run(run_name=self.config.get("run_name"))
            self.mlflow.log_params({
                "lr":         self.config.get("learning_rate"),
                "batch_size": self.config.get("batch_size"),
                "epochs":     self.max_epochs,
                "loss":       "0.5_dice_0.5_ce",
                "model":      self.config.get("model", "unet_baseline"),
            })

        t_start = time.time()

        for epoch in range(1, self.max_epochs + 1):
            t_epoch = time.time()

            train_metrics = self._train_epoch(epoch)
            val_metrics   = self._val_epoch(epoch)

            val_mean_dice = val_metrics["val_mean_dice"]
            self.scheduler.step(val_mean_dice)
            current_lr = self.optimizer.param_groups[0]["lr"]

            epoch_record = {
                "epoch": epoch,
                "lr":    current_lr,
                **train_metrics,
                **val_metrics,
            }
            self.history.append(epoch_record)

            elapsed = time.time() - t_epoch
            logger.info(
                f"Epoch {epoch:03d}/{self.max_epochs} | "
                f"Loss={train_metrics['train_loss']:.4f} | "
                f"Val Loss={val_metrics['val_loss']:.4f} | "
                f"Dice LV={val_metrics['val_dice_lv']:.4f} "
                f"RV={val_metrics['val_dice_rv']:.4f} "
                f"MYO={val_metrics['val_dice_myo']:.4f} "
                f"Mean={val_mean_dice:.4f} | "
                f"IoU={val_metrics['val_mean_iou']:.4f} | "
                f"LR={current_lr:.2e} | {elapsed:.1f}s"
            )

            if self.mlflow:
                self.mlflow.log_metrics(
                    {k: v for k, v in epoch_record.items() if k != "epoch"},
                    step=epoch
                )

            # Checkpoint on improvement
            if val_mean_dice > self.best_val_dice:
                self.best_val_dice     = val_mean_dice
                self.best_epoch        = epoch
                self.epochs_no_improve = 0
                self._save_checkpoint(epoch, val_mean_dice)
            else:
                self.epochs_no_improve += 1

            # Early stopping
            if self.epochs_no_improve >= self.es_patience:
                logger.info(
                    f"\nEarly stopping at epoch {epoch}. "
                    f"Best val_mean_dice={self.best_val_dice:.4f} "
                    f"at epoch {self.best_epoch}."
                )
                break

        total_time = time.time() - t_start
        logger.info(f"\nTraining complete in {total_time / 60:.1f} min.")
        logger.info(f"Best val_mean_dice={self.best_val_dice:.4f} at epoch {self.best_epoch}")

        self._save_history()

        if self.mlflow:
            self.mlflow.log_metric("best_val_mean_dice", self.best_val_dice)
            self.mlflow.log_metric("best_epoch",         self.best_epoch)
            self.mlflow.end_run()

        return self.history