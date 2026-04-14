"""
Training entry point for LWEU-Net project.

Usage:
    python scripts/train.py --config configs/train_unet_baseline.yaml
    python scripts/train.py --config configs/train_unet_baseline.yaml --device cuda
    python scripts/train.py --config configs/train_unet_baseline.yaml --device cpu

The script:
  1. Loads config from YAML
  2. Sets global seed for reproducibility
  3. Builds Dataset and DataLoaders
  4. Builds model, loss, optimizer, scheduler
  5. Calls Trainer.train()
  6. Logs completion summary
"""

import os
import sys
import argparse
import logging

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset      import ACDCDataset
from src.models.unet_baseline import UNetBaseline
from src.losses.combo_loss import CombinedLoss
from src.training.trainer  import Trainer

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    logger.info(f"Global seed set to {seed}")


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Config loaded from {path}")
    return cfg


# ---------------------------------------------------------------------------
# Build components
# ---------------------------------------------------------------------------

def build_dataloaders(cfg: dict):
    data_dir    = cfg["data_dir"]
    batch_size  = cfg["batch_size"]
    num_workers = cfg.get("num_workers", 4)
    pin_memory  = cfg.get("pin_memory", True)

    train_ds = ACDCDataset(data_dir=data_dir, split="train", augment=True)
    val_ds   = ACDCDataset(data_dir=data_dir, split="val",   augment=False)

    logger.info(f"Train dataset : {train_ds}")
    logger.info(f"Val   dataset : {val_ds}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,    # avoids BatchNorm issues with single-sample last batch
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    logger.info(f"Train batches : {len(train_loader)} × batch_size={batch_size}")
    logger.info(f"Val   batches : {len(val_loader)}")

    return train_loader, val_loader


def build_model(cfg: dict, device: torch.device) -> torch.nn.Module:
    model_name = cfg.get("model", "unet_baseline")
    assert model_name == "unet_baseline", \
        f"This script currently supports 'unet_baseline', got '{model_name}'"

    model = UNetBaseline(
        in_channels  = cfg.get("in_channels",  1),
        num_classes  = cfg.get("num_classes",  4),
        base_filters = cfg.get("base_filters", 64),
        dropout_p    = cfg.get("dropout_p",    0.5),
    )
    model = model.to(device)
    logger.info(f"Model : UNetBaseline | Parameters: {model.count_parameters():,}")
    return model


def build_loss(cfg: dict, device: torch.device) -> CombinedLoss:
    loss_cfg      = cfg.get("loss", {})
    dice_weight   = loss_cfg.get("dice_weight",   0.5)
    ce_weight     = loss_cfg.get("ce_weight",     0.5)
    class_weights = loss_cfg.get("class_weights", None)

    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = CombinedLoss(
        num_classes    = cfg.get("num_classes", 4),
        dice_weight    = dice_weight,
        ce_weight      = ce_weight,
        class_weights  = class_weights,
    )
    logger.info(f"Loss : {dice_weight:.1f} × Dice + {ce_weight:.1f} × CrossEntropy")
    return criterion


def build_optimizer_and_scheduler(cfg: dict, model):
    lr           = cfg.get("learning_rate", 1e-4)
    weight_decay = cfg.get("weight_decay",  0.0)

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    sched_cfg = cfg.get("lr_schedule", {})
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode    = sched_cfg.get("mode",     "max"),
        factor  = sched_cfg.get("factor",   0.5),
        patience= sched_cfg.get("patience", 10),
        min_lr  = sched_cfg.get("min_lr",   1e-6),
        verbose = sched_cfg.get("verbose",  True),
    )

    logger.info(f"Optimizer : Adam (lr={lr}, weight_decay={weight_decay})")
    logger.info(f"Scheduler : ReduceLROnPlateau (factor=0.5, patience=10)")
    return optimizer, scheduler


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train segmentation model on ACDC")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file, e.g. configs/train_unet_baseline.yaml"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device override: 'cuda', 'cpu', or 'cuda:0'. Auto-detected if not set."
    )
    args = parser.parse_args()

    # --- Config ---
    cfg = load_config(args.config)

    # --- Seed ---
    set_seed(cfg.get("seed", 42))

    # --- Device ---
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device : {device}")
    if device.type == "cuda":
        logger.info(f"GPU    : {torch.cuda.get_device_name(device)}")
        logger.info(f"VRAM   : {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")

    # --- Add file handler for log_dir ---
    log_dir = cfg.get("log_dir", "logs/unet_baseline")
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, "train.log"))
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logging.getLogger().addHandler(file_handler)

    # --- Build components ---
    train_loader, val_loader = build_dataloaders(cfg)
    model                    = build_model(cfg, device)
    criterion                = build_loss(cfg, device)
    optimizer, scheduler     = build_optimizer_and_scheduler(cfg, model)

    # --- Trainer ---
    trainer = Trainer(
        model          = model,
        criterion      = criterion,
        optimizer      = optimizer,
        scheduler      = scheduler,
        train_loader   = train_loader,
        val_loader     = val_loader,
        device         = device,
        config         = cfg,
        checkpoint_dir = cfg.get("checkpoint_dir", "checkpoints/unet_baseline"),
        log_dir        = log_dir,
        use_mlflow     = cfg.get("mlflow", False),
    )

    # --- Train ---
    history = trainer.train()

    logger.info("Done.")
    return history


if __name__ == "__main__":
    main()