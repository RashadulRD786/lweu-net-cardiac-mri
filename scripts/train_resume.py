"""
Training entry point with resume support.

Usage:
    # Normal training
    python scripts/train.py --config configs/train_lweunet_v2_bl.yaml

    # Resume from checkpoint
    python scripts/train.py --config configs/train_lweunet_v2_bl.yaml \
        --resume checkpoints/lweunet_v2_bl/best_model.pth
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset             import ACDCDataset
from src.models.unet_baseline     import UNetBaseline
from src.models.lweunet.lweunet   import LWEUNet
from src.models.lweunet.lweunet_v2 import LWEUNetV2
from src.losses.combo_loss        import CombinedLoss
from src.training.trainer         import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def set_seed(seed):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    logger.info(f"Seed: {seed}")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_dataloaders(cfg):
    train_ds = ACDCDataset(cfg["data_dir"], "train", augment=True)
    val_ds   = ACDCDataset(cfg["data_dir"], "val",   augment=False)
    logger.info(f"Train: {train_ds}")
    logger.info(f"Val  : {val_ds}")
    kw = dict(batch_size=cfg["batch_size"],
              num_workers=cfg.get("num_workers", 4),
              pin_memory=cfg.get("pin_memory", True))
    train_loader = DataLoader(train_ds, shuffle=True,  drop_last=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, drop_last=False, **kw)
    logger.info(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")
    return train_loader, val_loader


def build_model(cfg, device):
    name = cfg.get("model", "unet_baseline")
    if name == "unet_baseline":
        model = UNetBaseline(
            in_channels  = cfg.get("in_channels",  1),
            num_classes  = cfg.get("num_classes",  4),
            base_filters = cfg.get("base_filters", 64),
            dropout_p    = cfg.get("dropout_p",    0.5),
        )
        logger.info(f"Model: UNetBaseline | Params: {model.count_parameters():,}")
    elif name == "lweunet":
        model = LWEUNet(
            in_channels  = cfg.get("in_channels",  1),
            num_classes  = cfg.get("num_classes",  4),
            base_filters = cfg.get("base_filters", 32),
            dropout_p    = cfg.get("dropout_p",    0.5),
            use_eca      = cfg.get("use_eca",       True),
        )
        bd = model.parameter_breakdown()
        logger.info(f"Model: LWEUNet | Total={bd['total']:,} ({bd['total']/1e6:.2f}M)")
    elif name == "lweunet_v2":
        model = LWEUNetV2(
            in_channels  = cfg.get("in_channels",  1),
            num_classes  = cfg.get("num_classes",  4),
            base_filters = cfg.get("base_filters", 32),
            dropout_p    = cfg.get("dropout_p",    0.5),
        )
        bd = model.parameter_breakdown()
        logger.info(f"Model: LWEUNetV2")
        logger.info(f"  Encoder={bd['encoder']:,}  "
                    f"Bottleneck={bd['bottleneck']:,}  "
                    f"Decoder={bd['decoder']:,}  "
                    f"Total={bd['total']:,} ({bd['total']/1e6:.2f}M)")
    else:
        raise ValueError(f"Unknown model '{name}'")
    return model.to(device)


def build_loss(cfg, device):
    lc = cfg.get("loss", {})
    cw = lc.get("class_weights", None)
    if cw is not None:
        cw = torch.tensor(cw, dtype=torch.float32).to(device)
    criterion = CombinedLoss(
        num_classes       = cfg.get("num_classes", 4),
        dice_weight       = lc.get("dice_weight",       0.5),
        ce_weight         = lc.get("ce_weight",         0.5),
        use_boundary_loss = lc.get("use_boundary_loss", False),
        boundary_weight   = lc.get("boundary_weight",   0.1),
        warmup_epochs     = lc.get("warmup_epochs",     80),
        class_weights     = cw,
    )
    logger.info(f"Loss: dice={lc.get('dice_weight',0.5)} "
                f"ce={lc.get('ce_weight',0.5)} "
                f"boundary={lc.get('use_boundary_loss',False)}")
    return criterion


def build_optimizer_and_scheduler(cfg, model):
    lr  = cfg.get("learning_rate", 1e-4)
    wd  = cfg.get("weight_decay",  0.0)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sc  = cfg.get("lr_schedule", {})
    sched = ReduceLROnPlateau(
        opt,
        mode     = sc.get("mode",     "max"),
        factor   = sc.get("factor",   0.5),
        patience = sc.get("patience", 10),
        min_lr   = sc.get("min_lr",   1e-6),
        verbose  = sc.get("verbose",  True),
    )
    logger.info(f"Optimizer: Adam lr={lr}  Scheduler: ReduceLROnPlateau")
    return opt, sched


def load_checkpoint(checkpoint_path, model, optimizer, device):
    """
    Load model and optimizer state from checkpoint.
    Returns the epoch and best val_mean_dice from the checkpoint.
    """
    logger.info(f"Resuming from checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])

    start_epoch    = ckpt.get("epoch", 0)
    best_val_dice  = ckpt.get("val_mean_dice", 0.0)

    logger.info(f"Resumed from epoch {start_epoch} | "
                f"best_val_mean_dice={best_val_dice:.4f}")

    return start_epoch, best_val_dice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  required=True)
    parser.add_argument("--device",  default=None)
    parser.add_argument("--resume",  default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(device)}")

    log_dir = cfg.get("log_dir", "logs/lweunet")
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(log_dir, "train.log"), mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logging.getLogger().addHandler(fh)

    train_loader, val_loader = build_dataloaders(cfg)
    model                    = build_model(cfg, device)
    criterion                = build_loss(cfg, device)
    optimizer, scheduler     = build_optimizer_and_scheduler(cfg, model)

    # --- Resume logic ---
    start_epoch   = 0
    best_val_dice = -1.0

    if args.resume:
        start_epoch, best_val_dice = load_checkpoint(
            args.resume, model, optimizer, device
        )

    trainer = Trainer(
        model          = model,
        criterion      = criterion,
        optimizer      = optimizer,
        scheduler      = scheduler,
        train_loader   = train_loader,
        val_loader     = val_loader,
        device         = device,
        config         = cfg,
        checkpoint_dir = cfg.get("checkpoint_dir", "checkpoints/lweunet"),
        log_dir        = log_dir,
        use_mlflow     = cfg.get("mlflow", False),
    )

    # Inject resume state into trainer
    if args.resume:
        trainer.best_val_dice    = best_val_dice
        trainer.best_epoch       = start_epoch
        trainer.epochs_no_improve = 0
        # Override max_epochs to account for already completed epochs
        trainer.max_epochs       = cfg.get("epochs", 200)
        # Patch the train loop to start from correct epoch
        original_train = trainer.train

        def train_from_epoch():
            import time, csv
            logger.info("=" * 65)
            logger.info(f"  RESUMING from epoch {start_epoch}")
            logger.info(f"  Best val_mean_dice so far: {best_val_dice:.4f}")
            logger.info("=" * 65)

            import time as time_module
            t_start = time_module.time()

            for epoch in range(start_epoch + 1, trainer.max_epochs + 1):
                t_epoch = time_module.time()

                trainer.criterion.set_epoch(epoch)
                train_metrics = trainer._train_epoch(epoch)
                val_metrics   = trainer._val_epoch(epoch)

                val_mean_dice = val_metrics["val_mean_dice"]
                trainer.scheduler.step(val_mean_dice)
                current_lr = trainer.optimizer.param_groups[0]["lr"]

                epoch_record = {
                    "epoch": epoch,
                    "lr":    current_lr,
                    **train_metrics,
                    **val_metrics,
                }
                trainer.history.append(epoch_record)

                elapsed = time_module.time() - t_epoch
                logger.info(
                    f"Epoch {epoch:03d}/{trainer.max_epochs} | "
                    f"Loss={train_metrics['train_loss']:.4f} | "
                    f"Val Loss={val_metrics['val_loss']:.4f} | "
                    f"Dice LV={val_metrics['val_dice_lv']:.4f} "
                    f"RV={val_metrics['val_dice_rv']:.4f} "
                    f"MYO={val_metrics['val_dice_myo']:.4f} "
                    f"Mean={val_mean_dice:.4f} | "
                    f"IoU={val_metrics['val_mean_iou']:.4f} | "
                    f"LR={current_lr:.2e} | {elapsed:.1f}s"
                )

                if val_mean_dice > trainer.best_val_dice:
                    trainer.best_val_dice     = val_mean_dice
                    trainer.best_epoch        = epoch
                    trainer.epochs_no_improve = 0
                    trainer._save_checkpoint(epoch, val_mean_dice)
                else:
                    trainer.epochs_no_improve += 1

                if trainer.epochs_no_improve >= trainer.es_patience:
                    logger.info(
                        f"\nEarly stopping at epoch {epoch}. "
                        f"Best val_mean_dice={trainer.best_val_dice:.4f} "
                        f"at epoch {trainer.best_epoch}."
                    )
                    break

            total_time = time_module.time() - t_start
            logger.info(f"\nTraining complete in {total_time/60:.1f} min.")
            logger.info(f"Best val_mean_dice={trainer.best_val_dice:.4f} "
                        f"at epoch {trainer.best_epoch}")
            trainer._save_history()
            return trainer.history

        train_from_epoch()
    else:
        # Normal training — set_epoch called from within
        import time as time_module
        t_start = time_module.time()
        for epoch in range(1, trainer.max_epochs + 1):
            t_epoch = time_module.time()
            trainer.criterion.set_epoch(epoch)
            train_metrics = trainer._train_epoch(epoch)
            val_metrics   = trainer._val_epoch(epoch)
            val_mean_dice = val_metrics["val_mean_dice"]
            trainer.scheduler.step(val_mean_dice)
            current_lr = trainer.optimizer.param_groups[0]["lr"]
            epoch_record = {"epoch": epoch, "lr": current_lr,
                           **train_metrics, **val_metrics}
            trainer.history.append(epoch_record)
            elapsed = time_module.time() - t_epoch
            logger.info(
                f"Epoch {epoch:03d}/{trainer.max_epochs} | "
                f"Loss={train_metrics['train_loss']:.4f} | "
                f"Val Loss={val_metrics['val_loss']:.4f} | "
                f"Dice LV={val_metrics['val_dice_lv']:.4f} "
                f"RV={val_metrics['val_dice_rv']:.4f} "
                f"MYO={val_metrics['val_dice_myo']:.4f} "
                f"Mean={val_mean_dice:.4f} | "
                f"IoU={val_metrics['val_mean_iou']:.4f} | "
                f"LR={current_lr:.2e} | {elapsed:.1f}s"
            )
            if val_mean_dice > trainer.best_val_dice:
                trainer.best_val_dice     = val_mean_dice
                trainer.best_epoch        = epoch
                trainer.epochs_no_improve = 0
                trainer._save_checkpoint(epoch, val_mean_dice)
            else:
                trainer.epochs_no_improve += 1
            if trainer.epochs_no_improve >= trainer.es_patience:
                logger.info(f"\nEarly stopping at epoch {epoch}.")
                break
        logger.info(f"Best val_mean_dice={trainer.best_val_dice:.4f} "
                    f"at epoch {trainer.best_epoch}")
        trainer._save_history()

    logger.info("Done.")


if __name__ == "__main__":
    main()