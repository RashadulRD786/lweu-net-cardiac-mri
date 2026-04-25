"""
Training entry point — supports baseline U-Net and all LWEU-Net ablation variants.

Usage:
    python scripts/train.py --config configs/train_unet_baseline.yaml
    python scripts/train.py --config configs/train_lweunet_base.yaml
    python scripts/train.py --config configs/train_lweunet_eca.yaml
    python scripts/train.py --config configs/train_lweunet_full.yaml
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

from src.data.dataset           import ACDCDataset
from src.models.unet_baseline   import UNetBaseline
from src.models.lweunet.lweunet import LWEUNet
from src.losses.combo_loss      import CombinedLoss
from src.training.trainer       import Trainer

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
        logger.info(f"Model: LWEUNet (use_eca={cfg.get('use_eca')})")
        logger.info(f"  Encoder={bd['encoder']:,}  "
                    f"Bottleneck={bd['bottleneck']:,}  "
                    f"Decoder={bd['decoder']:,}  "
                    f"Total={bd['total']:,} ({bd['total']/1e6:.2f}M)")

    elif name == "lweunet_v2":
        from src.models.lweunet.lweunet_v2 import LWEUNetV2
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

    elif name == "lweunet_v2_lsa":
        from src.models.lweunet.lweunet_v2_lsa import LWEUNetV2LSA
        model = LWEUNetV2LSA(
            in_channels  = cfg.get("in_channels",  1),
            num_classes  = cfg.get("num_classes",  4),
            base_filters = cfg.get("base_filters", 32),
            dropout_p    = cfg.get("dropout_p",    0.5),
        )
        bd = model.parameter_breakdown()
        logger.info(f"Model: LWEUNetV2LSA")
        logger.info(f"  Total={bd['total']:,} ({bd['total']/1e6:.2f}M)")
    else:
        raise ValueError(f"Unknown model '{name}'")
    return model.to(device)




def build_loss(cfg, device):
    lc  = cfg.get("loss", {})
    cw  = lc.get("class_weights", None)
    if cw is not None:
        cw = torch.tensor(cw, dtype=torch.float32).to(device)
    criterion = CombinedLoss(
        num_classes       = cfg.get("num_classes", 4),
        dice_weight       = lc.get("dice_weight",       0.5),
        ce_weight         = lc.get("ce_weight",         0.5),
        use_boundary_loss = lc.get("use_boundary_loss", False),
        boundary_weight   = lc.get("boundary_weight",   0.2),
        warmup_epochs     = lc.get("warmup_epochs",     50),
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
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
    fh = logging.FileHandler(os.path.join(log_dir, "train.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logging.getLogger().addHandler(fh)

    train_loader, val_loader = build_dataloaders(cfg)
    model                    = build_model(cfg, device)
    criterion                = build_loss(cfg, device)
    optimizer, scheduler     = build_optimizer_and_scheduler(cfg, model)

    Trainer(
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
    ).train()

    logger.info("Done.")


if __name__ == "__main__":
    main()