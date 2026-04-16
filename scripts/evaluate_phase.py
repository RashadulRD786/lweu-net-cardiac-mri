"""
Test set evaluation with optional ED/ES phase filtering.

Usage:
    python scripts/evaluate_phase.py \
        --config configs/train_unet_baseline.yaml \
        --checkpoint checkpoints/unet_baseline/best_model.pth \
        --phase ALL
"""

import os
import sys
import json
import argparse
import logging

import torch
from torch.utils.data import DataLoader, Dataset
import yaml
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Model imports handled inside build_model() based on config
from src.evaluation.metrics   import (
    evaluate_on_test_set,
    get_efficiency_summary,
    print_results_table,
)
from src.data.augmentation import get_val_augmentation


class ACDCDatasetPhase(Dataset):
    NUM_CLASSES = 4
    IMAGE_SIZE  = (224, 224)

    def __init__(self, data_dir: str, split: str, phase: str = None):
        assert phase in ("ED", "ES", None)
        self.split_dir = os.path.join(data_dir, split)
        self.phase     = phase

        all_stems = sorted([
            f.replace("_img.npy", "")
            for f in os.listdir(self.split_dir)
            if f.endswith("_img.npy")
        ])

        self.stems = (
            [s for s in all_stems if f"_{phase}_" in s]
            if phase is not None else all_stems
        )

        if len(self.stems) == 0:
            raise RuntimeError(f"No slices found for split='{split}', phase='{phase}'")

        self.transform = get_val_augmentation()

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem  = self.stems[idx]
        image = np.load(os.path.join(self.split_dir, f"{stem}_img.npy"))
        mask  = np.load(os.path.join(self.split_dir, f"{stem}_msk.npy"))
        augmented = self.transform(image=image, mask=mask.astype(np.uint8))
        image = torch.from_numpy(augmented["image"]).unsqueeze(0).float()
        mask  = torch.from_numpy(augmented["mask"].astype(np.int64)).long()
        return image, mask

    def __repr__(self):
        phase_str = self.phase if self.phase else "ED+ES"
        return f"ACDCDatasetPhase(phase='{phase_str}', n_slices={len(self)})"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg, checkpoint_path, device):
    model_name = cfg.get("model", "unet_baseline")

    if model_name == "unet_baseline":
        from src.models.unet_baseline import UNetBaseline
        model = UNetBaseline(
            in_channels  = cfg.get("in_channels",  1),
            num_classes  = cfg.get("num_classes",  4),
            base_filters = cfg.get("base_filters", 64),
            dropout_p    = cfg.get("dropout_p",    0.5),
        )
    elif model_name == "lweunet":
        from src.models.lweunet.lweunet import LWEUNet
        model = LWEUNet(
            in_channels  = cfg.get("in_channels",  1),
            num_classes  = cfg.get("num_classes",  4),
            base_filters = cfg.get("base_filters", 32),
            dropout_p    = cfg.get("dropout_p",    0.5),
            use_eca      = cfg.get("use_eca",       True),
        )
    else:
        raise ValueError(f"Unknown model '{model_name}'")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    logger.info(f"Checkpoint: epoch {ckpt.get('epoch','?')}, "
                f"val_mean_dice={ckpt.get('val_mean_dice', 0):.4f}")
    return model


def run_evaluation(model, cfg, device, phase, label):
    ds = ACDCDatasetPhase(data_dir=cfg["data_dir"], split="test", phase=phase)
    logger.info(f"\n{ds}")

    loader = DataLoader(
        ds,
        batch_size  = cfg.get("batch_size", 16),
        shuffle     = False,
        num_workers = cfg.get("num_workers", 4),
        pin_memory  = (device.type == "cuda"),
        drop_last   = False,
    )

    results = evaluate_on_test_set(
        model            = model,
        test_loader      = loader,
        device           = device,
        num_classes      = cfg.get("num_classes", 4),
        pixel_spacing_mm = 1.5,          # resampling target → HD95 in mm
    )

    print_results_table(results, model_name=label)

    logger.info(f"  [{label}] Dice  LV={results['dice_lv']:.4f}  "
                f"RV={results['dice_rv']:.4f}  "
                f"MYO={results['dice_myo']:.4f}  "
                f"Mean={results['mean_dice']:.4f}")
    logger.info(f"  [{label}] IoU   LV={results['iou_lv']:.4f}  "
                f"RV={results['iou_rv']:.4f}  "
                f"MYO={results['iou_myo']:.4f}  "
                f"Mean={results['mean_iou']:.4f}")
    logger.info(f"  [{label}] HD95  LV={results['hd95_lv']:.2f}mm  "
                f"RV={results['hd95_rv']:.2f}mm  "
                f"MYO={results['hd95_myo']:.2f}mm  "
                f"Mean={results['mean_hd95']:.2f}mm")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--phase",      default="ALL",
                        choices=["ED", "ES", "ALL", "combined"])
    parser.add_argument("--device",     default=None)
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model      = build_model(cfg, args.checkpoint, device)
    efficiency = get_efficiency_summary(model, input_size=(1,1,224,224), device=device)

    all_results = {}

    if args.phase == "ALL":
        phases = [("ED", "ED"), ("ES", "ES"), (None, "ED+ES")]
    elif args.phase == "combined":
        phases = [(None, "ED+ES")]
    else:
        phases = [(args.phase, args.phase)]

    for phase_filter, label in phases:
        results = run_evaluation(model, cfg, device, phase_filter, label)
        all_results[label] = {**results, **efficiency}

    # Save JSON
    log_dir  = cfg.get("log_dir", "logs/unet_baseline")
    os.makedirs(log_dir, exist_ok=True)
    out_path = os.path.join(log_dir, "test_results_by_phase.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nAll results saved → {out_path}")

    # Final comparison summary
    if args.phase == "ALL":
        logger.info("\n" + "=" * 72)
        logger.info("  PHASE COMPARISON SUMMARY")
        logger.info("=" * 72)
        logger.info(f"  {'Phase':<12} {'LV Dice':>9} {'RV Dice':>9} "
                    f"{'MYO Dice':>9} {'Mean':>7} {'HD95 Mean':>10}")
        logger.info("-" * 72)
        for label, res in all_results.items():
            logger.info(
                f"  {label:<12} "
                f"{res['dice_lv']:>9.4f} "
                f"{res['dice_rv']:>9.4f} "
                f"{res['dice_myo']:>9.4f} "
                f"{res['mean_dice']:>7.4f} "
                f"{res['mean_hd95']:>9.2f}mm"
            )
        logger.info("=" * 72)


if __name__ == "__main__":
    main()