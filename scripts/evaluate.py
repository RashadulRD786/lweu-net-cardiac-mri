"""
Test set evaluation script for ACDC cardiac MRI segmentation.

Loads the best saved checkpoint, runs inference on the test set,
and computes all metrics for the thesis results table.

Usage:
    # Combined ED+ES (same as evaluate.py)
    python scripts/evaluate_phase.py 
        --config configs/train_unet_baseline.yaml 
        --checkpoint checkpoints/unet_baseline/best_model.pth
 
    # ED phase only
    python scripts/evaluate_phase.py 
        --config configs/train_unet_baseline.yaml 
        --checkpoint checkpoints/unet_baseline/best_model.pth 
        --phase ED
 
    # ES phase only
    python scripts/evaluate_phase.py 
        --config configs/train_unet_baseline.yaml 
        --checkpoint checkpoints/unet_baseline/best_model.pth 
        --phase ES
 
    # Run all three automatically
    python scripts/evaluate_phase.py 
        --config configs/train_unet_baseline.yaml 
        --checkpoint checkpoints/unet_baseline/best_model.pth 
        --phase ALL
"""

"""Outputs:
    - Formatted results table printed to console
    - JSON results file saved to logs/{experiment}/test_results.json
    - Saved to logs/{experiment}/test_results.txt  (plain text copy)
"""

import os
import sys
import json
import argparse
import logging
 
import torch
from torch.utils.data import DataLoader
import yaml
 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
 
from src.models.unet_baseline import UNetBaseline
from src.evaluation.metrics   import (
    evaluate_on_test_set,
    get_efficiency_summary,
    print_results_table,
)
 
# Import updated dataset with phase support
# We define it inline here so no file replacement is needed
import numpy as np
from torch.utils.data import Dataset
from src.data.augmentation import get_val_augmentation
 
 
class ACDCDatasetPhase(Dataset):
    """ACDCDataset with phase filtering. augment=False always for evaluation."""
 
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
 
        if phase is not None:
            self.stems = [s for s in all_stems if f"_{phase}_" in s]
        else:
            self.stems = all_stems
 
        if len(self.stems) == 0:
            raise RuntimeError(
                f"No slices found for split='{split}', phase='{phase}'"
            )
 
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
 
 
# ---------------------------------------------------------------------------
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
    model = UNetBaseline(
        in_channels  = cfg.get("in_channels",  1),
        num_classes  = cfg.get("num_classes",  4),
        base_filters = cfg.get("base_filters", 64),
        dropout_p    = cfg.get("dropout_p",    0.5),
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    best_epoch = ckpt.get("epoch", "?")
    best_dice  = ckpt.get("val_mean_dice", 0)
    logger.info(f"Checkpoint: epoch {best_epoch}, val_mean_dice={best_dice:.4f}")
    return model
 
 
def run_evaluation(model, cfg, device, phase, label):
    """Run evaluation for one phase and print results."""
    data_dir   = cfg["data_dir"]
    batch_size = cfg.get("batch_size", 16)
 
    ds = ACDCDatasetPhase(data_dir=data_dir, split="test", phase=phase)
    logger.info(f"\n{ds}")
 
    loader = DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = cfg.get("num_workers", 4),
        pin_memory  = (device.type == "cuda"),
        drop_last   = False,
    )
 
    results = evaluate_on_test_set(
        model       = model,
        test_loader = loader,
        device      = device,
        num_classes = cfg.get("num_classes", 4),
    )
 
    print_results_table(results, model_name=label)
 
    # Log thesis-ready numbers
    logger.info(f"  [{label}] Dice  LV={results['dice_lv']:.4f}  "
                f"RV={results['dice_rv']:.4f}  "
                f"MYO={results['dice_myo']:.4f}  "
                f"Mean={results['mean_dice']:.4f}")
    logger.info(f"  [{label}] IoU   LV={results['iou_lv']:.4f}  "
                f"RV={results['iou_rv']:.4f}  "
                f"MYO={results['iou_myo']:.4f}  "
                f"Mean={results['mean_iou']:.4f}")
 
    return results
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--phase",      default="ALL",
                        choices=["ED", "ES", "ALL", "combined"],
                        help="ED | ES | combined | ALL (runs all three)")
    parser.add_argument("--device",     default=None)
    args = parser.parse_args()
 
    cfg = load_config(args.config)
 
    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
 
    model = build_model(cfg, args.checkpoint, device)
 
    # Efficiency (once only)
    efficiency = get_efficiency_summary(model, input_size=(1,1,224,224), device=device)
 
    all_results = {}
 
    if args.phase == "ALL":
        phases = [("ED", "ED"), ("ES", "ES"), (None, "ED+ES (combined)")]
    elif args.phase == "combined":
        phases = [(None, "ED+ES (combined)")]
    else:
        phases = [(args.phase, args.phase)]
 
    for phase_filter, label in phases:
        results = run_evaluation(model, cfg, device, phase_filter, label)
        all_results[label] = {**results, **efficiency}
 
    # Save all results to JSON
    log_dir = cfg.get("log_dir", "logs/unet_baseline")
    os.makedirs(log_dir, exist_ok=True)
    out_path = os.path.join(log_dir, "test_results_by_phase.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nAll results saved → {out_path}")
 
    # Print final comparison summary
    if args.phase == "ALL":
        logger.info("\n" + "=" * 65)
        logger.info("  PHASE COMPARISON SUMMARY")
        logger.info("=" * 65)
        logger.info(f"  {'Phase':<12} {'LV':>7} {'RV':>7} {'MYO':>7} {'Mean':>7}")
        logger.info("-" * 65)
        for label, res in all_results.items():
            logger.info(
                f"  {label:<12} "
                f"{res['dice_lv']:>7.4f} "
                f"{res['dice_rv']:>7.4f} "
                f"{res['dice_myo']:>7.4f} "
                f"{res['mean_dice']:>7.4f}"
            )
        logger.info("=" * 65)
 
 
if __name__ == "__main__":
    main()
 