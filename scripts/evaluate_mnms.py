# scripts/evaluate_mnms.py
#
# Responsibility: Zero-shot generalisation evaluation on the M&Ms test set.
#
# Evaluates Dice and HD95 per structure (LV, RV, MYO) for:
#   - Each vendor separately : A (Siemens), B (Philips), C (GE), D (Canon)
#   - All vendors combined   : overall generalisation score
#
# Zero-shot means: the model was trained entirely on ACDC data.
# No M&Ms data was ever seen during training, validation, or model selection.
# This tests whether the EnhancedBlock learned generalisable cardiac features.
#
# Usage — evaluate V2 (proposed model):
#   python scripts/evaluate_mnms.py \
#       --config      configs/train_lweunet_v2.yaml \
#       --checkpoint  checkpoints/lweunet_v2/best_model.pth \
#       --model_label LWEU-Net_V2
#
# Usage — evaluate Base (for comparison):
#   python scripts/evaluate_mnms.py \
#       --config      configs/train_lweunet_base.yaml \
#       --checkpoint  checkpoints/lweunet_base/best_model.pth \
#       --model_label LWEU-Net_Base
#
# Output:
#   logs/mnms_eval_{model_label}.json  — full results in JSON
#   Printed per-vendor + overall table

import os
import sys
import json
import argparse
import logging
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.metrics import (
    evaluate_on_test_set,
    get_efficiency_summary,
    print_results_table,
)
from src.data.augmentation import get_val_augmentation

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)s | %(message)s",
    datefmt = "%H:%M:%S",
    handlers= [logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── Dataset ───────────────────────────────────────────────────────────────────

class MnMsDataset(Dataset):
    """
    Dataset for M&Ms preprocessed slices.

    Reads from data/MnM/preprocessed/ using slice_metadata.csv
    to filter slices by vendor when needed.

    Each item returns (image_tensor, mask_tensor) in the same format
    as ACDCDatasetPhase so evaluate_on_test_set() works unchanged.
    """
    NUM_CLASSES = 4
    IMAGE_SIZE  = (224, 224)

    def __init__(self, preprocessed_dir: str, vendor: str = None):
        """
        Args:
            preprocessed_dir : path to data/MnM/preprocessed/
            vendor           : None = all vendors; "A"/"B"/"C"/"D" = one vendor
        """
        self.data_dir  = Path(preprocessed_dir)
        meta_path      = self.data_dir / "slice_metadata.csv"

        if not meta_path.exists():
            raise FileNotFoundError(
                f"slice_metadata.csv not found at {meta_path}. "
                f"Run scripts/preprocess_mnms.py first."
            )

        meta_df = pd.read_csv(meta_path)

        # Filter to one vendor if requested
        if vendor is not None:
            meta_df = meta_df[meta_df["vendor"] == vendor].reset_index(drop=True)

        if len(meta_df) == 0:
            raise RuntimeError(
                f"No slices found for vendor='{vendor}' "
                f"in {meta_path}"
            )

        self.stems        = meta_df["filename_base"].tolist()
        self.vendor       = vendor
        self.n_patients   = meta_df["patient_id"].nunique()
        self.transform    = get_val_augmentation()

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem  = self.stems[idx]
        image = np.load(self.data_dir / f"{stem}_img.npy")
        mask  = np.load(self.data_dir / f"{stem}_msk.npy")

        augmented = self.transform(image=image, mask=mask.astype(np.uint8))
        image = torch.from_numpy(augmented["image"]).unsqueeze(0).float()
        mask  = torch.from_numpy(augmented["mask"].astype(np.int64)).long()

        return image, mask

    def __repr__(self):
        vendor_str = self.vendor if self.vendor else "ALL"
        return (f"MnMsDataset(vendor='{vendor_str}', "
                f"patients={self.n_patients}, "
                f"slices={len(self)})")


# ── Config and model loading ──────────────────────────────────────────────────

def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, checkpoint_path: str, device: torch.device):
    """
    Identical to evaluate_phase.py — load model from config + checkpoint.
    """
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
            use_eca      = cfg.get("use_eca",       False),
        )
    elif model_name == "lweunet_v2":
        from src.models.lweunet.lweunet_v2 import LWEUNetV2
        model = LWEUNetV2(
            in_channels  = cfg.get("in_channels",  1),
            num_classes  = cfg.get("num_classes",  4),
            base_filters = cfg.get("base_filters", 32),
            dropout_p    = cfg.get("dropout_p",    0.5),
        )
    else:
        raise ValueError(f"Unknown model '{model_name}'")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    logger.info(f"Checkpoint loaded: epoch {ckpt.get('epoch', '?')}, "
                f"val_mean_dice={ckpt.get('val_mean_dice', 0):.4f}")
    return model


# ── Per-vendor evaluation ─────────────────────────────────────────────────────

def run_vendor_evaluation(
    model            : torch.nn.Module,
    preprocessed_dir : str,
    vendor           : str,
    vendor_name      : str,
    cfg              : dict,
    device           : torch.device,
) -> dict:
    """
    Evaluate model on all slices for one vendor (or all vendors if vendor=None).

    Args:
        vendor      : "A"/"B"/"C"/"D" or None for overall
        vendor_name : display name for logging
    Returns:
        results dict from evaluate_on_test_set()
    """
    ds = MnMsDataset(preprocessed_dir=preprocessed_dir, vendor=vendor)
    logger.info(f"\n  {ds}")

    loader = DataLoader(
        ds,
        batch_size  = cfg.get("batch_size",  16),
        shuffle     = False,
        num_workers = cfg.get("num_workers",  4),
        pin_memory  = (device.type == "cuda"),
        drop_last   = False,
    )

    results = evaluate_on_test_set(
        model            = model,
        test_loader      = loader,
        device           = device,
        num_classes      = cfg.get("num_classes", 4),
        pixel_spacing_mm = 1.5,   # must match ACDC preprocessing target spacing
    )

    logger.info(
        f"  Vendor {vendor_name} | "
        f"Dice  LV={results['dice_lv']:.4f}  "
        f"RV={results['dice_rv']:.4f}  "
        f"MYO={results['dice_myo']:.4f}  "
        f"Mean={results['mean_dice']:.4f}"
    )
    logger.info(
        f"  Vendor {vendor_name} | "
        f"HD95  LV={results['hd95_lv']:.2f}mm  "
        f"RV={results['hd95_rv']:.2f}mm  "
        f"MYO={results['hd95_myo']:.2f}mm  "
        f"Mean={results['mean_hd95']:.2f}mm"
    )

    return results


# ── Results table ─────────────────────────────────────────────────────────────

def print_vendor_table(all_results: dict, model_label: str) -> None:
    """
    Print the per-vendor comparison table for the thesis.

    Format matches the target thesis table:
    Vendor | Scanner  | LV Dice | RV Dice | MYO Dice | Mean Dice | Mean HD95
    """
    vendor_names = {
        "A" : "Siemens",
        "B" : "Philips",
        "C" : "GE",
        "D" : "Canon",
        "Overall": "All vendors",
    }

    print("\n" + "=" * 80)
    print(f"  M&Ms ZERO-SHOT GENERALISATION RESULTS — {model_label}")
    print("=" * 80)
    print(f"  {'Vendor':<8} {'Scanner':<12} "
          f"{'LV Dice':>9} {'RV Dice':>9} {'MYO Dice':>10} "
          f"{'Mean Dice':>11} {'Mean HD95':>10}")
    print("-" * 80)

    for key in ["A", "B", "C", "D", "Overall"]:
        if key not in all_results:
            continue
        res  = all_results[key]
        name = vendor_names.get(key, key)
        sep  = "─" * 80 if key == "Overall" else ""
        if sep:
            print(sep)
        print(
            f"  {key:<8} {name:<12} "
            f"{res['dice_lv']:>9.4f} "
            f"{res['dice_rv']:>9.4f} "
            f"{res['dice_myo']:>10.4f} "
            f"{res['mean_dice']:>11.4f} "
            f"{res['mean_hd95']:>9.2f}mm"
        )
    print("=" * 80)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot M&Ms generalisation evaluation"
    )
    parser.add_argument("--config",
                        required=True,
                        help="Model training config, e.g. configs/train_lweunet_v2.yaml")
    parser.add_argument("--checkpoint",
                        required=True,
                        help="Model checkpoint, e.g. checkpoints/lweunet_v2/best_model.pth")
    parser.add_argument("--preprocessed_dir",
                        default="data/MnM/preprocessed",
                        help="Path to preprocessed M&Ms slices")
    parser.add_argument("--model_label",
                        default="LWEU-Net_V2",
                        help="Label used in output filenames and tables")
    parser.add_argument("--device",
                        default=None,
                        help="cuda or cpu (auto-detected if not set)")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = (torch.device(args.device) if args.device else
              torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    logger.info(f"Device        : {device}")
    logger.info(f"Model label   : {args.model_label}")
    logger.info(f"Checkpoint    : {args.checkpoint}")
    logger.info(f"Preprocessed  : {args.preprocessed_dir}")

    # ── Load model ────────────────────────────────────────────
    model = build_model(cfg, args.checkpoint, device)

    # ── Vendors to evaluate ───────────────────────────────────
    vendors = [
        ("A", "A (Siemens)"),
        ("B", "B (Philips)"),
        ("C", "C (GE)"),
        ("D", "D (Canon)"),
    ]

    all_results = {}

    logger.info("\n" + "=" * 60)
    logger.info("  EVALUATING PER VENDOR")
    logger.info("=" * 60)

    for vendor_code, vendor_display in vendors:
        logger.info(f"\n[Vendor {vendor_display}]")
        results = run_vendor_evaluation(
            model            = model,
            preprocessed_dir = args.preprocessed_dir,
            vendor           = vendor_code,
            vendor_name      = vendor_display,
            cfg              = cfg,
            device           = device,
        )
        all_results[vendor_code] = results

    # ── Overall (all vendors combined) ────────────────────────
    logger.info("\n[Overall — all vendors combined]")
    overall_results = run_vendor_evaluation(
        model            = model,
        preprocessed_dir = args.preprocessed_dir,
        vendor           = None,
        vendor_name      = "Overall",
        cfg              = cfg,
        device           = device,
    )
    all_results["Overall"] = overall_results

    # ── Print thesis table ────────────────────────────────────
    print_vendor_table(all_results, args.model_label)

    # ── Save JSON ─────────────────────────────────────────────
    log_dir  = Path(cfg.get("log_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / f"mnms_eval_{args.model_label}.json"

    # Convert any non-serialisable values before saving
    serialisable = {}
    for key, res in all_results.items():
        serialisable[key] = {k: float(v) if isinstance(v, (np.floating, float))
                             else v for k, v in res.items()}

    with open(out_path, "w") as f:
        json.dump(serialisable, f, indent=2)

    logger.info(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
