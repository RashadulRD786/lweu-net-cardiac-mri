# scripts/run_preprocessing.py
#
# Responsibility: Entry point for running the complete preprocessing
# pipeline on all 150 ACDC patients from the terminal.
#
# This script:
#   1. Loads the preprocessing config
#   2. Validates all required directories exist
#   3. Checks required split CSVs exist
#   4. Runs the full pipeline on all 150 patients
#   5. Produces a final report
#
# Usage:
#   python scripts/run_preprocessing.py
#
# Expected runtime: 10-20 minutes depending on hardware
# Expected output : ~2,900 .npy file pairs in data/preprocessed/

import sys
import time
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# ── Make src/ importable from scripts/ ───────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.pipeline import (
    process_single_patient,
    save_samples,
    setup_logging,
)


def validate_environment(cfg: dict) -> bool:
    """
    Check all required directories and files exist before
    starting the pipeline. Fail early with clear messages
    rather than crashing halfway through processing.

    Args:
        cfg : loaded config dict

    Returns:
        valid : True if all checks pass, False otherwise
    """
    print("\n[Validation] Checking environment...")
    all_ok = True

    # ── Check raw data directories ────────────────────────────
    raw_train = Path(cfg["paths"]["raw_train_dir"])
    raw_test  = Path(cfg["paths"]["raw_test_dir"])

    if not raw_train.exists():
        print(f"  ✗ Raw training dir not found : {raw_train}")
        all_ok = False
    else:
        n_train = len([d for d in raw_train.iterdir() if d.is_dir()])
        print(f"  ✓ Raw training dir found     : {n_train} patients")

    if not raw_test.exists():
        print(f"  ✗ Raw testing dir not found  : {raw_test}")
        all_ok = False
    else:
        n_test = len([d for d in raw_test.iterdir() if d.is_dir()])
        print(f"  ✓ Raw testing dir found      : {n_test} patients")

    # ── Check split CSV files exist ───────────────────────────
    splits_dir = Path(cfg["paths"]["splits_dir"])
    for split_file in ["train_patients.csv",
                        "val_patients.csv",
                        "test_patients.csv"]:
        split_path = splits_dir / split_file
        if not split_path.exists():
            print(f"  ✗ Split file not found : {split_path}")
            print(f"    → Run: python -m src.preprocessing.generate_split")
            all_ok = False
        else:
            df = pd.read_csv(split_path)
            print(f"  ✓ {split_file:<30} ({len(df)} patients)")

    # ── Check output directory is writable ────────────────────
    output_dir = Path(cfg["paths"]["output_dir"])
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        test_file = output_dir / ".write_test"
        test_file.touch()
        test_file.unlink()
        print(f"  ✓ Output directory writable  : {output_dir}")
    except Exception as e:
        print(f"  ✗ Output directory not writable: {e}")
        all_ok = False

    # ── Check preprocessing parameters ───────────────────────
    spacing = cfg["preprocessing"]["target_spacing"]
    size    = cfg["preprocessing"]["target_size"]
    print(f"  ✓ Target spacing             : {spacing}mm")
    print(f"  ✓ Target size                : {size[0]}×{size[1]}px")

    return all_ok


def run_split(
    split_name     : str,
    split_df       : pd.DataFrame,
    raw_dir        : Path,
    output_dir     : Path,
    target_spacing : float,
    target_size    : tuple,
    clip_low       : float,
    clip_high      : float,
) -> dict:
    """
    Process all patients in one split (train, val, or test).

    Args:
        split_name     : "train", "val", or "test"
        split_df       : DataFrame with patient_id and group
        raw_dir        : Path to raw NIfTI files
        output_dir     : Path to save preprocessed .npy files
        target_spacing : mm/pixel target
        target_size    : (height, width) pixel target
        clip_low       : percentile lower bound for clipping
        clip_high      : percentile upper bound for clipping

    Returns:
        stats : dict with slice counts and failure info
    """
    from tqdm import tqdm

    logging.info(f"\n{'='*60}")
    logging.info(f"Processing {split_name.upper()} "
                 f"({len(split_df)} patients)")
    logging.info(f"{'='*60}")

    split_output = output_dir / split_name
    split_output.mkdir(parents=True, exist_ok=True)

    total_slices    = 0
    failed_patients = []
    all_warnings    = []
    group_counts    = {}

    for _, row in tqdm(
        split_df.iterrows(),
        total = len(split_df),
        desc  = f"  {split_name:<6}",
        unit  = "patient",
    ):
        pid         = row["patient_id"]
        group       = row["group"]
        patient_dir = raw_dir / pid

        try:
            samples, warnings = process_single_patient(
                patient_dir,
                target_spacing,
                target_size,
                clip_low,
                clip_high,
            )

            saved = save_samples(samples, split_output)
            total_slices += saved
            all_warnings.extend(warnings)

            # Track per-group slice counts
            if group not in group_counts:
                group_counts[group] = 0
            group_counts[group] += saved

            logging.info(
                f"  OK | {pid} | {group} | {saved} slices"
            )

        except Exception as e:
            failed_patients.append(pid)
            logging.error(f"  FAILED | {pid} | {e}")
            continue

    return {
        "split"          : split_name,
        "total_slices"   : total_slices,
        "failed_patients": failed_patients,
        "warnings"       : all_warnings,
        "group_counts"   : group_counts,
    }


def print_final_report(results: list, elapsed: float) -> None:
    """
    Print a comprehensive final report after all splits are processed.
    This is the summary you can reference in your thesis methodology.
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE — FINAL REPORT")
    print("=" * 60)
    print(f"  Total time : {elapsed/60:.1f} minutes")
    print()

    total_slices   = 0
    total_failures = 0
    total_warnings = 0

    for result in results:
        split   = result["split"].upper()
        slices  = result["total_slices"]
        failed  = len(result["failed_patients"])
        warned  = len(result["warnings"])

        total_slices   += slices
        total_failures += failed
        total_warnings += warned

        print(f"  {split:<6} split:")
        print(f"    Slices saved    : {slices}")
        print(f"    Failed patients : "
              f"{failed} "
              f"{'✓' if failed == 0 else '✗ ' + str(result['failed_patients'])}")
        print(f"    Label warnings  : "
              f"{warned} "
              f"{'✓' if warned == 0 else '⚠'}")

        if result["group_counts"]:
            print(f"    Group breakdown :")
            for grp, cnt in sorted(result["group_counts"].items()):
                print(f"      {grp:<6} : {cnt} slices")
        print()

    print(f"  TOTAL slices saved : {total_slices}")
    print(f"  TOTAL failures     : {total_failures}")
    print(f"  TOTAL warnings     : {total_warnings}")
    print()

    if total_failures == 0 and total_warnings == 0:
        print("  STATUS: ✓ All patients processed cleanly")
    elif total_failures == 0:
        print("  STATUS: ⚠ Completed with label warnings — check log")
    else:
        print("  STATUS: ✗ Some patients failed — check log")

    print("=" * 60)
    print(f"\n  Preprocessed data saved to:")
    print(f"    data/preprocessed/train/")
    print(f"    data/preprocessed/val/")
    print(f"    data/preprocessed/test/")
    print(f"\n  Log saved to:")
    print(f"    logs/preprocessing.log")
    print()


# ── Main entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("  LWEU-NET PREPROCESSING PIPELINE")
    print("=" * 60)

    # ── Load config ───────────────────────────────────────────
    config_path = PROJECT_ROOT / "configs" / "preprocessing_config.yaml"

    if not config_path.exists():
        print(f"✗ Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print(f"  Config loaded: {config_path}")

    # ── Validate environment ──────────────────────────────────
    if not validate_environment(cfg):
        print("\n✗ Validation failed. Fix the issues above and retry.")
        sys.exit(1)

    print("\n  All checks passed. Starting pipeline...\n")

    # ── Setup logging ─────────────────────────────────────────
    setup_logging(PROJECT_ROOT / "logs")

    # ── Load paths and parameters ─────────────────────────────
    RAW_TRAIN_DIR  = Path(cfg["paths"]["raw_train_dir"])
    RAW_TEST_DIR   = Path(cfg["paths"]["raw_test_dir"])
    OUTPUT_DIR     = Path(cfg["paths"]["output_dir"])
    SPLITS_DIR     = Path(cfg["paths"]["splits_dir"])

    TARGET_SPACING = float(cfg["preprocessing"]["target_spacing"])
    TARGET_SIZE    = tuple(cfg["preprocessing"]["target_size"])
    CLIP_LOW       = float(cfg["preprocessing"]["clip_low"])
    CLIP_HIGH      = float(cfg["preprocessing"]["clip_high"])

    # ── Load split CSVs ───────────────────────────────────────
    train_df = pd.read_csv(SPLITS_DIR / "train_patients.csv")
    val_df   = pd.read_csv(SPLITS_DIR / "val_patients.csv")
    test_df  = pd.read_csv(SPLITS_DIR / "test_patients.csv")

    # ── Run all three splits ──────────────────────────────────
    start_time = time.time()
    results    = []

    for split_name, split_df, raw_dir in [
        ("train", train_df, RAW_TRAIN_DIR),
        ("val",   val_df,   RAW_TRAIN_DIR),
        ("test",  test_df,  RAW_TEST_DIR),
    ]:
        result = run_split(
            split_name     = split_name,
            split_df       = split_df,
            raw_dir        = raw_dir,
            output_dir     = OUTPUT_DIR,
            target_spacing = TARGET_SPACING,
            target_size    = TARGET_SIZE,
            clip_low       = CLIP_LOW,
            clip_high      = CLIP_HIGH,
        )
        results.append(result)

    elapsed = time.time() - start_time

    # ── Print final report ────────────────────────────────────
    print_final_report(results, elapsed)