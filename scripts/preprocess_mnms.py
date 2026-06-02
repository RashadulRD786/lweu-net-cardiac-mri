# scripts/preprocess_mnms.py
#
# Responsibility: Preprocess M&Ms Testing split patients for zero-shot
# generalisation evaluation.
#
# Why Testing split only:
#   The Testing split (136 patients) is the official M&Ms benchmark set.
#   Results computed on it are directly comparable to published challenge
#   results (Table VIII, Table IX in Campello et al. 2021).
#   Using the official test split is the most defensible choice for thesis
#   and journal reporting.
#
# Two adaptations from the ACDC pipeline:
#
#   1. 4D NIfTI extraction — M&Ms stores all time frames in one file
#      ({ID}_sa.nii). ED and ES frame indices are read from the dataset CSV.
#      The correct 3D volume is extracted as: volume_4d[:, :, :, frame_idx]
#
#   2. Label remapping — M&Ms uses LV=1, MYO=2, RV=3.
#      Remapped to ACDC convention (RV=1, MYO=2, LV=3) before any
#      processing. Without this, LV and RV Dice scores would be computed
#      against the wrong predicted classes.
#
# All other steps — resample, resize, normalize, verify — import and call
# the existing ACDC functions unchanged. This guarantees an identical
# preprocessing pipeline for a valid zero-shot comparison.
#
# Output:
#   data/MnM/preprocessed/
#       A0S9V9_ED_slice00_img.npy   ← float32 (224, 224)
#       A0S9V9_ED_slice00_msk.npy   ← uint8   (224, 224), labels {0,1,2,3}
#       ...
#   data/MnM/preprocessed/slice_metadata.csv
#       → maps every slice file to patient, vendor, phase, pathology
#       → used by evaluate_mnms.py to compute per-vendor Dice and HD95
#
# Usage:
#   cd /home/rashadulnafisriyad/FYP_UNet/LWEU_NET
#   conda activate lweunet
#   python scripts/preprocess_mnms.py

import sys
import time
import yaml
import logging
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

# ── Make src/ importable ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.resample      import resample_slice
from src.preprocessing.resize        import resize_pair
from src.preprocessing.normalize     import normalize_slice
from src.preprocessing.verify_labels import (verify_mask,
                                              clean_mask,
                                              verify_alignment)
from src.preprocessing.pipeline      import setup_logging, save_samples


# ── Label remapping ───────────────────────────────────────────────────────────

def remap_mnms_labels(mask_2d: np.ndarray) -> np.ndarray:
    """
    Remap M&Ms label convention to ACDC convention.

    M&Ms ground truth:  LV=1, MYO=2, RV=3
    ACDC / model:       RV=1, MYO=2, LV=3

    MYO (label 2) is identical in both — no change needed.
    Only LV and RV are swapped.

    This must be applied BEFORE resample/resize so that nearest-neighbour
    interpolation operates on the correctly mapped integer labels.

    Args:
        mask_2d : 2D uint8 array with M&Ms labels {0,1,2,3}
    Returns:
        remapped : 2D uint8 array with ACDC labels {0,1,2,3}
    """
    remapped = np.zeros_like(mask_2d, dtype=np.uint8)
    remapped[mask_2d == 1] = 3   # LV  : 1 → 3
    remapped[mask_2d == 2] = 2   # MYO : 2 → 2  (unchanged)
    remapped[mask_2d == 3] = 1   # RV  : 3 → 1
    return remapped


# ── Core patient processing ───────────────────────────────────────────────────

def process_single_patient_mnms(
    patient_id     : str,
    patient_dir    : Path,
    ed_frame       : int,
    es_frame       : int,
    vendor         : str,
    target_spacing : float,
    target_size    : tuple,
    clip_low       : float,
    clip_high      : float,
) -> tuple:
    """
    Run the full preprocessing pipeline for one M&Ms patient.
    Processes both ED and ES phases, all slices.

    Differences from ACDC process_single_patient():
      - Loads a single 4D NIfTI and extracts 3D frames by index
      - Applies remap_mnms_labels() before resampling
      - Uses vendor letter code as the 'group' field

    Args:
        patient_id     : e.g. "A1K2P5"
        patient_dir    : Path to the patient folder
        ed_frame       : 0-indexed ED frame number from CSV
        es_frame       : 0-indexed ES frame number from CSV
        vendor         : letter code e.g. "D" — used as group for reporting
        target_spacing : mm/pixel (1.5 — identical to ACDC)
        target_size    : pixels   ((224,224) — identical to ACDC)
        clip_low       : percentile lower bound (0.5 — identical to ACDC)
        clip_high      : percentile upper bound (99.5 — identical to ACDC)

    Returns:
        samples  : list of sample dicts (same structure as ACDC pipeline)
        warnings : list of warning strings for logging
    """
    img_path = patient_dir / f"{patient_id}_sa.nii"
    msk_path = patient_dir / f"{patient_id}_sa_gt.nii"

    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    if not msk_path.exists():
        raise FileNotFoundError(f"Mask not found:  {msk_path}")

    # ── Load 4D NIfTI files ───────────────────────────────────
    # Shape: (H, W, Z, T) — T is the number of cardiac cycle time frames
    img_nii = nib.load(str(img_path))
    msk_nii = nib.load(str(msk_path))

    img_4d  = img_nii.get_fdata().astype(np.float32)
    msk_4d  = msk_nii.get_fdata().astype(np.uint8)

    # ── In-plane pixel spacing from NIfTI header ──────────────
    # get_zooms() returns (spacing_x, spacing_y, spacing_z, spacing_t)
    # Index 0 is the in-plane spacing in mm/pixel
    spacing  = float(img_nii.header.get_zooms()[0])
    n_frames = img_4d.shape[3]

    samples  = []
    warnings = []

    # ── Process ED and ES phases ──────────────────────────────
    for phase_name, frame_idx in [("ED", ed_frame), ("ES", es_frame)]:

        # Guard against invalid frame indices
        if frame_idx >= n_frames:
            msg = (f"{patient_id} | {phase_name} | "
                   f"frame {frame_idx} out of range "
                   f"({n_frames} frames total) — skipped")
            warnings.append(msg)
            logging.warning(f"  SKIP | {msg}")
            continue

        # ── Extract 3D volume for this phase ──────────────────
        # img_4d[:,:,:,frame_idx] → shape (H, W, Z)
        img_vol  = img_4d[:, :, :, frame_idx]
        msk_vol  = msk_4d[:, :, :, frame_idx]
        n_slices = img_vol.shape[2]

        for s_idx in range(n_slices):
            img_2d = img_vol[:, :, s_idx]
            msk_2d = msk_vol[:, :, s_idx]

            # Step 1 — Remap M&Ms labels to ACDC convention
            # Done first so all downstream functions see correct labels
            msk_2d = remap_mnms_labels(msk_2d)

            # Step 2 — Resample to uniform 1.5 mm/pixel
            img_2d = resample_slice(
                img_2d, spacing, target_spacing, is_mask=False
            )
            msk_2d = resample_slice(
                msk_2d, spacing, target_spacing, is_mask=True
            )

            # Step 3 — Resize to fixed 224×224 pixels
            img_2d, msk_2d = resize_pair(img_2d, msk_2d, target_size)

            # Step 4 — Z-score normalize image (mask is never touched)
            img_2d = normalize_slice(img_2d, clip_low, clip_high)

            # Step 5 — Verify and clean labels
            result = verify_mask(msk_2d, patient_id, s_idx, phase_name)
            if not result["is_clean"]:
                warnings.append(
                    f"{patient_id} | {phase_name} | slice {s_idx} "
                    f"— cleaned labels: {result['corrupt_vals']}"
                )
                msk_2d = clean_mask(msk_2d, patient_id, s_idx)

            # Step 6 — Verify spatial alignment
            if not verify_alignment(img_2d, msk_2d, patient_id, s_idx):
                raise ValueError(
                    f"Shape mismatch: {img_2d.shape} vs {msk_2d.shape} "
                    f"for {patient_id} {phase_name} slice {s_idx}"
                )

            samples.append({
                "img"       : img_2d,
                "msk"       : msk_2d,
                "patient_id": patient_id,
                "group"     : vendor,
                "phase"     : phase_name,
                "slice_idx" : s_idx,
            })

    return samples, warnings


# ── Validation ────────────────────────────────────────────────────────────────

def validate_environment(cfg: dict) -> bool:
    """Check all required paths exist before starting."""
    print("\n[Validation] Checking environment...")
    all_ok = True

    mnms_root  = Path(cfg["paths"]["mnms_root"])
    test_dir   = mnms_root / "Testing"
    meta_csv   = Path(cfg["paths"]["metadata_csv"])
    output_dir = Path(cfg["paths"]["output_dir"])

    # Check Testing folder
    if not test_dir.exists():
        print(f"  ✗ Testing folder not found : {test_dir}")
        all_ok = False
    else:
        n = len([d for d in test_dir.iterdir() if d.is_dir()])
        print(f"  ✓ Testing folder found     : {n} patients")

    # Check metadata CSV
    if not meta_csv.exists():
        print(f"  ✗ Metadata CSV not found   : {meta_csv}")
        all_ok = False
    else:
        df = pd.read_csv(meta_csv)
        df = df.rename(columns={"External code": "patient_id"})

        # Count how many CSV rows match Testing patients
        testing_ids   = set(
            p.name for p in test_dir.iterdir() if p.is_dir()
        ) if test_dir.exists() else set()
        test_df = df[df["patient_id"].isin(testing_ids)]

        print(f"  ✓ Metadata CSV found       : {len(df)} total patients")
        print(f"  ✓ Testing patients matched : {len(test_df)} in CSV")
        vendor_counts = test_df["Vendor"].value_counts().to_dict()
        print(f"  ✓ Vendor distribution      : {vendor_counts}")

    # Check output is writable
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        test_file = output_dir / ".write_test"
        test_file.touch()
        test_file.unlink()
        print(f"  ✓ Output dir writable      : {output_dir}")
    except Exception as e:
        print(f"  ✗ Output dir not writable  : {e}")
        all_ok = False

    spacing = cfg["preprocessing"]["target_spacing"]
    size    = cfg["preprocessing"]["target_size"]
    print(f"  ✓ Target spacing           : {spacing}mm  (ACDC: 1.5)")
    print(f"  ✓ Target size              : {size[0]}×{size[1]}px (ACDC: 224×224)")

    return all_ok


# ── Final report ──────────────────────────────────────────────────────────────

def print_final_report(vendor_stats  : dict,
                       failed_patients: list,
                       total_warnings : int,
                       elapsed        : float,
                       output_dir     : Path) -> None:
    """Print a summary report after all patients are processed."""
    print("\n" + "=" * 60)
    print("M&Ms PREPROCESSING COMPLETE — FINAL REPORT")
    print("=" * 60)
    print(f"  Total time       : {elapsed/60:.1f} minutes")
    print()
    print(f"  {'Vendor':<8} {'Name':<10} {'Patients':>9} {'Slices':>8}")
    print(f"  {'-' * 40}")
    total_patients = 0
    total_slices   = 0
    for vendor, stats in sorted(vendor_stats.items()):
        print(f"  {vendor:<8} {stats['vendor_name']:<10} "
              f"{stats['patients']:>9} {stats['slices']:>8}")
        total_patients += stats["patients"]
        total_slices   += stats["slices"]
    print(f"  {'-' * 40}")
    print(f"  {'TOTAL':<20} {total_patients:>9} {total_slices:>8}")
    print()
    print(f"  Failed patients  : {len(failed_patients)} "
          f"{'✓' if not failed_patients else '✗ ' + str(failed_patients)}")
    print(f"  Label warnings   : {total_warnings} "
          f"{'✓' if total_warnings == 0 else '⚠  check log'}")
    print()
    print(f"  Output directory : {output_dir}")
    print(f"  Metadata CSV     : {output_dir}/slice_metadata.csv")
    print()
    if not failed_patients and total_warnings == 0:
        print("  STATUS: ✓ All patients processed cleanly")
    elif not failed_patients:
        print("  STATUS: ⚠ Completed with label warnings — check log")
    else:
        print("  STATUS: ✗ Some patients failed — check log")
    print("=" * 60)


# ── Main entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("  M&Ms PREPROCESSING PIPELINE")
    print("  Testing Split — Zero-Shot Generalisation")
    print("=" * 60)

    # ── Load config ───────────────────────────────────────────
    config_path = PROJECT_ROOT / "configs" / "preprocessing_config_mnms.yaml"
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

    # ── Load config values ────────────────────────────────────
    MNMS_ROOT      = Path(cfg["paths"]["mnms_root"])
    TEST_DIR       = MNMS_ROOT / "Testing"
    META_CSV_PATH  = Path(cfg["paths"]["metadata_csv"])
    OUTPUT_DIR     = Path(cfg["paths"]["output_dir"])
    TARGET_SPACING = float(cfg["preprocessing"]["target_spacing"])
    TARGET_SIZE    = tuple(cfg["preprocessing"]["target_size"])
    CLIP_LOW       = float(cfg["preprocessing"]["clip_low"])
    CLIP_HIGH      = float(cfg["preprocessing"]["clip_high"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load metadata CSV and filter to Testing patients only ─
    meta_df = pd.read_csv(META_CSV_PATH)
    meta_df = meta_df.rename(columns={"External code": "patient_id"})

    testing_ids = set(
        p.name for p in TEST_DIR.iterdir() if p.is_dir()
    )
    meta_df = meta_df[meta_df["patient_id"].isin(testing_ids)].reset_index(drop=True)

    logging.info("=" * 60)
    logging.info("M&Ms PREPROCESSING — TESTING SPLIT")
    logging.info(f"  Patients to process : {len(meta_df)}")
    logging.info(f"  Target spacing      : {TARGET_SPACING}mm")
    logging.info(f"  Target size         : {TARGET_SIZE}")
    logging.info("=" * 60)

    # ── Processing loop ───────────────────────────────────────
    all_metadata_rows = []
    all_warnings      = []
    failed_patients   = []
    vendor_stats      = {}

    start_time = time.time()

    for _, row in tqdm(
        meta_df.iterrows(),
        total = len(meta_df),
        desc  = "  Processing",
        unit  = "patient",
    ):
        pid         = str(row["patient_id"])
        vendor      = str(row["Vendor"])
        vendor_name = str(row["VendorName"])
        ed_frame    = int(row["ED"])
        es_frame    = int(row["ES"])
        pathology   = str(row["Pathology"])

        if vendor not in vendor_stats:
            vendor_stats[vendor] = {
                "vendor_name": vendor_name,
                "patients"   : 0,
                "slices"     : 0,
            }

        patient_dir = TEST_DIR / pid

        try:
            samples, warnings = process_single_patient_mnms(
                patient_id     = pid,
                patient_dir    = patient_dir,
                ed_frame       = ed_frame,
                es_frame       = es_frame,
                vendor         = vendor,
                target_spacing = TARGET_SPACING,
                target_size    = TARGET_SIZE,
                clip_low       = CLIP_LOW,
                clip_high      = CLIP_HIGH,
            )

            saved = save_samples(samples, OUTPUT_DIR)

            # Build metadata rows for this patient's slices
            for sample in samples:
                all_metadata_rows.append({
                    "filename_base" : (f"{sample['patient_id']}_"
                                      f"{sample['phase']}_"
                                      f"slice{sample['slice_idx']:02d}"),
                    "patient_id"    : sample["patient_id"],
                    "vendor"        : vendor,
                    "vendor_name"   : vendor_name,
                    "phase"         : sample["phase"],
                    "slice_idx"     : sample["slice_idx"],
                    "pathology"     : pathology,
                })

            all_warnings.extend(warnings)
            vendor_stats[vendor]["patients"] += 1
            vendor_stats[vendor]["slices"]   += saved

            logging.info(
                f"  OK | {pid} | Vendor {vendor} "
                f"({vendor_name}) | {saved} slices"
            )

        except Exception as e:
            failed_patients.append(pid)
            logging.error(f"  FAILED | {pid} | {e}")
            continue

    elapsed = time.time() - start_time

    # ── Save metadata CSV ─────────────────────────────────────
    if all_metadata_rows:
        metadata_df   = pd.DataFrame(all_metadata_rows)
        metadata_path = OUTPUT_DIR / "slice_metadata.csv"
        metadata_df.to_csv(metadata_path, index=False)
        logging.info(f"Metadata CSV saved → {metadata_path} "
                     f"({len(metadata_df)} rows)")

    if all_warnings:
        logging.warning(f"\n{len(all_warnings)} warnings:")
        for w in all_warnings:
            logging.warning(f"  {w}")

    if failed_patients:
        logging.error(
            f"\nFailed patients ({len(failed_patients)}): {failed_patients}"
        )

    # ── Final report ──────────────────────────────────────────
    print_final_report(
        vendor_stats   = vendor_stats,
        failed_patients= failed_patients,
        total_warnings = len(all_warnings),
        elapsed        = elapsed,
        output_dir     = OUTPUT_DIR,
    )
