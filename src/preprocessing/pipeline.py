# src/preprocessing/pipeline.py
#
# Responsibility: Assemble all preprocessing steps into one complete
# pipeline and process every patient in the dataset.
#
# This file is the conductor — it imports and calls every individual
# step in the correct order:
#
#   extract_phases   → find ED and ES frame paths
#   slice_converter  → convert 3D volume to 2D slices
#   resample         → uniform 1.5mm pixel spacing
#   resize           → fixed 224×224 spatial dimensions
#   normalize        → z-score per image (foreground, with clipping)
#   verify_labels    → confirm mask integrity
#
# Output structure:
#   data/preprocessed/
#       train/
#           patient001_ED_slice00_img.npy
#           patient001_ED_slice00_msk.npy
#           patient001_ED_slice01_img.npy
#           ...
#       val/
#           patient021_ED_slice00_img.npy
#           ...
#       test/
#           patient101_ED_slice00_img.npy
#           ...
#
# Each .npy pair is one training sample:
#   _img.npy → float32 (224, 224) normalized image
#   _msk.npy → uint8   (224, 224) label mask {0,1,2,3}

import numpy as np
import pandas as pd
import yaml
import logging
import time
from pathlib import Path
from tqdm import tqdm

from src.preprocessing.extract_phases   import get_phase_paths
from src.preprocessing.slice_converter  import volume_to_slices
from src.preprocessing.resample         import resample_slice
from src.preprocessing.resize           import resize_pair
from src.preprocessing.normalize        import normalize_slice
from src.preprocessing.verify_labels    import (verify_mask,
                                                 clean_mask,
                                                 verify_alignment)


def setup_logging(log_dir: Path) -> None:
    """
    Configure logging to both terminal and log file.
    Every patient processed is recorded with timestamp.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "preprocessing.log"

    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s | %(levelname)s | %(message)s",
        handlers= [
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ]
    )
    logging.info(f"Logging initialized → {log_path}")


def process_single_patient(
    patient_dir     : Path,
    target_spacing  : float,
    target_size     : tuple,
    clip_low        : float,
    clip_high       : float,
) -> list:
    """
    Run the complete preprocessing pipeline for one patient.
    Processes both ED and ES phases, all slices.

    Args:
        patient_dir    : Path to patient folder
        target_spacing : mm/pixel after resampling (1.5)
        target_size    : pixel dimensions after resize (224, 224)
        clip_low       : lower percentile for normalization clipping
        clip_high      : upper percentile for normalization clipping

    Returns:
        samples : list of dicts, each containing:
                  img       → float32 (224, 224) normalized image
                  msk       → uint8   (224, 224) clean mask
                  patient_id→ str
                  group     → str
                  phase     → str (ED or ES)
                  slice_idx → int
    """
    pid    = patient_dir.name
    samples = []
    warnings = []

    # ── Step 1: Extract phase paths ───────────────────────────
    phase_info = get_phase_paths(patient_dir)
    group      = phase_info["group"]

    # ── Process both ED and ES phases ─────────────────────────
    for phase_name, img_path, msk_path in [
        ("ED", phase_info["ed_img_path"], phase_info["ed_msk_path"]),
        ("ES", phase_info["es_img_path"], phase_info["es_msk_path"]),
    ]:
        # ── Step 2: Convert 3D volume to 2D slices ────────────
        img_slices, spacing, n_slices = volume_to_slices(
            img_path, is_mask=False
        )
        msk_slices, _, _ = volume_to_slices(
            msk_path, is_mask=True
        )

        for s_idx in range(n_slices):
            img_2d = img_slices[s_idx]
            msk_2d = msk_slices[s_idx]

            # ── Step 3: Resample to uniform spacing ───────────
            img_2d = resample_slice(
                img_2d, spacing, target_spacing, is_mask=False
            )
            msk_2d = resample_slice(
                msk_2d, spacing, target_spacing, is_mask=True
            )

            # ── Step 4: Resize to fixed dimensions ────────────
            img_2d, msk_2d = resize_pair(img_2d, msk_2d, target_size)

            # ── Step 5: Z-score normalize ─────────────────────
            img_2d = normalize_slice(img_2d, clip_low, clip_high)

            # ── Step 6: Verify and clean labels ───────────────
            result = verify_mask(msk_2d, pid, s_idx, phase_name)

            if not result["is_clean"]:
                warnings.append(
                    f"{pid} | {phase_name} | slice {s_idx} "
                    f"— cleaned labels: {result['corrupt_vals']}"
                )
                msk_2d = clean_mask(msk_2d, pid, s_idx)

            # ── Verify alignment ──────────────────────────────
            if not verify_alignment(img_2d, msk_2d, pid, s_idx):
                raise ValueError(
                    f"Shape mismatch after pipeline: "
                    f"{img_2d.shape} vs {msk_2d.shape} "
                    f"for {pid} {phase_name} slice {s_idx}"
                )

            samples.append({
                "img"       : img_2d,
                "msk"       : msk_2d,
                "patient_id": pid,
                "group"     : group,
                "phase"     : phase_name,
                "slice_idx" : s_idx,
            })

    return samples, warnings


def save_samples(samples: list, output_dir: Path) -> int:
    """
    Save all processed samples as numpy arrays.
    Each sample becomes two .npy files: _img and _msk.

    Args:
        samples    : list of sample dicts from process_single_patient
        output_dir : split-specific output directory
                     e.g. data/preprocessed/train/

    Returns:
        count : number of samples saved
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        pid   = sample["patient_id"]
        phase = sample["phase"]
        sidx  = sample["slice_idx"]

        base_name = f"{pid}_{phase}_slice{sidx:02d}"

        np.save(output_dir / f"{base_name}_img.npy", sample["img"])
        np.save(output_dir / f"{base_name}_msk.npy", sample["msk"])

    return len(samples)


def run_pipeline(cfg: dict) -> None:
    """
    Run the complete preprocessing pipeline for all patients.
    Reads split CSVs, processes each patient, saves to disk.

    Args:
        cfg : loaded preprocessing config dict
    """
    # ── Paths from config ─────────────────────────────────────
    RAW_TRAIN_DIR = Path(cfg["paths"]["raw_train_dir"])
    RAW_TEST_DIR  = Path(cfg["paths"]["raw_test_dir"])
    OUTPUT_DIR    = Path(cfg["paths"]["output_dir"])
    SPLITS_DIR    = Path(cfg["paths"]["splits_dir"])
    LOG_DIR       = Path("logs")

    TARGET_SPACING = float(cfg["preprocessing"]["target_spacing"])
    TARGET_SIZE    = tuple(cfg["preprocessing"]["target_size"])
    CLIP_LOW       = float(cfg["preprocessing"]["clip_low"])
    CLIP_HIGH      = float(cfg["preprocessing"]["clip_high"])

    # ── Setup logging ─────────────────────────────────────────
    setup_logging(LOG_DIR)
    logging.info("=" * 60)
    logging.info("PREPROCESSING PIPELINE STARTED")
    logging.info(f"  Target spacing : {TARGET_SPACING}mm")
    logging.info(f"  Target size    : {TARGET_SIZE}")
    logging.info(f"  Clip range     : [{CLIP_LOW}, {CLIP_HIGH}]")
    logging.info("=" * 60)

    # ── Load split CSVs ───────────────────────────────────────
    train_df = pd.read_csv(SPLITS_DIR / "train_patients.csv")
    val_df   = pd.read_csv(SPLITS_DIR / "val_patients.csv")
    test_df  = pd.read_csv(SPLITS_DIR / "test_patients.csv")

    splits = [
        ("train", train_df, RAW_TRAIN_DIR),
        ("val",   val_df,   RAW_TRAIN_DIR),
        ("test",  test_df,  RAW_TEST_DIR),
    ]

    # ── Track overall statistics ──────────────────────────────
    total_stats   = {}
    all_warnings  = []
    failed_patients = []
    start_time    = time.time()

    # ── Process each split ────────────────────────────────────
    for split_name, split_df, raw_dir in splits:
        logging.info(f"\nProcessing {split_name.upper()} split "
                     f"({len(split_df)} patients)...")

        output_dir  = OUTPUT_DIR / split_name
        slice_count = 0

        for _, row in tqdm(
            split_df.iterrows(),
            total      = len(split_df),
            desc       = f"  {split_name}",
            unit       = "patient",
        ):
            pid         = row["patient_id"]
            patient_dir = raw_dir / pid

            try:
                samples, warnings = process_single_patient(
                    patient_dir,
                    TARGET_SPACING,
                    TARGET_SIZE,
                    CLIP_LOW,
                    CLIP_HIGH,
                )

                saved = save_samples(samples, output_dir)
                slice_count += saved

                if warnings:
                    all_warnings.extend(warnings)

                logging.info(
                    f"  {pid} | {row['group']} | "
                    f"{saved} slices saved"
                )

            except Exception as e:
                failed_patients.append(pid)
                logging.error(f"  FAILED: {pid} — {e}")
                continue

        total_stats[split_name] = slice_count
        logging.info(
            f"{split_name.upper()} complete: {slice_count} slices saved"
        )

    # ── Final report ──────────────────────────────────────────
    elapsed = time.time() - start_time
    logging.info("\n" + "=" * 60)
    logging.info("PREPROCESSING COMPLETE")
    logging.info(f"  Total time     : {elapsed/60:.1f} minutes")
    logging.info(f"  Train slices   : {total_stats.get('train', 0)}")
    logging.info(f"  Val slices     : {total_stats.get('val',   0)}")
    logging.info(f"  Test slices    : {total_stats.get('test',  0)}")
    logging.info(
        f"  Total slices   : {sum(total_stats.values())}"
    )

    if all_warnings:
        logging.warning(
            f"\n  {len(all_warnings)} label warnings during processing:"
        )
        for w in all_warnings:
            logging.warning(f"    {w}")
    else:
        logging.info("  Label warnings : None")

    if failed_patients:
        logging.error(
            f"\n  Failed patients ({len(failed_patients)}): "
            f"{failed_patients}"
        )
    else:
        logging.info("  Failed patients: None")

    logging.info("=" * 60)


# ── Quick test ────────────────────────────────────────────────────────────────
# Test the pipeline on 3 patients before running on all 150:
#   python -m src.preprocessing.pipeline

if __name__ == "__main__":

    test_patients = [
        "patient001",  # DCM
        "patient021",  # HCM
        "patient041",  # MINF
    ]

    RAW_DIR     = Path("data/raw/training")
    OUTPUT_DIR  = Path("data/preprocessed_test")
    TARGET_SIZE = (224, 224)

    print("Testing pipeline.py on 3 patients...")
    print("=" * 50)

    all_warnings  = []
    total_samples = 0

    for pid in test_patients:
        patient_dir = RAW_DIR / pid
        print(f"\nProcessing {pid}...")

        samples, warnings = process_single_patient(
            patient_dir,
            target_spacing = 1.5,
            target_size    = TARGET_SIZE,
            clip_low       = 0.5,
            clip_high      = 99.5,
        )

        saved = save_samples(samples, OUTPUT_DIR / "train")
        total_samples += saved
        all_warnings.extend(warnings)

        # Report per patient
        phases = set(s["phase"] for s in samples)
        slices = len(samples) // len(phases)
        print(f"  Group          : {samples[0]['group']}")
        print(f"  Phases         : {phases}")
        print(f"  Slices/phase   : {slices}")
        print(f"  Total saved    : {saved}")
        print(f"  Image shape    : {samples[0]['img'].shape}")
        print(f"  Image dtype    : {samples[0]['img'].dtype}")
        print(f"  Mask shape     : {samples[0]['msk'].shape}")
        print(f"  Mask dtype     : {samples[0]['msk'].dtype}")
        print(f"  Img mean~0     : "
              f"{samples[0]['img'][samples[0]['img']!=0].mean():.4f}")
        print(f"  Img std~1      : "
              f"{samples[0]['img'][samples[0]['img']!=0].std():.4f}")
        print(f"  Mask labels    : "
              f"{np.unique(samples[0]['msk']).tolist()}")

    # Verify saved files on disk
    print(f"\nVerifying saved files...")
    saved_files = list((OUTPUT_DIR / "train").glob("*.npy"))
    img_files   = [f for f in saved_files if f.name.endswith("_img.npy")]
    msk_files   = [f for f in saved_files if f.name.endswith("_msk.npy")]

    print(f"  Total .npy files : {len(saved_files)}")
    print(f"  Image files      : {len(img_files)}")
    print(f"  Mask files       : {len(msk_files)}")
    print(f"  Pairs matched    : {len(img_files) == len(msk_files)}")

    # Load one back and verify
    print(f"\nLoading one sample back from disk...")
    sample_img = np.load(img_files[0])
    sample_msk = np.load(msk_files[0])
    print(f"  File     : {img_files[0].name}")
    print(f"  Img shape: {sample_img.shape}")
    print(f"  Msk shape: {sample_msk.shape}")
    print(f"  Img dtype: {sample_img.dtype}")
    print(f"  Msk dtype: {sample_msk.dtype}")
    print(f"  Labels   : {np.unique(sample_msk).tolist()}")

    print(f"\nWarnings : {len(all_warnings)} "
          f"{'(none)' if not all_warnings else ''}")

    final_pass = (
        total_samples > 0 and
        len(img_files) == len(msk_files) and
        sample_img.shape == TARGET_SIZE
    )

    print("\n" + "=" * 50)
    print("pipeline.py — OK" if final_pass else "FAILED — check output")