# src/preprocessing/generate_split.py
#
# Responsibility: Generate patient-level train / val / test splits
# and save them as CSV files for reproducibility.
#
# Why this step exists:
#   You have 150 patients — 100 official training, 50 official testing.
#   Within the 100 training patients, you need to reserve 20 for
#   validation. This split must be:
#
#   1. PATIENT-LEVEL — not slice-level. If slices from the same patient
#      appear in both train and val, the model has already seen that
#      heart and validation accuracy becomes artificially inflated.
#      This is called data leakage and invalidates your results.
#
#   2. STRATIFIED — each pathology group (NOR, DCM, HCM, MINF, ARV)
#      must be proportionally represented in both train and val.
#      Without stratification, you might accidentally put all HCM
#      patients in training and never evaluate on that condition.
#
#   3. REPRODUCIBLE — the same random seed must always produce the
#      same split. This is essential for thesis reproducibility —
#      your supervisor or examiner must be able to recreate your
#      exact results.
#
# Output:
#   data/splits/train_patients.csv  — 80 patients for training
#   data/splits/val_patients.csv    — 20 patients for validation
#   data/splits/test_patients.csv   — 50 patients for testing
#   data/splits/split_summary.txt   — human-readable split report

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_patient_inventory(splits_dir: Path) -> pd.DataFrame:
    """
    Load the dataset inventory CSV built during Phase 1 exploration.
    This contains patient_id, group, and metadata for all 100 patients.

    Args:
        splits_dir : Path to data/splits/

    Returns:
        df : DataFrame with one row per patient
    """
    inventory_path = splits_dir / "dataset_inventory.csv"

    if not inventory_path.exists():
        raise FileNotFoundError(
            f"dataset_inventory.csv not found at {inventory_path}.\n"
            f"Run notebook 01_dataset_exploration.ipynb first to "
            f"generate this file."
        )

    df = pd.read_csv(inventory_path)
    print(f"  Loaded inventory: {len(df)} patients")
    return df


def generate_splits(
    df           : pd.DataFrame,
    val_fraction : float = 0.2,
    random_seed  : int   = 42,
) -> tuple:
    """
    Split 100 training patients into train and val sets.
    Uses stratified splitting to maintain group balance.

    Args:
        df           : DataFrame with patient_id and group columns
        val_fraction : fraction of training patients for validation
        random_seed  : random seed for reproducibility

    Returns:
        train_df : DataFrame of training patients
        val_df   : DataFrame of validation patients
    """
    patient_ids = df["patient_id"].tolist()
    groups      = df["group"].tolist()

    train_ids, val_ids = train_test_split(
        patient_ids,
        test_size   = val_fraction,
        random_state= random_seed,
        stratify    = groups,
    )

    train_df = df[df["patient_id"].isin(train_ids)].reset_index(drop=True)
    val_df   = df[df["patient_id"].isin(val_ids)].reset_index(drop=True)

    return train_df, val_df


def build_test_inventory(raw_test_dir: Path) -> pd.DataFrame:
    """
    Build a simple inventory for the 50 official test patients.
    Reads group labels from their Info.cfg files.

    Args:
        raw_test_dir : Path to data/raw/testing/

    Returns:
        df : DataFrame with patient_id and group for test patients
    """
    records = []

    for patient_dir in sorted(raw_test_dir.iterdir()):
        if not patient_dir.is_dir():
            continue

        cfg_path = patient_dir / "Info.cfg"
        if not cfg_path.exists():
            continue

        info = {}
        with open(cfg_path) as f:
            for line in f:
                if ":" in line:
                    k, v = line.strip().split(":", 1)
                    info[k.strip()] = v.strip()

        group = info.get("Group", "RV")
        if group == "RV":
            group = "ARV"

        records.append({
            "patient_id": patient_dir.name,
            "group"     : group,
            "split"     : "test",
        })

    return pd.DataFrame(records)


def save_splits(
    train_df   : pd.DataFrame,
    val_df     : pd.DataFrame,
    test_df    : pd.DataFrame,
    splits_dir : Path,
) -> None:
    """
    Save all three splits as CSV files and write a summary report.

    Args:
        train_df   : training patients DataFrame
        val_df     : validation patients DataFrame
        test_df    : test patients DataFrame
        splits_dir : output directory (data/splits/)
    """
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Add split label column
    train_df = train_df.copy()
    val_df   = val_df.copy()

    train_df["split"] = "train"
    val_df["split"]   = "val"

    # Save CSVs
    train_df[["patient_id", "group", "split"]].to_csv(
        splits_dir / "train_patients.csv", index=False
    )
    val_df[["patient_id", "group", "split"]].to_csv(
        splits_dir / "val_patients.csv", index=False
    )
    test_df[["patient_id", "group", "split"]].to_csv(
        splits_dir / "test_patients.csv", index=False
    )

    # Write human-readable summary
    summary_lines = []
    summary_lines.append("DATASET SPLIT SUMMARY")
    summary_lines.append("=" * 50)
    summary_lines.append(f"Total patients : "
                         f"{len(train_df) + len(val_df) + len(test_df)}")
    summary_lines.append(f"  Train        : {len(train_df)}")
    summary_lines.append(f"  Val          : {len(val_df)}")
    summary_lines.append(f"  Test         : {len(test_df)}")
    summary_lines.append("")
    summary_lines.append("Group distribution:")
    summary_lines.append(
        f"  {'Group':<8} {'Train':>6} {'Val':>6} {'Test':>6}"
    )
    summary_lines.append(f"  {'-'*30}")

    for group in sorted(train_df["group"].unique()):
        t = int((train_df["group"] == group).sum())
        v = int((val_df["group"]   == group).sum())
        e = int((test_df["group"]  == group).sum())
        summary_lines.append(f"  {group:<8} {t:>6} {v:>6} {e:>6}")

    summary_lines.append("")
    summary_lines.append("Estimated slice counts (×2 phases ×~9.5 slices):")
    summary_lines.append(f"  Train slices : ~{len(train_df) * 2 * 10}")
    summary_lines.append(f"  Val slices   : ~{len(val_df)   * 2 * 10}")
    summary_lines.append(f"  Test slices  : ~{len(test_df)  * 2 * 10}")

    summary_text = "\n".join(summary_lines)

    with open(splits_dir / "split_summary.txt", "w") as f:
        f.write(summary_text)

    print(summary_text)


# ── Quick test ────────────────────────────────────────────────────────────────
# Run directly to verify this file works on your data:
#   python -m src.preprocessing.generate_split

if __name__ == "__main__":
    import yaml

    # Load config
    with open("configs/preprocessing_config.yaml") as f:
        cfg = yaml.safe_load(f)

    SPLITS_DIR   = Path(cfg["paths"]["splits_dir"])
    RAW_TEST_DIR = Path(cfg["paths"]["raw_test_dir"])
    VAL_FRACTION = cfg["split"]["val_fraction"]
    RANDOM_SEED  = cfg["split"]["random_seed"]

    print("Testing generate_split.py...")
    print("=" * 50)

    # ── Step 1: Load inventory ────────────────────────────────
    print("\n[Step 1] Loading patient inventory...")
    df = load_patient_inventory(SPLITS_DIR)

    print(f"  Group distribution in full training set:")
    for group, count in df["group"].value_counts().sort_index().items():
        print(f"    {group}: {count} patients")

    # ── Step 2: Generate train/val split ──────────────────────
    print(f"\n[Step 2] Generating stratified split "
          f"(val={VAL_FRACTION}, seed={RANDOM_SEED})...")
    train_df, val_df = generate_splits(df, VAL_FRACTION, RANDOM_SEED)

    print(f"  Train patients : {len(train_df)}")
    print(f"  Val patients   : {len(val_df)}")

    # ── Step 3: Verify stratification ────────────────────────
    print(f"\n[Step 3] Verifying stratification...")
    print(f"  {'Group':<8} {'Train':>6} {'Val':>6} {'Ratio':>8}")
    print(f"  {'-'*32}")

    stratification_ok = True
    for group in sorted(df["group"].unique()):
        t = int((train_df["group"] == group).sum())
        v = int((val_df["group"]   == group).sum())
        ratio = v / (t + v) if (t + v) > 0 else 0
        ok = "✓" if abs(ratio - VAL_FRACTION) < 0.05 else "✗"
        print(f"  {group:<8} {t:>6} {v:>6} {ratio:>7.0%}  {ok}")
        if ok == "✗":
            stratification_ok = False

    print(f"\n  Stratification correct: {stratification_ok}")

    # ── Step 4: Build test inventory ─────────────────────────
    print(f"\n[Step 4] Building test patient inventory...")
    test_df = build_test_inventory(RAW_TEST_DIR)
    print(f"  Test patients found: {len(test_df)}")
    for group, count in test_df["group"].value_counts().sort_index().items():
        print(f"    {group}: {count} patients")

    # ── Step 5: Verify no data leakage ───────────────────────
    print(f"\n[Step 5] Checking for data leakage...")
    train_set = set(train_df["patient_id"].tolist())
    val_set   = set(val_df["patient_id"].tolist())
    test_set  = set(test_df["patient_id"].tolist())

    train_val_overlap  = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap   = val_set   & test_set

    print(f"  Train ∩ Val  : {len(train_val_overlap)}  "
          f"{'✓ No leakage' if not train_val_overlap else '✗ LEAKAGE'}")
    print(f"  Train ∩ Test : {len(train_test_overlap)}  "
          f"{'✓ No leakage' if not train_test_overlap else '✗ LEAKAGE'}")
    print(f"  Val ∩ Test   : {len(val_test_overlap)}  "
          f"{'✓ No leakage' if not val_test_overlap else '✗ LEAKAGE'}")

    no_leakage = (not train_val_overlap and
                  not train_test_overlap and
                  not val_test_overlap)

    # ── Step 6: Save splits ───────────────────────────────────
    print(f"\n[Step 6] Saving split files to {SPLITS_DIR}...")
    save_splits(train_df, val_df, test_df, SPLITS_DIR)

    # ── Summary ───────────────────────────────────────────────
    final_pass = stratification_ok and no_leakage
    print("\n" + "=" * 50)
    print("generate_split.py — OK" if final_pass
          else "FAILED — check output")