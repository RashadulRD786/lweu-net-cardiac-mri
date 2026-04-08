# src/preprocessing/verify_labels.py
#
# Responsibility: Verify and clean ground truth mask labels.
#
# Why this step exists:
#   Your pipeline has processed images through resampling, resizing,
#   and normalization. Before saving anything, you must confirm that
#   every mask still contains only valid ACDC class labels {0,1,2,3}.
#   Any corrupt value — a 4, a 255, a -1 from a loading or interpolation
#   error — would silently create a fifth class during one-hot encoding
#   in training, causing incorrect loss computation without any warning.
#
# What this step checks:
#   1. No labels outside {0, 1, 2, 3} exist in the mask
#   2. Image and mask shapes are identical (spatial alignment)
#   3. Mask dtype is uint8 (not float, not int32)
#   4. Class pixel counts are reported for imbalance awareness
#
# Key rule:
#   This step only inspects and cleans masks.
#   It never modifies image intensity values.

import numpy as np
from pathlib import Path


# Valid ACDC label set
VALID_LABELS = {0, 1, 2, 3}

# Human-readable names for reporting
LABEL_NAMES = {
    0: "Background",
    1: "RV         ",
    2: "MYO        ",
    3: "LV         ",
}


def verify_mask(
    mask_2d     : np.ndarray,
    patient_id  : str = "unknown",
    slice_idx   : int = 0,
    phase       : str = "ED",
) -> dict:
    """
    Verify a single 2D mask contains only valid ACDC labels.

    Args:
        mask_2d    : 2D numpy array (rows, cols) — the ground truth mask
        patient_id : patient identifier for error reporting
        slice_idx  : slice index for error reporting
        phase      : ED or ES for error reporting

    Returns:
        result : dict containing:
            is_clean      : bool — True if no corrupt labels found
            corrupt_vals  : set  — any values outside {0,1,2,3}
            labels_present: set  — which valid labels are in this mask
            pixel_counts  : dict — pixel count per label
            dtype_ok      : bool — True if dtype is uint8
    """
    mask = mask_2d.copy()

    # Check 1 — Find any values outside valid set
    unique_vals  = set(np.unique(mask).tolist())
    corrupt_vals = unique_vals - VALID_LABELS
    valid_present = unique_vals & VALID_LABELS

    # Check 2 — Dtype should be uint8
    dtype_ok = mask.dtype == np.uint8

    # Check 3 — Count pixels per label
    pixel_counts = {}
    for label in sorted(VALID_LABELS):
        pixel_counts[label] = int(np.sum(mask == label))

    is_clean = len(corrupt_vals) == 0 and dtype_ok

    if corrupt_vals:
        print(f"  ⚠ WARNING: {patient_id} | {phase} | slice {slice_idx} "
              f"— corrupt labels found: {corrupt_vals}")

    if not dtype_ok:
        print(f"  ⚠ WARNING: {patient_id} | {phase} | slice {slice_idx} "
              f"— dtype is {mask.dtype}, expected uint8")

    return {
        "is_clean"      : is_clean,
        "corrupt_vals"  : corrupt_vals,
        "labels_present": valid_present,
        "pixel_counts"  : pixel_counts,
        "dtype_ok"      : dtype_ok,
    }


def clean_mask(
    mask_2d    : np.ndarray,
    patient_id : str = "unknown",
    slice_idx  : int = 0,
) -> np.ndarray:
    """
    Remove corrupt labels by setting them to background (0).
    Always returns a uint8 array.

    This is a safety net — in a clean dataset like ACDC this
    should never need to do anything. But it protects against
    unexpected edge cases from interpolation or loading errors.

    Args:
        mask_2d    : 2D numpy array (rows, cols)
        patient_id : for reporting purposes
        slice_idx  : for reporting purposes

    Returns:
        cleaned : 2D uint8 array with only values in {0,1,2,3}
    """
    cleaned = mask_2d.copy().astype(np.uint8)

    unique_vals  = set(np.unique(cleaned).tolist())
    corrupt_vals = unique_vals - VALID_LABELS

    for bad_val in corrupt_vals:
        n_corrupt = int(np.sum(cleaned == bad_val))
        print(f"  Cleaning: {patient_id} slice {slice_idx} — "
              f"label {bad_val} ({n_corrupt} pixels) → set to 0")
        cleaned[cleaned == bad_val] = 0

    return cleaned


def verify_alignment(
    img_2d     : np.ndarray,
    mask_2d    : np.ndarray,
    patient_id : str = "unknown",
    slice_idx  : int = 0,
) -> bool:
    """
    Verify that image and mask have identical spatial dimensions.
    A shape mismatch means the mask is misaligned with the image —
    the model would learn from wrong labels.

    Args:
        img_2d     : 2D image array
        mask_2d    : 2D mask array
        patient_id : for reporting
        slice_idx  : for reporting

    Returns:
        aligned : bool — True if shapes match
    """
    aligned = img_2d.shape == mask_2d.shape

    if not aligned:
        print(f"  ⚠ ALIGNMENT ERROR: {patient_id} slice {slice_idx} "
              f"— image {img_2d.shape} vs mask {mask_2d.shape}")

    return aligned


def report_class_distribution(pixel_counts: dict,
                               total_pixels: int) -> None:
    """
    Print a visual class distribution report for one slice.
    Shows the severe class imbalance between background and structures.
    """
    for label in sorted(pixel_counts.keys()):
        count = pixel_counts[label]
        pct   = 100.0 * count / total_pixels if total_pixels > 0 else 0
        bar   = "█" * int(pct / 2)
        name  = LABEL_NAMES.get(label, f"Label {label}")
        print(f"    {name} ({label}): {pct:5.1f}%  {bar}")


# ── Quick test ────────────────────────────────────────────────────────────────
# Run directly to verify this file works on your data:
#   python -m src.preprocessing.verify_labels

if __name__ == "__main__":
    from src.preprocessing.extract_phases import get_phase_paths
    from src.preprocessing.slice_converter import volume_to_slices
    from src.preprocessing.resample import resample_slice
    from src.preprocessing.resize import resize_pair
    from src.preprocessing.normalize import normalize_slice

    test_patient = Path("data/raw/training/patient001")
    TARGET_SIZE  = (224, 224)

    print("Testing verify_labels.py on patient001...")
    print("=" * 50)

    # Load and run through full pipeline so far
    phase_info = get_phase_paths(test_patient)
    img_slices, spacing, n_slices = volume_to_slices(
        phase_info["ed_img_path"], is_mask=False
    )
    msk_slices, _, _ = volume_to_slices(
        phase_info["ed_msk_path"], is_mask=True
    )

    # ── Test 1: Verify middle slice ───────────────────────────
    mid       = n_slices // 2
    img_2d    = img_slices[mid]
    msk_2d    = msk_slices[mid]

    img_res   = resample_slice(img_2d, spacing, 1.5, is_mask=False)
    msk_res   = resample_slice(msk_2d, spacing, 1.5, is_mask=True)
    img_sized, msk_sized = resize_pair(img_res, msk_res, TARGET_SIZE)
    img_norm  = normalize_slice(img_sized)

    print(f"\n[Test 1] Verifying middle slice (index {mid})...")
    result = verify_mask(msk_sized, "patient001", mid, "ED")

    print(f"  Is clean        : {result['is_clean']}")
    print(f"  Corrupt values  : {result['corrupt_vals'] or 'None'}")
    print(f"  Labels present  : {result['labels_present']}")
    print(f"  Dtype OK        : {result['dtype_ok']}")

    # ── Test 2: Alignment check ───────────────────────────────
    print(f"\n[Test 2] Alignment check...")
    aligned = verify_alignment(img_norm, msk_sized, "patient001", mid)
    print(f"  Image shape     : {img_norm.shape}")
    print(f"  Mask shape      : {msk_sized.shape}")
    print(f"  Aligned         : {aligned}")

    # ── Test 3: Class distribution ────────────────────────────
    print(f"\n[Test 3] Class distribution (middle slice)...")
    total = msk_sized.size
    report_class_distribution(result["pixel_counts"], total)

    # ── Test 4: All slices ────────────────────────────────────
    print(f"\n[Test 4] Verifying all {n_slices} slices...")
    all_clean   = True
    all_aligned = True

    for i in range(n_slices):
        r_img = resample_slice(img_slices[i], spacing, 1.5, is_mask=False)
        r_msk = resample_slice(msk_slices[i], spacing, 1.5, is_mask=True)
        f_img, f_msk = resize_pair(r_img, r_msk, TARGET_SIZE)
        n_img = normalize_slice(f_img)

        res = verify_mask(f_msk, "patient001", i, "ED")
        aln = verify_alignment(n_img, f_msk, "patient001", i)

        if not res["is_clean"]:
            all_clean = False
        if not aln:
            all_aligned = False

    print(f"  All masks clean    : {all_clean}")
    print(f"  All masks aligned  : {all_aligned}")

    # ── Test 5: Simulate corrupt label ────────────────────────
    print(f"\n[Test 5] Corrupt label simulation...")
    corrupt_mask = msk_sized.copy()
    corrupt_mask[10:15, 10:15] = 99   # inject fake corrupt label
    print(f"  Injected label 99 into mask")

    res_corrupt = verify_mask(corrupt_mask, "patient001", mid, "ED")
    cleaned     = clean_mask(corrupt_mask, "patient001", mid)

    print(f"  Corrupt detected   : {not res_corrupt['is_clean']}")
    print(f"  After cleaning     : {set(np.unique(cleaned).tolist())}")
    print(f"  Cleaning worked    : "
          f"{set(np.unique(cleaned).tolist()).issubset(VALID_LABELS)}")

    # ── Summary ───────────────────────────────────────────────
    final_pass = all_clean and all_aligned
    print("\n" + "=" * 50)
    print("verify_labels.py — OK" if final_pass
          else "FAILED — check output")