# src/preprocessing/resize.py
#
# Responsibility: Resize a 2D slice to a fixed spatial dimension (224×224).
#
# Why this step exists:
#   After resampling, images are still different sizes because patients
#   have different physical heart sizes. For example:
#     patient001 after resampling → (225, 267)
#     patient with 1.920mm spacing → (165, 196)
#   Neural networks require a fixed input shape for batching.
#   This step eliminates that remaining size variability.
#
# Why this comes AFTER resampling and BEFORE normalization:
#   Resampling corrects physical scale (mm/pixel) first.
#   Resizing then corrects pixel dimensions to a fixed grid.
#   Normalization last operates on the final pixel values.
#   Doing resize before resample would undo the scale correction.
#
# Key rule:
#   Images → cv2.INTER_LINEAR    (bilinear, smooth for MRI intensities)
#   Masks  → cv2.INTER_NEAREST   (nearest-neighbour, preserves integer labels)

import numpy as np
import cv2
from pathlib import Path


def resize_slice(
    slice_2d    : np.ndarray,
    target_size : tuple = (224, 224),
    is_mask     : bool  = False,
) -> np.ndarray:
    """
    Resize a single 2D slice to target_size.

    Args:
        slice_2d    : 2D numpy array (rows, cols)
        target_size : (height, width) in pixels — default (224, 224)
        is_mask     : if True, use nearest-neighbour interpolation
                      to preserve discrete integer class labels

    Returns:
        resized : 2D numpy array of shape target_size
    """
    # Choose interpolation method
    # INTER_LINEAR   → bilinear, creates smooth gradients (good for images)
    # INTER_NEAREST  → snaps to nearest pixel value (essential for masks)
    if is_mask:
        interpolation = cv2.INTER_NEAREST
    else:
        interpolation = cv2.INTER_LINEAR

    # cv2.resize expects (width, height) — opposite of numpy (rows, cols)
    target_wh = (target_size[1], target_size[0])

    resized = cv2.resize(
        slice_2d.astype(np.float32),
        target_wh,
        interpolation=interpolation
    )

    # Restore correct dtype after cv2 processing
    if is_mask:
        resized = resized.astype(np.uint8)
    else:
        resized = resized.astype(np.float32)

    return resized


def resize_pair(
    img_slice   : np.ndarray,
    msk_slice   : np.ndarray,
    target_size : tuple = (224, 224),
) -> tuple:
    """
    Resize an image slice and its corresponding mask together.
    Ensures both receive the correct interpolation method.

    Args:
        img_slice   : 2D float32 array (rows, cols)
        msk_slice   : 2D uint8 array  (rows, cols)
        target_size : (height, width) in pixels

    Returns:
        resized_img : 2D float32 array of shape target_size
        resized_msk : 2D uint8 array  of shape target_size
    """
    resized_img = resize_slice(img_slice, target_size, is_mask=False)
    resized_msk = resize_slice(msk_slice, target_size, is_mask=True)
    return resized_img, resized_msk


# ── Quick test ────────────────────────────────────────────────────────────────
# Run directly to verify this file works on your data:
#   python -m src.preprocessing.resize

if __name__ == "__main__":
    import nibabel as nib
    from src.preprocessing.extract_phases import get_phase_paths
    from src.preprocessing.slice_converter import volume_to_slices
    from src.preprocessing.resample import resample_slice

    test_patient = Path("data/raw/training/patient001")
    TARGET_SIZE  = (224, 224)

    print("Testing resize.py on patient001...")
    print("=" * 50)

    # Load and slice the ED volume
    phase_info = get_phase_paths(test_patient)
    img_slices, spacing, n_slices = volume_to_slices(
        phase_info["ed_img_path"], is_mask=False
    )
    msk_slices, _, _ = volume_to_slices(
        phase_info["ed_msk_path"], is_mask=True
    )

    # Take the middle slice
    mid = n_slices // 2
    img_2d = img_slices[mid]
    msk_2d = msk_slices[mid]

    # ── Step 1: Resample first (as pipeline dictates) ─────────
    img_resampled = resample_slice(img_2d, spacing,
                                   target_spacing=1.5, is_mask=False)
    msk_resampled = resample_slice(msk_2d, spacing,
                                   target_spacing=1.5, is_mask=True)

    print(f"After resampling:")
    print(f"  Image shape     : {img_resampled.shape}")
    print(f"  Mask shape      : {msk_resampled.shape}")

    # ── Step 2: Resize to fixed dimensions ────────────────────
    img_resized, msk_resized = resize_pair(
        img_resampled, msk_resampled, target_size=TARGET_SIZE
    )

    print(f"\nAfter resizing to {TARGET_SIZE}:")
    print(f"  Image shape     : {img_resized.shape}")
    print(f"  Mask shape      : {msk_resized.shape}")
    print(f"  Image dtype     : {img_resized.dtype}")
    print(f"  Mask dtype      : {msk_resized.dtype}")
    print(f"  Image min/max   : {img_resized.min():.2f} / "
          f"{img_resized.max():.2f}")

    # ── Label integrity check ─────────────────────────────────
    labels_before = set(np.unique(msk_resampled).tolist())
    labels_after  = set(np.unique(msk_resized).tolist())
    no_corruption = labels_after.issubset({0, 1, 2, 3})
    no_new_labels = labels_after == labels_before

    print(f"\nLabel integrity check:")
    print(f"  Labels before resize : {labels_before}")
    print(f"  Labels after resize  : {labels_after}")
    print(f"  No corruption        : {no_corruption}")
    print(f"  No new labels        : {no_new_labels}")

    # ── Shape confirmation ────────────────────────────────────
    correct_shape  = img_resized.shape == TARGET_SIZE
    shapes_match   = img_resized.shape == msk_resized.shape

    print(f"\nShape checks:")
    print(f"  Target size reached  : {correct_shape}  "
          f"({img_resized.shape})")
    print(f"  Image/mask match     : {shapes_match}")

    # ── Test all slices, not just middle ──────────────────────
    print(f"\nResizing all {n_slices} slices...")
    all_passed = True
    for i in range(n_slices):
        r_img = resample_slice(img_slices[i], spacing, 1.5, is_mask=False)
        r_msk = resample_slice(msk_slices[i], spacing, 1.5, is_mask=True)
        f_img, f_msk = resize_pair(r_img, r_msk, TARGET_SIZE)

        if f_img.shape != TARGET_SIZE or f_msk.shape != TARGET_SIZE:
            print(f"  Slice {i}: FAILED — shape {f_img.shape}")
            all_passed = False
        if not set(np.unique(f_msk).tolist()).issubset({0, 1, 2, 3}):
            print(f"  Slice {i}: FAILED — corrupt labels")
            all_passed = False

    if all_passed:
        print(f"  All {n_slices} slices → {TARGET_SIZE} ✓")

    # ── Summary ───────────────────────────────────────────────
    final_pass = correct_shape and shapes_match and no_corruption
    print("\n" + "=" * 50)
    print("resize.py — OK" if final_pass else "FAILED — check output")