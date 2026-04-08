# src/preprocessing/resample.py
#
# Responsibility: Resample a 2D slice from its original pixel spacing
# to a uniform target spacing (1.5mm isotropic).
#
# Why this step exists:
#   Your dataset analysis showed in-plane spacing ranges from 0.703 to
#   1.920mm across patients — almost a 3x difference. Without resampling,
#   the same cardiac structure appears at different pixel sizes depending
#   on which scanner was used. This step corrects for that.
#
# Key rule:
#   Images  → bilinear interpolation  (smooth, acceptable for MRI)
#   Masks   → nearest-neighbour       (preserves discrete class labels)

import numpy as np
import SimpleITK as sitk


def resample_slice(
    slice_2d       : np.ndarray,
    original_spacing : float,
    target_spacing   : float = 1.5,
    is_mask          : bool  = False,
) -> np.ndarray:
    """
    Resample a single 2D slice to target_spacing (mm/pixel).

    The physical coverage of the image is preserved — only the zoom
    level changes. A slice covering 300mm at 1.5625mm/pixel will still
    cover 300mm after resampling to 1.5mm/pixel, just with slightly
    more pixels.

    Args:
        slice_2d         : 2D numpy array (rows, cols)
        original_spacing : current mm/pixel value from NIfTI header
        target_spacing   : desired mm/pixel (default 1.5)
        is_mask          : if True, use nearest-neighbour interpolation
                           to preserve integer class labels

    Returns:
        resampled : 2D numpy array at target_spacing
    """
    # SimpleITK expects float32
    sitk_img = sitk.GetImageFromArray(slice_2d.astype(np.float32))
    sitk_img.SetSpacing([float(original_spacing), float(original_spacing)])

    # Compute new pixel dimensions to maintain physical coverage
    # new_size = original_size × (original_spacing / target_spacing)
    original_size = np.array(sitk_img.GetSize())   # (cols, rows) in sitk
    scale_factor  = original_spacing / target_spacing
    new_size      = np.round(original_size * scale_factor).astype(int).tolist()

    # Configure resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing([target_spacing, target_spacing])
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(sitk_img.GetDirection())
    resampler.SetOutputOrigin(sitk_img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)

    # CRITICAL: different interpolation for images vs masks
    if is_mask:
        # Nearest-neighbour — never creates non-integer label values
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        # Bilinear — smooth interpolation acceptable for MRI intensities
        resampler.SetInterpolator(sitk.sitkLinear)

    resampled = resampler.Execute(sitk_img)
    result    = sitk.GetArrayFromImage(resampled)

    # Preserve dtype — masks must stay integer
    if is_mask:
        result = result.astype(np.uint8)
    else:
        result = result.astype(np.float32)

    return result


def get_resampled_size(original_shape, original_spacing, target_spacing=1.5):
    """
    Calculate the output shape after resampling without doing the resampling.
    Useful for validation and debugging.

    Args:
        original_shape   : tuple (rows, cols)
        original_spacing : float, current mm/pixel
        target_spacing   : float, desired mm/pixel

    Returns:
        new_shape : tuple (new_rows, new_cols)
    """
    scale    = original_spacing / target_spacing
    new_rows = int(round(original_shape[0] * scale))
    new_cols = int(round(original_shape[1] * scale))
    return (new_rows, new_cols)


# ── Quick test ────────────────────────────────────────────────────────────────
# Run directly to verify this file works on your data:
#   python src/preprocessing/resample.py

if __name__ == "__main__":
    import nibabel as nib
    from pathlib import Path
    from src.preprocessing.extract_phases import get_phase_paths

    test_patient = Path("data/raw/training/patient001")
    print("Testing resample.py on patient001...")
    print("=" * 50)

    # Load patient info and ED volume
    phase_info   = get_phase_paths(test_patient)
    ed_img_nib   = nib.load(str(phase_info["ed_img_path"]))
    ed_msk_nib   = nib.load(str(phase_info["ed_msk_path"]))

    spacing      = float(ed_img_nib.header.get_zooms()[0])
    ed_img_vol   = ed_img_nib.get_fdata().astype(np.float32)
    ed_msk_vol   = ed_msk_nib.get_fdata().astype(np.uint8)

    # Take the middle slice for testing
    mid_slice    = ed_img_vol.shape[2] // 2
    img_2d       = ed_img_vol[:, :, mid_slice]
    msk_2d       = ed_msk_vol[:, :, mid_slice]

    print(f"Original spacing      : {spacing} mm/pixel")
    print(f"Original image shape  : {img_2d.shape}")
    print(f"Original mask labels  : {np.unique(msk_2d)}")

    # Resample both image and mask
    img_resampled = resample_slice(img_2d, spacing, target_spacing=1.5,
                                   is_mask=False)
    msk_resampled = resample_slice(msk_2d, spacing, target_spacing=1.5,
                                   is_mask=True)

    print(f"\nResampled image shape : {img_resampled.shape}")
    print(f"Resampled mask shape  : {msk_resampled.shape}")
    print(f"Resampled mask labels : {np.unique(msk_resampled)}")

    # Verify physical coverage is preserved
    orig_phys_w  = img_2d.shape[1] * spacing
    new_phys_w   = img_resampled.shape[1] * 1.5
    orig_phys_h  = img_2d.shape[0] * spacing
    new_phys_h   = img_resampled.shape[0] * 1.5

    print(f"\nPhysical coverage check:")
    print(f"  Width  before: {orig_phys_w:.1f}mm  "
          f"after: {new_phys_w:.1f}mm  "
          f"diff: {abs(orig_phys_w - new_phys_w):.2f}mm")
    print(f"  Height before: {orig_phys_h:.1f}mm  "
          f"after: {new_phys_h:.1f}mm  "
          f"diff: {abs(orig_phys_h - new_phys_h):.2f}mm")

    # Verify no new labels were created in mask
    original_labels = set(np.unique(msk_2d).tolist())
    resampled_labels = set(np.unique(msk_resampled).tolist())
    labels_clean = resampled_labels.issubset({0, 1, 2, 3})

    print(f"\nLabel integrity check:")
    print(f"  Labels before : {original_labels}")
    print(f"  Labels after  : {resampled_labels}")
    print(f"  No corruption : {labels_clean}")

    print("=" * 50)
    print("resample.py — OK" if labels_clean else "FAILED — label corruption")