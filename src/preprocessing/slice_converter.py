# src/preprocessing/slice_converter.py
#
# Responsibility: Load a 3D NIfTI volume and convert it into
# individual 2D slices as numpy arrays.
#
# Why this step exists:
#   Your dataset analysis showed slice thickness is ~10mm while
#   in-plane spacing is ~1.5mm — a 6-7x difference. This means
#   the Z dimension has no meaningful volumetric continuity.
#   Processing each 2D slice independently is the correct approach.
#
#   It also solves the variable depth problem — patients have
#   between 6 and 18 slices. By converting to 2D, you never
#   have to handle variable-length 3D volumes.

import numpy as np
import nibabel as nib
from pathlib import Path


def load_volume(nii_path: Path, is_mask: bool = False) -> tuple:
    """
    Load a NIfTI volume and return the data array plus header metadata.

    Args:
        nii_path : Path to the .nii or .nii.gz file
        is_mask  : if True, cast to uint8 to preserve integer labels

    Returns:
        volume  : 3D numpy array of shape (rows, cols, n_slices)
        spacing : float, in-plane pixel spacing in mm (from header)
        n_slices: int, number of axial slices
    """
    nii_path = Path(nii_path)

    if not nii_path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {nii_path}")

    img    = nib.load(str(nii_path))
    data   = img.get_fdata()

    if is_mask:
        data = data.astype(np.uint8)
    else:
        data = data.astype(np.float32)

    # Extract in-plane spacing from header
    # get_zooms() returns (spacing_x, spacing_y, spacing_z)
    # We use spacing_x — always equal to spacing_y in ACDC (square pixels)
    spacing  = float(img.header.get_zooms()[0])
    n_slices = data.shape[2]

    return data, spacing, n_slices


def volume_to_slices(nii_path: Path, is_mask: bool = False) -> tuple:
    """
    Convert a 3D NIfTI volume into a list of individual 2D slices.

    Each slice is one cross-section through the heart from base to apex.
    Slice 0 is the base, the last slice is near the apex.

    Args:
        nii_path : Path to the .nii or .nii.gz file
        is_mask  : if True, cast to uint8 to preserve integer labels

    Returns:
        slices  : list of 2D numpy arrays, each shape (rows, cols)
        spacing : float, in-plane pixel spacing in mm
        n_slices: int, total number of slices extracted
    """
    volume, spacing, n_slices = load_volume(nii_path, is_mask=is_mask)

    # Unpack the third dimension into individual 2D arrays
    # volume shape is (rows, cols, n_slices)
    # each slice is volume[:, :, i] → shape (rows, cols)
    slices = [volume[:, :, i] for i in range(n_slices)]

    return slices, spacing, n_slices


def extract_slice_pair(img_path: Path, msk_path: Path,
                        slice_idx: int) -> tuple:
    """
    Extract one specific 2D slice from both the image and its mask.
    Verifies that image and mask shapes match before returning.

    Args:
        img_path  : Path to the image NIfTI file
        msk_path  : Path to the mask NIfTI file
        slice_idx : which slice index to extract (0-based)

    Returns:
        img_slice : 2D float32 array (rows, cols)
        msk_slice : 2D uint8 array  (rows, cols)
        spacing   : float, in-plane pixel spacing in mm
    """
    img_vol, spacing, n_slices = load_volume(img_path, is_mask=False)
    msk_vol, _,       _        = load_volume(msk_path, is_mask=True)

    # Verify image and mask volumes match in shape
    if img_vol.shape != msk_vol.shape:
        raise ValueError(
            f"Image and mask shape mismatch: "
            f"{img_vol.shape} vs {msk_vol.shape} "
            f"in {img_path.name}"
        )

    # Verify slice index is valid
    if slice_idx >= n_slices:
        raise IndexError(
            f"slice_idx {slice_idx} out of range "
            f"for volume with {n_slices} slices."
        )

    img_slice = img_vol[:, :, slice_idx]
    msk_slice = msk_vol[:, :, slice_idx]

    return img_slice, msk_slice, spacing


# ── Quick test ────────────────────────────────────────────────────────────────
# Run directly to verify this file works on your data:
#   python src/preprocessing/slice_converter.py

if __name__ == "__main__":
    from src.preprocessing.extract_phases import get_phase_paths

    test_patient = Path("data/raw/training/patient001")
    print("Testing slice_converter.py on patient001...")
    print("=" * 50)

    # Get phase paths
    phase_info = get_phase_paths(test_patient)

    # ── Test 1: Load full volume and convert to slices ────────
    print("\n[Test 1] Converting ED volume to 2D slices...")
    img_slices, spacing, n_slices = volume_to_slices(
        phase_info["ed_img_path"], is_mask=False
    )
    msk_slices, _,       _        = volume_to_slices(
        phase_info["ed_msk_path"], is_mask=True
    )

    print(f"  In-plane spacing  : {spacing} mm/pixel")
    print(f"  Number of slices  : {n_slices}")
    print(f"  Each slice shape  : {img_slices[0].shape}")
    print(f"  Image dtype       : {img_slices[0].dtype}")
    print(f"  Mask dtype        : {msk_slices[0].dtype}")

    # ── Test 2: Check slice content across depth ──────────────
    print("\n[Test 2] Checking label content per slice...")
    print(f"  {'Slice':<8} {'Image min':>10} {'Image max':>10} "
          f"{'Mask labels':>20}")
    print(f"  {'-'*52}")

    for i in range(n_slices):
        labels = np.unique(msk_slices[i]).tolist()
        print(f"  {i:<8} {img_slices[i].min():>10.1f} "
              f"{img_slices[i].max():>10.1f} "
              f"{str(labels):>20}")

    # ── Test 3: Both phases produce slices ────────────────────
    print("\n[Test 3] Checking ES phase...")
    es_slices, _, es_n = volume_to_slices(
        phase_info["es_img_path"], is_mask=False
    )
    print(f"  ED slices : {n_slices}")
    print(f"  ES slices : {es_n}")

    # ── Test 4: extract_slice_pair on middle slice ────────────
    print("\n[Test 4] extract_slice_pair on middle slice...")
    mid = n_slices // 2
    img_s, msk_s, sp = extract_slice_pair(
        phase_info["ed_img_path"],
        phase_info["ed_msk_path"],
        slice_idx=mid
    )
    print(f"  Slice index       : {mid}")
    print(f"  Image shape       : {img_s.shape}")
    print(f"  Mask shape        : {msk_s.shape}")
    print(f"  Shapes match      : {img_s.shape == msk_s.shape}")
    print(f"  Mask labels       : {np.unique(msk_s).tolist()}")

    # ── Summary ───────────────────────────────────────────────
    all_passed = (
        len(img_slices) == n_slices and
        img_slices[0].shape == msk_slices[0].shape and
        img_s.shape == msk_s.shape
    )
    print("\n" + "=" * 50)
    print("slice_converter.py — OK" if all_passed else "FAILED — check output")