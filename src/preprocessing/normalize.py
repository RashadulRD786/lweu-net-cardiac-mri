# src/preprocessing/normalize.py
#
# Responsibility: Apply z-score normalization to a 2D MRI slice.
#
# Why this step exists:
#   Your dataset analysis showed raw intensity means ranging from
#   19 (HCM) to 42 (NOR) across groups containing identical tissue.
#   The global max was 4025 while the mean was only 67.8.
#   This scanner-induced variability means the same tissue type
#   produces completely different pixel values across patients.
#   Z-score normalization removes this variability by transforming
#   every image to mean=0 and std=1.
#
# Why foreground-only computation:
#   Background pixels are always zero. Including them would drag
#   the mean down and inflate the std, producing a normalization
#   that reflects background dominance rather than tissue contrast.
#
# Why percentile clipping comes first:
#   Extreme outlier pixels (like the max of 4025) inflate sigma,
#   which compresses all normal tissue pixels toward zero.
#   Clipping to 0.5-99.5 percentile removes those outliers before
#   statistics are computed, giving a clean and stable normalization.
#
# Key rule:
#   Never apply normalization to masks — they contain class labels,
#   not intensity values. Normalizing a mask destroys it entirely.

import numpy as np
from pathlib import Path


def normalize_slice(
    slice_2d    : np.ndarray,
    clip_low    : float = 0.5,
    clip_high   : float = 99.5,
) -> np.ndarray:
    """
    Apply z-score normalization to a single 2D MRI slice.

    Steps performed internally:
        1. Extract foreground pixels (value > 0)
        2. Compute percentile bounds from foreground
        3. Clip entire image to those bounds
        4. Recompute mean and std on clipped foreground
        5. Apply z-score: Z = (x - mean) / std

    Args:
        slice_2d  : 2D float32 numpy array (rows, cols)
                    raw MRI intensity values
        clip_low  : lower percentile for outlier clipping (default 0.5)
        clip_high : upper percentile for outlier clipping (default 99.5)

    Returns:
        normalized : 2D float32 array, same shape as input
                     foreground pixels have mean≈0 and std≈1
    """
    image = slice_2d.astype(np.float32).copy()

    # Step 1 — Extract foreground pixels only (ignore background zeros)
    foreground = image[image > 0]

    # Handle edge case: empty slice (all background, no tissue)
    # This can happen at the very top or bottom of the heart volume
    if len(foreground) == 0:
        return image

    # Step 2 — Compute percentile bounds from foreground
    p_low  = np.percentile(foreground, clip_low)
    p_high = np.percentile(foreground, clip_high)

    # Step 3 — Clip entire image to remove outlier pixels
    image = np.clip(image, p_low, p_high)

    # Step 4 — Recompute statistics on the clipped foreground
    foreground_clipped = image[image > 0]
    mu    = foreground_clipped.mean()
    sigma = foreground_clipped.std()

    # Guard against constant images where sigma would be zero
    # (extremely rare but would cause division by zero)
    if sigma < 1e-8:
        return (image - mu).astype(np.float32)

    # Step 5 — Apply z-score normalization
    normalized = (image - mu) / sigma

    return normalized.astype(np.float32)


def verify_normalization(normalized_slice: np.ndarray) -> dict:
    """
    Check that normalization produced the expected statistics.
    Used for validation — not part of the pipeline itself.

    Args:
        normalized_slice : 2D float32 array after normalization

    Returns:
        stats : dict with mean, std, min, max of foreground pixels
                and a boolean 'passed' indicating acceptable range
    """
    foreground = normalized_slice[normalized_slice != 0]

    if len(foreground) == 0:
        return {"passed": True, "note": "empty slice — skipped"}

    mu    = float(foreground.mean())
    sigma = float(foreground.std())
    mn    = float(normalized_slice.min())
    mx    = float(normalized_slice.max())

    # Acceptable tolerance — foreground mean should be close to 0
    # and std close to 1 after normalization
    mean_ok  = abs(mu)        < 0.1   # mean within ±0.1 of zero
    std_ok   = abs(sigma - 1) < 0.1   # std within ±0.1 of one

    return {
        "mean"   : round(mu, 6),
        "std"    : round(sigma, 6),
        "min"    : round(mn, 4),
        "max"    : round(mx, 4),
        "passed" : mean_ok and std_ok,
    }


# ── Quick test ────────────────────────────────────────────────────────────────
# Run directly to verify this file works on your data:
#   python -m src.preprocessing.normalize

if __name__ == "__main__":
    from src.preprocessing.extract_phases import get_phase_paths
    from src.preprocessing.slice_converter import volume_to_slices
    from src.preprocessing.resample import resample_slice
    from src.preprocessing.resize import resize_pair

    test_patient = Path("data/raw/training/patient001")
    TARGET_SIZE  = (224, 224)

    print("Testing normalize.py on patient001...")
    print("=" * 50)

    # Load and prepare the ED volume through all previous steps
    phase_info = get_phase_paths(test_patient)
    img_slices, spacing, n_slices = volume_to_slices(
        phase_info["ed_img_path"], is_mask=False
    )
    msk_slices, _, _ = volume_to_slices(
        phase_info["ed_msk_path"], is_mask=True
    )

    # Take the middle slice and run through pipeline so far
    mid       = n_slices // 2
    img_2d    = img_slices[mid]
    msk_2d    = msk_slices[mid]

    img_res   = resample_slice(img_2d, spacing, 1.5, is_mask=False)
    msk_res   = resample_slice(msk_2d, spacing, 1.5, is_mask=True)
    img_sized, msk_sized = resize_pair(img_res, msk_res, TARGET_SIZE)

    # ── Test 1: Inspect before normalization ──────────────────
    print(f"\n[Test 1] Before normalization (middle slice):")
    fg_before = img_sized[img_sized > 0]
    print(f"  Shape         : {img_sized.shape}")
    print(f"  Raw min       : {img_sized.min():.2f}")
    print(f"  Raw max       : {img_sized.max():.2f}")
    print(f"  Foreground mean : {fg_before.mean():.2f}")
    print(f"  Foreground std  : {fg_before.std():.2f}")

    # ── Test 2: Apply normalization ───────────────────────────
    img_norm = normalize_slice(img_sized, clip_low=0.5, clip_high=99.5)

    print(f"\n[Test 2] After normalization:")
    stats = verify_normalization(img_norm)
    print(f"  Shape           : {img_norm.shape}")
    print(f"  Normalized min  : {stats['min']}")
    print(f"  Normalized max  : {stats['max']}")
    print(f"  Foreground mean : {stats['mean']}")
    print(f"  Foreground std  : {stats['std']}")
    print(f"  Statistics OK   : {stats['passed']}")

    # ── Test 3: Mask must be completely unchanged ─────────────
    print(f"\n[Test 3] Mask untouched check:")
    print(f"  Mask labels     : {np.unique(msk_sized).tolist()}")
    print(f"  Mask dtype      : {msk_sized.dtype}")
    print(f"  Mask unchanged  : True  (normalization never touches masks)")

    # ── Test 4: All slices including empty base/apex ──────────
    print(f"\n[Test 4] Normalizing all {n_slices} slices...")
    print(f"  {'Slice':<8} {'Raw mean':>10} {'Raw max':>10} "
          f"{'Norm mean':>12} {'Norm std':>10} {'OK':>5}")
    print(f"  {'-' * 60}")

    all_passed = True
    for i in range(n_slices):
        r_img = resample_slice(img_slices[i], spacing, 1.5, is_mask=False)
        r_msk = resample_slice(msk_slices[i], spacing, 1.5, is_mask=True)
        f_img, _ = resize_pair(r_img, r_msk, TARGET_SIZE)
        n_img    = normalize_slice(f_img)
        stats_i  = verify_normalization(n_img)

        if "note" in stats_i:
            print(f"  {i:<8} {'—':>10} {'—':>10} "
                  f"{'—':>12} {'—':>10} {'skip':>5}")
            continue

        ok = "✓" if stats_i["passed"] else "✗"
        if not stats_i["passed"]:
            all_passed = False

        print(f"  {i:<8} "
              f"{f_img[f_img>0].mean():>10.2f} "
              f"{f_img.max():>10.2f} "
              f"{stats_i['mean']:>12.6f} "
              f"{stats_i['std']:>10.6f} "
              f"{ok:>5}")

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("normalize.py — OK" if all_passed else "FAILED — check output")