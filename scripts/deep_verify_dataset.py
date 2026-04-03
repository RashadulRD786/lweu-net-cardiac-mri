import numpy as np
import nibabel as nib
from pathlib import Path
from collections import defaultdict

RAW_DIR = Path("data/raw")

# ── Expected ACDC structure constants ────────────────────────────────────────
VALID_GROUPS     = {"NOR", "DCM", "HCM", "MINF", "RV"}
VALID_LABELS     = {0, 1, 2, 3}
EXPECTED_TRAIN   = 100
EXPECTED_TEST    = 50


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_nii(path: Path) -> Path | None:
    """Returns .nii.gz if exists, else .nii, else None."""
    if (gz := path.with_suffix(".nii.gz")).exists():
        return gz
    if (nii := path.with_suffix(".nii")).exists():
        return nii
    return None


def read_info_cfg(patient_dir: Path) -> dict | None:
    cfg_path = patient_dir / "Info.cfg"
    if not cfg_path.exists():
        return None
    info = {}
    with open(cfg_path) as f:
        for line in f:
            if ":" in line:
                k, v = line.strip().split(":", 1)
                info[k.strip()] = v.strip()
    return info


def try_load_nifti(filepath: Path) -> tuple[bool, str, object]:
    """
    Attempts to load a NIfTI file.
    Returns (success, error_message, nib_object).
    """
    try:
        img = nib.load(str(filepath))
        _ = img.get_fdata()   # Force actual data load
        return True, "", img
    except Exception as e:
        return False, str(e), None


# ── Per-patient checks ────────────────────────────────────────────────────────

def verify_patient(patient_dir: Path, split: str) -> dict:
    """
    Runs all checks on a single patient.
    Returns a dict of issues found (empty = clean).
    """
    pid    = patient_dir.name
    issues = []
    info   = {}

    # ── Check 1: Info.cfg exists and is parseable ─────────────────────────
    raw_info = read_info_cfg(patient_dir)
    if raw_info is None:
        issues.append("MISSING Info.cfg")
        return {"pid": pid, "issues": issues, "info": {}}

    # ── Check 2: ED and ES keys present in Info.cfg ───────────────────────
    for key in ["ED", "ES", "Group"]:
        if key not in raw_info:
            issues.append(f"Info.cfg missing key: {key}")

    ed_frame = raw_info.get("ED")
    es_frame = raw_info.get("ES")
    group    = raw_info.get("Group", "UNKNOWN")

    # ── Check 3: Group is a valid ACDC pathology ──────────────────────────
    if group not in VALID_GROUPS:
        issues.append(f"Unknown pathology group: '{group}'")

    # ── Check 4: ED != ES (they must be different frames) ─────────────────
    if ed_frame and es_frame and ed_frame == es_frame:
        issues.append(f"ED and ES point to same frame: {ed_frame}")

    # ── Check 5: All 4 NIfTI files exist and are readable ─────────────────
    for phase, frame_id in [("ED", ed_frame), ("ES", es_frame)]:
        if frame_id is None:
            continue

        base      = patient_dir / f"{pid}_frame{int(frame_id):02d}"
        img_path  = find_nii(base)
        mask_base = patient_dir / f"{pid}_frame{int(frame_id):02d}_gt"
        msk_path  = find_nii(mask_base)

        # File existence
        if img_path is None:
            issues.append(f"MISSING image: {base.name}.nii[.gz]")
        if msk_path is None:
            issues.append(f"MISSING mask:  {mask_base.name}.nii[.gz]")

        # File readability (actually open and decode)
        if img_path:
            ok, err, img_obj = try_load_nifti(img_path)
            if not ok:
                issues.append(f"CORRUPT image ({phase}): {err}")
            else:
                img_data = img_obj.get_fdata()

                # ── Check 6: Image is 3D ──────────────────────────────────
                if img_data.ndim != 3:
                    issues.append(
                        f"Image not 3D ({phase}): shape={img_data.shape}"
                    )

                # ── Check 7: No NaN or Inf in image ──────────────────────
                if np.any(np.isnan(img_data)):
                    issues.append(f"NaN values in image ({phase})")
                if np.any(np.isinf(img_data)):
                    issues.append(f"Inf values in image ({phase})")

                # ── Check 8: Image is not all zeros ──────────────────────
                if img_data.max() == 0:
                    issues.append(f"Image is all zeros ({phase})")

                # ── Check 9: Pixel spacing is present ────────────────────
                pixdim = img_obj.header.get_zooms()
                if len(pixdim) < 2 or pixdim[0] == 0 or pixdim[1] == 0:
                    issues.append(f"Invalid pixel spacing ({phase}): {pixdim}")

                info[f"{phase}_shape"]   = img_data.shape
                info[f"{phase}_spacing"] = tuple(round(float(p), 3)
                                                  for p in pixdim)

        if msk_path:
            ok, err, msk_obj = try_load_nifti(msk_path)
            if not ok:
                issues.append(f"CORRUPT mask ({phase}): {err}")
            else:
                msk_data = msk_obj.get_fdata().astype(np.uint8)

                # ── Check 10: Mask label values are valid ─────────────────
                unique_labels = set(np.unique(msk_data).tolist())
                invalid       = unique_labels - VALID_LABELS
                if invalid:
                    issues.append(
                        f"Invalid mask labels ({phase}): {invalid}"
                    )

                # ── Check 11: Mask has at least one non-background label ──
                if msk_data.max() == 0:
                    issues.append(f"Mask is all background ({phase})")

                # ── Check 12: Image and mask shape match ──────────────────
                if img_path and ok:
                    img_shape = img_obj.get_fdata().shape
                    if img_shape != msk_data.shape:
                        issues.append(
                            f"Shape mismatch ({phase}): "
                            f"image={img_shape} mask={msk_data.shape}"
                        )

                info[f"{phase}_labels"] = sorted(unique_labels)

    info["group"] = group
    return {"pid": pid, "issues": issues, "info": info}


# ── Dataset-level summary ─────────────────────────────────────────────────────

def verify_dataset() -> None:
    print("\n" + "=" * 60)
    print("   DEEP DATASET VERIFICATION — ACDC")
    print("=" * 60)

    grand_total_issues = 0

    for split, expected_count in [
        ("training", EXPECTED_TRAIN),
        ("testing",  EXPECTED_TEST)
    ]:
        split_dir = RAW_DIR / split
        if not split_dir.exists():
            print(f"\n[ERROR] Folder not found: {split_dir}")
            continue

        patients = sorted([p for p in split_dir.iterdir() if p.is_dir()])

        print(f"\n {'─'*56}")
        print(f"  {split.upper()}  ({len(patients)} patients found, "
              f"expected {expected_count})")
        print(f" {'─'*56}")

        # ── Check patient count ───────────────────────────────────────────
        if len(patients) != expected_count:
            print(f"  [WARNING] Expected {expected_count} patients, "
                  f"found {len(patients)}")

        group_counts  = defaultdict(int)
        spacing_stats = []
        shape_stats   = []
        patients_with_issues = []

        for patient_dir in patients:
            result = verify_patient(patient_dir, split)
            pid    = result["pid"]
            issues = result["issues"]
            info   = result["info"]

            if issues:
                patients_with_issues.append((pid, issues))
                grand_total_issues += len(issues)
            else:
                # Collect stats from clean patients
                if "group" in info:
                    group_counts[info["group"]] += 1
                for phase in ["ED", "ES"]:
                    if f"{phase}_spacing" in info:
                        spacing_stats.append(info[f"{phase}_spacing"])
                    if f"{phase}_shape" in info:
                        shape_stats.append(info[f"{phase}_shape"])

        # ── Report issues ─────────────────────────────────────────────────
        if patients_with_issues:
            print(f"\n  [ISSUES FOUND — {len(patients_with_issues)} patients]\n")
            for pid, issues in patients_with_issues:
                print(f"   {pid}:")
                for issue in issues:
                    print(f"      ✗  {issue}")
        else:
            print(f"\n  All {len(patients)} patients passed all checks.\n")

        # ── Pathology group distribution ──────────────────────────────────
        print(f"  Pathology group distribution:")
        for grp in sorted(VALID_GROUPS):
            count = group_counts.get(grp, 0)
            bar   = "█" * count
            print(f"    {grp:6s}: {count:3d}  {bar}")

        # ── Pixel spacing summary ─────────────────────────────────────────
        if spacing_stats:
            in_plane = [s[0] for s in spacing_stats if len(s) >= 2]
            print(f"\n  Pixel spacing (in-plane):")
            print(f"    Min : {min(in_plane):.3f} mm")
            print(f"    Max : {max(in_plane):.3f} mm")
            print(f"    Mean: {np.mean(in_plane):.3f} mm")
            print(f"    → Resampling to 1.5mm isotropic is justified")

        # ── Slice count summary ───────────────────────────────────────────
        if shape_stats:
            depths = [s[2] for s in shape_stats]
            print(f"\n  Slices per volume (depth dimension):")
            print(f"    Min : {min(depths)}")
            print(f"    Max : {max(depths)}")
            print(f"    Mean: {np.mean(depths):.1f}")

    # ── Final verdict ─────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    if grand_total_issues == 0:
        print("  RESULT: Dataset is clean and ready for preprocessing.")
    else:
        print(f"  RESULT: {grand_total_issues} issue(s) found — "
              f"review above before proceeding.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    verify_dataset()