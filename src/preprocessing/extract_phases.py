# src/preprocessing/extract_phases.py
#
# Responsibility: Extract ED and ES phase information from a patient directory.
# This is always the first step — every other preprocessing step depends on
# knowing which frame numbers correspond to ED and ES.

from pathlib import Path


def read_info_cfg(patient_dir: Path) -> dict:
    """
    Parse the Info.cfg file for a single patient.

    The Info.cfg file contains:
        ED    : frame number of End-Diastole (heart fully relaxed)
        ES    : frame number of End-Systole  (heart fully contracted)
        Group : pathology class (NOR, DCM, HCM, MINF, RV)
        Height: patient height in cm
        Weight: patient weight in kg

    Args:
        patient_dir : Path to the patient folder
                      e.g. data/raw/training/patient001/

    Returns:
        info : dict with keys ED, ES, Group, Height, Weight
    """
    cfg_path = patient_dir / "Info.cfg"

    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Info.cfg not found in {patient_dir}. "
            f"Check that the patient directory is correct."
        )

    info = {}
    with open(cfg_path, "r") as f:
        for line in f:
            line = line.strip()
            if ":" in line:
                key, value = line.split(":", 1)
                info[key.strip()] = value.strip()

    # Validate required keys exist
    for required_key in ["ED", "ES", "Group"]:
        if required_key not in info:
            raise ValueError(
                f"Missing key '{required_key}' in {cfg_path}. "
                f"File may be corrupt or incomplete."
            )

    return info


def find_nifti(base_path: Path) -> Path:
    """
    Find a NIfTI file at base_path with either .nii.gz or .nii extension.

    ACDC stores files as .nii.gz but some tools extract them as .nii.
    This handles both cases transparently.

    Args:
        base_path : Path without extension
                    e.g. data/raw/training/patient001/patient001_frame01

    Returns:
        Path to the found NIfTI file

    Raises:
        FileNotFoundError if neither extension exists
    """
    for ext in [".nii.gz", ".nii"]:
        candidate = Path(str(base_path) + ext)
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No NIfTI file found at {base_path} "
        f"(tried .nii.gz and .nii)"
    )


def get_phase_paths(patient_dir: Path) -> dict:
    """
    Get all file paths and metadata for a single patient's ED and ES phases.

    This is the main function called by the pipeline. It combines
    read_info_cfg() and find_nifti() into a single structured output
    that the rest of the pipeline consumes.

    Args:
        patient_dir : Path to the patient folder
                      e.g. data/raw/training/patient001/

    Returns:
        result : dict containing:
            patient_id  : str  e.g. "patient001"
            group       : str  e.g. "DCM" (ARV used instead of RV)
            ed_frame    : int  frame number for ED
            es_frame    : int  frame number for ES
            ed_img_path : Path to ED image NIfTI
            ed_msk_path : Path to ED mask NIfTI
            es_img_path : Path to ES image NIfTI
            es_msk_path : Path to ES mask NIfTI
            height      : float patient height in cm
            weight      : float patient weight in kg
    """
    patient_dir = Path(patient_dir)
    pid         = patient_dir.name  # e.g. "patient001"

    # Read metadata from Info.cfg
    info = read_info_cfg(patient_dir)

    ed_frame = int(info["ED"])
    es_frame = int(info["ES"])

    # Normalize group label — dataset uses "RV" but official name is "ARV"
    group = info["Group"]
    if group == "RV":
        group = "ARV"

    # Build file paths for both phases
    ed_base = patient_dir / f"{pid}_frame{ed_frame:02d}"
    es_base = patient_dir / f"{pid}_frame{es_frame:02d}"

    return {
        "patient_id"  : pid,
        "group"       : group,
        "ed_frame"    : ed_frame,
        "es_frame"    : es_frame,
        "ed_img_path" : find_nifti(ed_base),
        "ed_msk_path" : find_nifti(Path(str(ed_base) + "_gt")),
        "es_img_path" : find_nifti(es_base),
        "es_msk_path" : find_nifti(Path(str(es_base) + "_gt")),
        "height"      : float(info.get("Height", 0)),
        "weight"      : float(info.get("Weight", 0)),
    }


# ── Quick test ────────────────────────────────────────────────────────────────
# Run directly to verify this file works on your data:
#   python src/preprocessing/extract_phases.py

if __name__ == "__main__":
    from pathlib import Path

    test_patient = Path("data/raw/training/patient001")

    print("Testing extract_phases.py on patient001...")
    print("=" * 50)

    result = get_phase_paths(test_patient)

    print(f"Patient ID    : {result['patient_id']}")
    print(f"Group         : {result['group']}")
    print(f"ED frame      : {result['ed_frame']}")
    print(f"ES frame      : {result['es_frame']}")
    print(f"Height / Weight: {result['height']} cm / {result['weight']} kg")
    print()
    print(f"ED image path : {result['ed_img_path']}")
    print(f"ED mask path  : {result['ed_msk_path']}")
    print(f"ES image path : {result['es_img_path']}")
    print(f"ES mask path  : {result['es_msk_path']}")
    print()

    # Verify all paths actually exist
    all_exist = all([
        result['ed_img_path'].exists(),
        result['ed_msk_path'].exists(),
        result['es_img_path'].exists(),
        result['es_msk_path'].exists(),
    ])
    print(f"All files exist: {all_exist}")
    print("=" * 50)
    print("extract_phases.py — OK" if all_exist else "FAILED — check paths")