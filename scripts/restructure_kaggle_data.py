# scripts/restructure_kaggle_data.py

import shutil
from pathlib import Path

RAW_DIR = Path("data/raw")


def find_actual_nii_file(folder: Path):
    nii_files = list(folder.glob("*.nii")) + list(folder.glob("*.nii.gz"))
    if nii_files:
        return nii_files[0]
    return None


def restructure_patient(patient_dir: Path) -> None:
    for item in list(patient_dir.iterdir()):
        if item.is_dir() and item.suffix == ".nii":

            actual_file = find_actual_nii_file(item)

            if actual_file is None:
                print(f"  [WARNING] No .nii file found inside {item}")
                continue

            # Copy to a temp name OUTSIDE the folder first
            temp_dest = patient_dir / f"_temp_{item.name}"
            shutil.copy2(actual_file, temp_dest)

            # Remove the original folder entirely
            shutil.rmtree(item)

            # Rename temp file to the correct final name
            final_dest = patient_dir / item.name
            temp_dest.rename(final_dest)

            print(f"  [FIXED] {item.name}/  →  {final_dest.name}")


def restructure_dataset() -> None:
    print("\n Restructuring Kaggle ACDC dataset...\n")

    for split in ["training", "testing"]:
        split_dir = RAW_DIR / split
        if not split_dir.exists():
            print(f"[WARNING] {split_dir} not found, skipping.")
            continue

        patients = sorted([p for p in split_dir.iterdir() if p.is_dir()])
        print(f" {split.upper()} — {len(patients)} patients\n")

        for patient_dir in patients:
            print(f" Processing {patient_dir.name}...")
            restructure_patient(patient_dir)

    print("\n Restructuring complete.")
    print(" Now run: python scripts/verify_raw_dataset.py\n")


if __name__ == "__main__":
    restructure_dataset()