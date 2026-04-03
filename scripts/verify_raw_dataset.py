# scripts/verify_raw_dataset.py

from pathlib import Path

RAW_DIR = Path("data/raw")

def verify_dataset():
    print("\n Verifying raw ACDC dataset structure...\n")

    for split in ["training", "testing"]:
        split_dir = RAW_DIR / split
        patients = sorted([p for p in split_dir.iterdir() if p.is_dir()])

        print(f"  {split.upper()}: {len(patients)} patients found")

        missing_info   = []
        missing_frames = []

        for patient_dir in patients:
            pid = patient_dir.name

            # Check Info.cfg exists
            if not (patient_dir / "Info.cfg").exists():
                missing_info.append(pid)
                continue

            # Read ED and ES frame numbers
            with open(patient_dir / "Info.cfg") as f:
                info = {}
                for line in f:
                    if ":" in line:
                        k, v = line.strip().split(":", 1)
                        info[k.strip()] = v.strip()

            ed = info.get("ED")
            es = info.get("ES")

            # Check all 4 required files exist
            for frame in [ed, es]:
                img = patient_dir / f"{pid}_frame{int(frame):02d}.nii.gz"
                msk = patient_dir / f"{pid}_frame{int(frame):02d}_gt.nii.gz"
                if not img.exists() or not msk.exists():
                    missing_frames.append(f"{pid} frame{frame}")

        if missing_info:
            print(f"   Missing Info.cfg : {missing_info}")
        if missing_frames:
            print(f"   Missing frames   : {missing_frames}")
        if not missing_info and not missing_frames:
            print(f"   All files present and accounted for\n")

if __name__ == "__main__":
    verify_dataset()