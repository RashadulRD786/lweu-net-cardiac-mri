"""
Worst RV Case Analysis: LiteU-Net vs EnhU-Net
Finds test slices where LiteU-Net has worst RV segmentation vs EnhU-Net,
then generates side-by-side overlay figures.
"""

import sys
import os
import re
import glob
import csv

# Add project root to sys.path
PROJECT_ROOT = "/home/rashadulnafisriyad/FYP_UNet/LWEU_NET"
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.lweunet.lweunet    import LWEUNet
from src.models.lweunet.lweunet_v2 import LWEUNetV2


# ── Paths ──────────────────────────────────────────────────────────────────────
TEST_DIR        = os.path.join(PROJECT_ROOT, "data", "preprocessed", "test")
CKPT_LITE       = os.path.join(PROJECT_ROOT, "checkpoints", "lweunet_base", "best_model.pth")
CKPT_ENH        = os.path.join(PROJECT_ROOT, "checkpoints", "lweunet_v2",   "best_model.pth")
OUT_DIR         = os.path.join(PROJECT_ROOT, "results", "worst_case_analysis")
CSV_PATH        = os.path.join(OUT_DIR, "worst_rv_slices.csv")

os.makedirs(OUT_DIR, exist_ok=True)

RV_CLASS = 1   # label index for right ventricle


# ── Helpers ────────────────────────────────────────────────────────────────────
def dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Binary Dice coefficient. Returns NaN if denominator is zero."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    denom = pred_mask.sum() + gt_mask.sum()
    if denom == 0:
        return float("nan")
    return 2.0 * intersection / denom


def load_model(model_cls, ckpt_path, **kwargs):
    model = model_cls(**kwargs)
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state", ckpt)   # key is "model_state"
    model.load_state_dict(state)
    model.eval()
    return model


def run_inference(model, img_np: np.ndarray) -> np.ndarray:
    """
    img_np: (H, W) float32 array (already normalised)
    Returns predicted label map (H, W) as int.
    """
    tensor = torch.from_numpy(img_np).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    with torch.no_grad():
        logits = model(tensor)           # (1, C, H, W)
    pred = logits.argmax(dim=1).squeeze(0).numpy()   # (H, W)
    return pred.astype(np.int32)


# ── Discover test slices ───────────────────────────────────────────────────────
img_files = sorted(glob.glob(os.path.join(TEST_DIR, "*_img.npy")))
print(f"Found {len(img_files)} image files in {TEST_DIR}")

# Parse filenames: patientXXX_<phase>_sliceYY_img.npy
pattern = re.compile(r"(patient\d+)_(ED|ES)_slice(\d+)_img\.npy")

slices = []
for f in img_files:
    fname = os.path.basename(f)
    m = pattern.match(fname)
    if m:
        slices.append({
            "patient": m.group(1),
            "phase":   m.group(2),
            "slice":   m.group(3),
            "img_path": f,
            "msk_path": f.replace("_img.npy", "_msk.npy"),
        })

print(f"Parsed {len(slices)} valid slices (ED + ES)")


# ── Load models ────────────────────────────────────────────────────────────────
print("\nLoading LiteU-Net …")
lite_model = load_model(LWEUNet,   CKPT_LITE,
                        in_channels=1, num_classes=4,
                        base_filters=32, dropout_p=0.5, use_eca=False)
print("Loading EnhU-Net …")
enh_model  = load_model(LWEUNetV2, CKPT_ENH,
                        in_channels=1, num_classes=4,
                        base_filters=32, dropout_p=0.5)
print("Both models loaded and in eval() mode.\n")


# ── Inference loop ─────────────────────────────────────────────────────────────
results = []

for idx, sl in enumerate(slices):
    img = np.load(sl["img_path"]).astype(np.float32)
    msk = np.load(sl["msk_path"])

    gt_rv = (msk == RV_CLASS).astype(np.uint8)

    # Skip slices with no RV pixels in ground truth
    if gt_rv.sum() == 0:
        continue

    pred_lite = run_inference(lite_model, img)
    pred_enh  = run_inference(enh_model,  img)

    lite_rv = (pred_lite == RV_CLASS).astype(np.uint8)
    enh_rv  = (pred_enh  == RV_CLASS).astype(np.uint8)

    lite_dice = dice_score(lite_rv, gt_rv)
    enh_dice  = dice_score(enh_rv,  gt_rv)
    gap       = enh_dice - lite_dice          # positive means EnhU-Net better

    results.append({
        "patient":       sl["patient"],
        "phase":         sl["phase"],
        "slice":         sl["slice"],
        "lite_rv_dice":  lite_dice,
        "enh_rv_dice":   enh_dice,
        "gap":           gap,
        "img_path":      sl["img_path"],
        "msk_path":      sl["msk_path"],
        "pred_lite":     pred_lite,
        "pred_enh":      pred_enh,
        "img_np":        img,
        "gt_rv":         gt_rv,
    })

    if (idx + 1) % 100 == 0:
        print(f"  Processed {idx + 1}/{len(slices)} slices …")

print(f"\nTotal slices with RV ground truth: {len(results)}")


# ── Rank worst slices ──────────────────────────────────────────────────────────
# Sort: lite_rv_dice ascending (worst first), gap descending as tiebreaker
ranked = sorted(results, key=lambda r: (r["lite_rv_dice"], -r["gap"]))

worst10 = ranked[:10]

header = f"{'Rank':<5} {'Patient':<12} {'Phase':<6} {'Slice':<7} {'LiteDice':<10} {'EnhDice':<10} {'Gap':<8}"
print("\n" + "=" * 60)
print("TOP 10 WORST RV SLICES (LiteU-Net vs EnhU-Net)")
print("=" * 60)
print(header)
print("-" * 60)
for rank, r in enumerate(worst10, 1):
    print(f"{rank:<5} {r['patient']:<12} {r['phase']:<6} {r['slice']:<7} "
          f"{r['lite_rv_dice']:<10.4f} {r['enh_rv_dice']:<10.4f} {r['gap']:<8.4f}")
print("=" * 60)


# ── Save CSV ───────────────────────────────────────────────────────────────────
csv_fields = ["rank", "patient", "phase", "slice",
              "lite_rv_dice", "enh_rv_dice", "gap"]
with open(CSV_PATH, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=csv_fields)
    writer.writeheader()
    for rank, r in enumerate(ranked, 1):
        writer.writerow({
            "rank":          rank,
            "patient":       r["patient"],
            "phase":         r["phase"],
            "slice":         r["slice"],
            "lite_rv_dice":  round(r["lite_rv_dice"], 6),
            "enh_rv_dice":   round(r["enh_rv_dice"],  6),
            "gap":           round(r["gap"],           6),
        })
print(f"\nCSV saved to: {CSV_PATH}")


# ── Generate figures for top 5 worst slices ────────────────────────────────────
print("\nGenerating figures for top 5 worst slices …")
saved_figs = []

for rank, r in enumerate(ranked[:5], 1):
    img   = r["img_np"]
    gt_rv = r["gt_rv"]
    pred_lite_rv = (r["pred_lite"] == RV_CLASS).astype(np.uint8)
    pred_enh_rv  = (r["pred_enh"]  == RV_CLASS).astype(np.uint8)

    lite_d = r["lite_rv_dice"]
    enh_d  = r["enh_rv_dice"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    suptitle = (f"Patient {r['patient']} Phase {r['phase']} Slice {r['slice']}"
                f" — Worst RV Case #{rank}")
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")

    for ax, title, pred_rv, pred_color in [
        (axes[0], f"LiteU-Net RV Dice: {lite_d:.3f}", pred_lite_rv, "red"),
        (axes[1], f"EnhU-Net RV Dice:  {enh_d:.3f}",  pred_enh_rv,  "cyan"),
    ]:
        ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=11)
        ax.axis("off")

        # GT contour — dashed white
        if gt_rv.sum() > 0:
            ax.contour(gt_rv, levels=[0.5],
                       colors=["white"], linestyles=["dashed"], linewidths=[1.5])

        # Predicted contour — solid coloured
        if pred_rv.sum() > 0:
            ax.contour(pred_rv, levels=[0.5],
                       colors=[pred_color], linestyles=["solid"], linewidths=[1.5])

    plt.tight_layout()

    fname = f"{r['patient']}_{r['phase']}_slice{r['slice']}.png"
    fpath = os.path.join(OUT_DIR, fname)
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_figs.append(fpath)
    print(f"  Saved rank-{rank} figure → {fpath}")

print("\nDone! Saved figures:")
for fp in saved_figs:
    print(f"  {fp}")
