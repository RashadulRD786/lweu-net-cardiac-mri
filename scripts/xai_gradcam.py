"""
XAI Method 1 — Guided Grad-CAM for Cardiac MRI Segmentation
=============================================================
Generates per-class (LV, RV, MYO) Guided Grad-CAM attention heatmaps for
three models side-by-side to visually validate Claim 1:

    "EnhancedBlock learns to focus on correct cardiac structures"

Models compared:
    Row 1 — U-Net Baseline      (checkpoints/unet_baseline/best_model.pth)
    Row 2 — LWEU-Net Base       (checkpoints/lweunet_base/best_model.pth)
    Row 3 — LWEU-Net V2         (checkpoints/lweunet_v2/best_model.pth)

Output figures:
    figures/xai/gradcam_dec3.png         Primary thesis figure (Dec3 layer)
    figures/xai/gradcam_bottleneck.png   Supplementary figure (Bottleneck layer)
    figures/xai/selected_slice_info.txt  Reproducibility record

Target layers (verified via inspect_layers.py):
    U-Net Baseline : bottleneck[0]          | dec3.conv
    LWEU-Net Base  : bottleneck.block        | decoder.level3.conv
    LWEU-Net V2    : bottleneck.block        | decoder.level3.block

Usage:
    python scripts/xai_gradcam.py

Citations:
    Selvaraju et al. (2017) ICCV     — Original Grad-CAM
    Springenberg et al. (2014) arXiv — Guided Backpropagation
    Vinogradova et al. (2020) AAAI   — Grad-CAM for segmentation tasks
    Adebayo et al. (2018) NeurIPS    — Faithfulness limitation of Guided BP
"""

import os
import sys
import glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
# import torch.nn.functional as F

# Make src/ importable from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#from pytorch_grad_cam import GuidedGradCAM
from pytorch_grad_cam import GradCAM #GuidedBackpropReLUModel
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.models.unet_baseline import UNetBaseline
from src.models.lweunet.lweunet import LWEUNet
from src.models.lweunet.lweunet_v2 import LWEUNetV2


# =============================================================================
# CONFIGURATION
# All paths and constants are defined here — edit only this section if needed
# =============================================================================

CHECKPOINTS = {
    "U-Net Baseline" : "checkpoints/unet_baseline/best_model.pth",
    "LWEU-Net Base"  : "checkpoints/lweunet_base/best_model.pth",
    "LWEU-Net V2"    : "checkpoints/lweunet_v2/best_model.pth",
}

TEST_DATA_DIR     = "data/preprocessed/test"
OUTPUT_DIR        = "figures/xai"
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Segmentation class definitions — must match training label convention
CLASS_NAMES  = {1: "RV", 2: "MYO", 3: "LV"}
# Column order in the figure: LV first (clearest), then RV, then MYO (hardest)
CAM_COL_ORDER = [3, 1, 2]   # LV=3, RV=1, MYO=2

# Minimum foreground pixels required per structure for slice selection
MIN_PIXELS_PER_CLASS = 100

# 4-class segmentation colormap: BG=black, RV=red, MYO=lime, LV=blue
SEG_CMAP = ListedColormap(["black", "red", "lime", "blue"])


# =============================================================================
# SECTION 1 — MODEL LOADING
# =============================================================================

def build_model(name: str) -> torch.nn.Module:
    """Instantiate the correct model architecture for each model name."""
    if name == "U-Net Baseline":
        return UNetBaseline(in_channels=1, num_classes=4)
    elif name == "LWEU-Net Base":
        # use_eca=False — pure base model, no ECA, matches checkpoint
        return LWEUNet(in_channels=1, num_classes=4, use_eca=False)
    elif name == "LWEU-Net V2":
        return LWEUNetV2(in_channels=1, num_classes=4)
    else:
        raise ValueError(f"Unknown model name: {name}")


def load_model(name: str, ckpt_path: str) -> torch.nn.Module:
    """
    Load model weights from checkpoint.
    Supports two checkpoint formats:
        - {'model_state': ..., 'epoch': ..., ...}   (trainer.py format)
        - raw state_dict
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = build_model(name)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)

    # Try both key names used across the project
    state = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state)
    model.eval()
    model.to(DEVICE)
    print(f"    Loaded  : {name}  ({ckpt_path})")
    return model


def get_target_layers(name: str, model: torch.nn.Module) -> dict:
    """
    Return the bottleneck and dec3 target layers for each model.
    Layer paths verified from inspect_layers.py output.

    Returns
    -------
    dict with keys 'bottleneck' and 'dec3', values are nn.Module objects
    """
    if name == "U-Net Baseline":
        return {
            "bottleneck" : model.bottleneck[0],      # ConvBlock inside nn.Sequential
            "dec3"       : model.dec3.conv,           # ConvBlock at decoder stage 3
        }
    elif name == "LWEU-Net Base":
        return {
            "bottleneck" : model.bottleneck.block,           # DepthwiseSepResidualBlock
            "dec3"       : model.decoder.level3.conv,        # DecoderConvBlock
        }
    elif name == "LWEU-Net V2":
        return {
            "bottleneck" : model.bottleneck.block,           # EnhancedBlock (with context gate)
            "dec3"       : model.decoder.level3.block,       # EnhancedBlock (with context gate)
        }


# =============================================================================
# SECTION 2 — SLICE SELECTION
# =============================================================================

def find_slice_pairs(test_dir: str) -> list:
    """
    Discover image/mask pairs in the test directory.
    Naming convention: <stem>_img.npy + <stem>_msk.npy (flat directory)
    Example: patient101_ED_slice00_img.npy + patient101_ED_slice00_msk.npy
    """
    pairs = []

    img_files = sorted(glob.glob(os.path.join(test_dir, "*_img.npy")))

    if not img_files:
        raise RuntimeError(
            f"No *_img.npy files found in {test_dir}. "
            "Check TEST_DATA_DIR in the configuration section."
        )

    for img_path in img_files:
        mask_path = img_path.replace("_img.npy", "_msk.npy")
        if os.path.exists(mask_path):
            fname = os.path.basename(img_path)
            pairs.append((fname, img_path, mask_path))

    print(f"    Found {len(pairs)} image/mask pairs in {test_dir}")
    return pairs

    # # Convention B — subdirectory structure images/ and masks/
    # img_dir  = os.path.join(test_dir, "images")
    # mask_dir = os.path.join(test_dir, "masks")
    # if os.path.isdir(img_dir) and os.path.isdir(mask_dir):
    #     for fname in sorted(os.listdir(img_dir)):
    #         if not fname.endswith(".npy"):
    #             continue
    #         img_path  = os.path.join(img_dir,  fname)
    #         mask_path = os.path.join(mask_dir, fname)
    #         if os.path.exists(mask_path):
    #             pairs.append((fname, img_path, mask_path))
    #     if pairs:
    #         print(f"    Found {len(pairs)} pairs (Convention B: images/ + masks/)")
    #         return pairs

    # raise RuntimeError(
    #     f"No image/mask pairs found in {test_dir}.\n"
    #     "Expected either:\n"
    #     "  (A) *_image.npy + *_mask.npy in the same directory, or\n"
    #     "  (B) images/*.npy + masks/*.npy subdirectories."
    # )


def select_best_slice(pairs: list) -> tuple:
    """
    Automatically select the most representative mid-ventricular slice.

    Selection criteria (applied in order):
      1. All three structures present: RV (1), MYO (2), LV (3)
      2. Each structure has >= MIN_PIXELS_PER_CLASS foreground pixels
      3. Among qualifying slices, minimise coefficient of variation across
         structure sizes — this biases toward balanced mid-ventricular slices
         where all three structures are comparably visible

    Returns
    -------
    (fname, img_path, mask_path) for the selected slice
    """
    candidates = []

    for fname, img_path, mask_path in pairs:
        mask = np.load(mask_path)

        rv_px  = int(np.sum(mask == 1))
        myo_px = int(np.sum(mask == 2))
        lv_px  = int(np.sum(mask == 3))

        if (rv_px  >= MIN_PIXELS_PER_CLASS and
            myo_px >= MIN_PIXELS_PER_CLASS and
            lv_px  >= MIN_PIXELS_PER_CLASS):

            counts  = np.array([rv_px, myo_px, lv_px], dtype=float)
            cv      = counts.std() / counts.mean()  # coefficient of variation
            candidates.append((cv, fname, img_path, mask_path, rv_px, myo_px, lv_px))

    print(f"    Qualifying slices (all 3 structures >= {MIN_PIXELS_PER_CLASS}px): "
          f"{len(candidates)} / {len(pairs)}")

    if not candidates:
        raise RuntimeError(
            "No qualifying slices found. "
            f"Reduce MIN_PIXELS_PER_CLASS (currently {MIN_PIXELS_PER_CLASS}) "
            "or check test data labels."
        )

    candidates.sort(key=lambda x: x[0])   # ascending CV = most balanced first
    cv, fname, img_path, mask_path, rv_px, myo_px, lv_px = candidates[0]

    print(f"    Selected  : {fname}")
    print(f"    Structure sizes — RV: {rv_px}px  MYO: {myo_px}px  LV: {lv_px}px")
    print(f"    Balance score (CV) : {cv:.3f}  (lower = more balanced)")
    return fname, img_path, mask_path


# =============================================================================
# SECTION 3 — GRAD-CAM GENERATION
# =============================================================================

def get_prediction(model: torch.nn.Module, input_tensor: torch.Tensor) -> np.ndarray:
    """
    Run forward pass and return the predicted segmentation mask.

    Returns
    -------
    pred_np : (H, W) int64 numpy array of predicted class indices
    """
    with torch.no_grad():
        logits = model(input_tensor)           # (1, 4, H, W)
    pred = logits.argmax(dim=1).squeeze(0)     # (H, W)
    return pred.cpu().numpy().astype(np.int64)


def generate_guided_gradcam(
    model        : torch.nn.Module,
    target_layer : torch.nn.Module,
    input_tensor : torch.Tensor,
    class_idx    : int,
) -> np.ndarray:
    """
    Generate Guided Seg-Grad-CAM heatmap for one class at one layer.

    Combines:
      - Seg-Grad-CAM (Vinogradova 2020): soft ROI mask from predicted
        probabilities focuses gradients on the segmented region only
      - GradCAM (Selvaraju 2017): spatially weighted feature map activations
      - Guided Backpropagation (Springenberg 2014): manual implementation
        with ReLU hooks — suppresses noisy negative gradients

    Parameters
    ----------
    model         : nn.Module in eval mode
    target_layer  : the layer whose activations/gradients are captured
    input_tensor  : (1, 1, H, W) float tensor on DEVICE
    class_idx     : 1=RV, 2=MYO, 3=LV

    Returns
    -------
    heatmap : (H, W) float32 numpy array normalised to [0, 1]
    """

    """
    Generate Seg-Grad-CAM heatmap for one class at one layer.

    Implements Seg-Grad-CAM (Vinogradova et al., 2020 AAAI):
      - Soft ROI mask from predicted probabilities (faithful to ClassRoI
        in the official SegGradCAM repo, kiraving/SegGradCAM)
      - GradCAM (Selvaraju et al., 2017) with segmentation-aware target
      - SemanticSegmentationTarget focuses gradients on predicted region only

    Note: Guided Backpropagation was evaluated but removed — the product
    of GradCAM and guided backprop gradients collapsed to near-zero for
    all three structures due to ReLU sparsity, producing uninterpretable
    heatmaps. GradCAM alone with Seg-Grad-CAM target formulation is the
    method used and cited.
    """

    # ── Step 1: Soft ROI mask — faithful to Vinogradova 2020 ─────────
    with torch.no_grad():
        probs      = torch.softmax(model(input_tensor), dim=1)  # (1, 4, H, W)
        argmax_np  = probs.argmax(dim=1).squeeze(0).cpu().numpy()
        class_prob = probs[0, class_idx].cpu().numpy()

    soft_mask = np.round(
        class_prob * (argmax_np == class_idx)
    ).astype(np.float32)                                    # (H, W) in {0, 1}

    # ── Step 2: GradCAM — coarse spatial attention ────────────────────
    target      = SemanticSegmentationTarget(class_idx, soft_mask)
    cam         = GradCAM(model=model, target_layers=[target_layer])
    gradcam_out = cam(input_tensor=input_tensor, targets=[target])
    gradcam_map = gradcam_out[0]                            # (H, W) in [0, 1]

    
    # return guided_gradcam.astype(np.float32)
    return gradcam_out[0].astype(np.float32)   # (H, W) in [0, 1]

def overlay_cam_on_mri(img_np: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on a grayscale MRI slice using jet colormap.

    Parameters
    ----------
    img_np  : (H, W) float32 in [0, 1] — normalised MRI slice
    heatmap : (H, W) float32 in [0, 1] — Grad-CAM output

    Returns
    -------
    overlay : (H, W, 3) uint8 RGB image — jet heatmap blended over grayscale MRI
    """

    """
    Overlay a Grad-CAM heatmap on a grayscale MRI slice using jet colormap.
    Gamma correction (gamma=0.4) is applied to boost low-value activations
    and make the full attention pattern visible, not just peak pixels.

    """

    # Gamma correction — boosts dim activations before overlay
    # heatmap_enhanced = np.power(np.clip(heatmap, 0, 1), 0.4)

    rgb = np.stack([img_np, img_np, img_np], axis=-1).astype(np.float32)
    rgb = np.clip(rgb, 0.0, 1.0)
    return show_cam_on_image(rgb, heatmap, use_rgb=True)


# =============================================================================
# SECTION 4 — FIGURE COMPOSITION
# =============================================================================

def compose_and_save_figure(
    models_data : list,
    img_display : np.ndarray,
    mask_np     : np.ndarray,
    layer_name  : str,
    save_path   : str,
) -> None:
    """
    Compose and save the 3-row × 6-column thesis figure.

    Layout
    ------
    Columns : Input MRI | Ground Truth | LV Attention | RV Attention | MYO Attention | Prediction
    Rows    : U-Net Baseline | LWEU-Net Base | LWEU-Net V2

    Parameters
    ----------
    models_data  : list of dicts — one per model row, each containing:
                   {'name': str, 'heatmaps': {class_idx: heatmap}, 'prediction': np.ndarray}
    img_display  : (H, W) float [0,1] — normalised MRI for display
    mask_np      : (H, W) int — ground truth segmentation mask
    layer_name   : 'dec3' or 'bottleneck' — used in figure title
    save_path    : full path to save the figure
    """
    n_rows = len(models_data)
    n_cols = 6

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize     = (18, 9),
        gridspec_kw = {"wspace": 0.04, "hspace": 0.12},
    )

    col_titles = [
        "Input MRI",
        "Ground Truth",
        "LV Attention",
        "RV Attention",
        "MYO Attention",
        "Prediction",
    ]

    # ── Column headers (top row only) ────────────────────────────────
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(
            title, fontsize=11, fontweight="bold", pad=7
        )

    # ── One row per model ─────────────────────────────────────────────
    for row_idx, mdata in enumerate(models_data):
        model_name = mdata["name"]
        heatmaps   = mdata["heatmaps"]    # dict {class_idx: (H,W) heatmap}
        pred_np    = mdata["prediction"]  # (H, W) int

        # Row label on the left y-axis
        axes[row_idx, 0].set_ylabel(
            model_name,
            fontsize   = 10,
            fontweight = "bold",
            rotation   = 90,
            labelpad   = 8,
            va         = "center",
        )

        # Col 0 — Input MRI
        axes[row_idx, 0].imshow(img_display, cmap="gray", vmin=0, vmax=1)

        # Col 1 — Ground Truth mask
        axes[row_idx, 1].imshow(
            mask_np, cmap=SEG_CMAP, vmin=0, vmax=3, interpolation="nearest"
        )

        # Cols 2–4 — Per-class Guided Grad-CAM overlays
        # Order: LV (col 2), RV (col 3), MYO (col 4)
        for col_offset, class_idx in enumerate(CAM_COL_ORDER):
            overlay = overlay_cam_on_mri(img_display, heatmaps[class_idx])
            axes[row_idx, 2 + col_offset].imshow(overlay)

        # Col 5 — Predicted segmentation mask
        axes[row_idx, 5].imshow(
            pred_np, cmap=SEG_CMAP, vmin=0, vmax=3, interpolation="nearest"
        )

        # Remove all tick marks from every panel
        for col_idx in range(n_cols):
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])

    # ── Legend ────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color="black", label="Background"),
        mpatches.Patch(color="red",   label="RV"),
        mpatches.Patch(color="lime",  label="MYO"),
        mpatches.Patch(color="blue",  label="LV"),
    ]
    fig.legend(
        handles         = legend_patches,
        loc             = "lower center",
        ncol            = 4,
        fontsize        = 9,
        frameon         = True,
        bbox_to_anchor  = (0.5, 0.01),
    )

    # ── Figure title ──────────────────────────────────────────────────
    layer_label = "Decoder Stage 3" if layer_name == "dec3" else "Bottleneck"
    fig.suptitle(
        f"Seg-Grad-CAM  —  {layer_label} Layer  |  Per-Class Attention Maps (LV / RV / MYO)",
        fontsize   = 12,
        fontweight = "bold",
        y          = 0.99,
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Saved : {save_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 62)
    print("  XAI Method 1 — Guided Grad-CAM")
    print("  Claim: EnhancedBlock learns correct anatomical localisation")
    print("=" * 62)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 1 — Select representative test slice ─────────────────────
    print("\n[1/4] Selecting representative test slice...")
    pairs                          = find_slice_pairs(TEST_DATA_DIR)
    fname, img_path, mask_path     = select_best_slice(pairs)

    img_np   = np.load(img_path).astype(np.float32)    # (H, W) raw normalised MRI
    mask_np  = np.load(mask_path).astype(np.int64)     # (H, W) ground truth labels

    # Normalise to [0, 1] for display and heatmap overlay
    img_min, img_max = img_np.min(), img_np.max()
    img_display = (img_np - img_min) / (img_max - img_min + 1e-8)

    # Input tensor: (1, 1, H, W) — batch size 1, 1 channel, spatial dims
    input_tensor = (
        torch.from_numpy(img_np)
        .unsqueeze(0)   # add batch dim  → (1, H, W)
        .unsqueeze(0)   # add channel dim → (1, 1, H, W)
        .float()
        .to(DEVICE)
    )

    # Write reproducibility record
    info_path = os.path.join(OUTPUT_DIR, "selected_slice_info.txt")
    with open(info_path, "w") as f:
        f.write(f"Selected slice : {fname}\n")
        f.write(f"Image path     : {img_path}\n")
        f.write(f"Mask path      : {mask_path}\n")
        f.write(f"Image range    : [{img_min:.4f}, {img_max:.4f}]\n")
        f.write(f"RV pixels      : {int(np.sum(mask_np == 1))}\n")
        f.write(f"MYO pixels     : {int(np.sum(mask_np == 2))}\n")
        f.write(f"LV pixels      : {int(np.sum(mask_np == 3))}\n")
    print(f"    Reproducibility record: {info_path}")

    # ── Step 2 — Load all three models ───────────────────────────────
    print("\n[2/4] Loading models...")
    models = {}
    for name, ckpt_path in CHECKPOINTS.items():
        models[name] = load_model(name, ckpt_path)

    # ── Step 3 — Generate heatmaps for both target layers ────────────
    print("\n[3/4] Generating Guided Grad-CAM heatmaps...")

    # Accumulate results per layer: {"dec3": [...], "bottleneck": [...]}
    # Each entry in the list is one model row dict for compose_figure
    results = {"dec3": [], "bottleneck": []}

    for model_name, model in models.items():
        print(f"\n  Model: {model_name}")
        target_layers = get_target_layers(model_name, model)

        # Prediction is computed once per model (same for both layers)
        pred_np = get_prediction(model, input_tensor)
        print(f"    Predicted unique labels: {np.unique(pred_np).tolist()}")

        for layer_key, target_layer in target_layers.items():
            heatmaps = {}

            for class_idx, class_name in CLASS_NAMES.items():
                print(f"    [{layer_key}] Class {class_idx} ({class_name}) ... ", end="", flush=True)
                heatmap = generate_guided_gradcam(
                    model        = model,
                    target_layer = target_layer,
                    input_tensor = input_tensor,
                    class_idx    = class_idx,
                    #pred_mask_np = pred_np,
                )
                heatmaps[class_idx] = heatmap
                print("done")

            results[layer_key].append({
                "name"       : model_name,
                "heatmaps"   : heatmaps,
                "prediction" : pred_np,
            })

    # ── Step 4 — Compose and save figures ────────────────────────────
    print("\n[4/4] Composing figures...")

    for layer_key, models_data in results.items():
        save_path = os.path.join(OUTPUT_DIR, f"gradcam_{layer_key}.png")
        compose_and_save_figure(
            models_data = models_data,
            img_display = img_display,
            mask_np     = mask_np,
            layer_name  = layer_key,
            save_path   = save_path,
        )

    print("\n" + "=" * 62)
    print("  Guided Grad-CAM complete.")
    print(f"  Primary figure  : {OUTPUT_DIR}/gradcam_dec3.png")
    print(f"  Secondary figure: {OUTPUT_DIR}/gradcam_bottleneck.png")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
