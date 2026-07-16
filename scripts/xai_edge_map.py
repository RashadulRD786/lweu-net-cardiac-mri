"""
XAI Method 3 — Edge Map Overlay on Seg-Grad-CAM
================================================
Overlays ground truth MYO boundary pixels on the MYO Seg-Grad-CAM
heatmap to validate Claim 3:

    "EnhancedBlock produces sharper boundary attention,
     supporting MYO HD95 improvement"

The MYO (myocardium) is a thin ring ~8-12mm thick. Any diffuse
attention bleeds into the LV cavity or epicardium. By overlaying
the true MYO boundary on the attention map, we directly test
whether the model's attention is boundary-aware or merely
region-aware.

Expected findings:
    U-Net Baseline  → boundary-aligned attention (reference ceiling)
    LWEU-Net Base   → diffuse, boundary-misaligned attention
    LWEU-Net V2     → tighter boundary alignment than Base,
                      consistent with MYO HD95 of 3.28mm

Same slice used as Method 1 (patient138_ED_slice04) for consistency.
Target layer: Decoder Stage 3 (Dec3) — same as Method 1 primary figure.

Output figures:
    figures/xai/edge_map_overlay.png    Primary thesis figure (3×4 grid)
    figures/xai/edge_map_scores.txt     Boundary alignment scores per model

Usage:
    python scripts/xai_edge_map.py

Citations:
    Bernard et al. (2018) IEEE TMI   — HD95 as boundary metric in cardiac MRI
    Tomsett et al. (2020) AAAI       — Saliency map verification against GT
    Vinogradova et al. (2020) AAAI   — Seg-Grad-CAM segmentation target
    Selvaraju et al. (2017) ICCV     — GradCAM foundation
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from scipy.ndimage import binary_erosion

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.models.unet_baseline import UNetBaseline
from src.models.lweunet.lweunet import LWEUNet
from src.models.lweunet.lweunet_v2 import LWEUNetV2


# =============================================================================
# CONFIGURATION
# =============================================================================

CHECKPOINTS = {
    "U-Net Baseline" : "checkpoints/unet_baseline/best_model.pth",
    "LWEU-Net Base"  : "checkpoints/lweunet_base/best_model.pth",
    "LWEU-Net V2"    : "checkpoints/lweunet_v2/best_model.pth",
}

# Slice selected in Method 1 — kept identical for thesis consistency
SELECTED_SLICE_INFO = "figures/xai/selected_slice_info.txt"
TEST_DATA_DIR        = "data/preprocessed/test"
OUTPUT_DIR           = "figures/xai"
DEVICE               = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MYO class index — must match training label convention
MYO_CLASS = 2

# Boundary alignment score: average heatmap value within N pixels of GT boundary
BOUNDARY_RADIUS_PX = 5

# 4-class segmentation colormap: BG=black, RV=red, MYO=lime, LV=blue
SEG_CMAP = ListedColormap(["black", "red", "lime", "blue"])


# =============================================================================
# SECTION 1 — MODEL LOADING
# =============================================================================

def build_model(name: str) -> torch.nn.Module:
    """Instantiate the correct architecture for each model name."""
    if name == "U-Net Baseline":
        return UNetBaseline(in_channels=1, num_classes=4)
    elif name == "LWEU-Net Base":
        return LWEUNet(in_channels=1, num_classes=4, use_eca=False)
    elif name == "LWEU-Net V2":
        return LWEUNetV2(in_channels=1, num_classes=4)
    else:
        raise ValueError(f"Unknown model name: {name}")


def load_model(name: str, ckpt_path: str) -> torch.nn.Module:
    """
    Load model weights from checkpoint.
    Supports both checkpoint formats used across the project.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = build_model(name)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state)
    model.eval()
    model.to(DEVICE)
    print(f"    Loaded: {name}")
    return model


def get_dec3_layer(name: str, model: torch.nn.Module) -> torch.nn.Module:
    """
    Return the Dec3 target layer for each model.
    Consistent with Method 1 (xai_gradcam.py) target layer selection.
    Layer paths verified from inspect_layers.py.
    """
    if name == "U-Net Baseline":
        return model.dec3.conv
    elif name == "LWEU-Net Base":
        return model.decoder.level3.conv
    elif name == "LWEU-Net V2":
        return model.decoder.level3.block


# =============================================================================
# SECTION 2 — SLICE LOADING
# =============================================================================

def load_selected_slice() -> tuple:
    """
    Load the slice selected in Method 1 from selected_slice_info.txt.
    Using the same slice across all XAI methods ensures thesis consistency.

    Returns
    -------
    (img_np, mask_np, fname) where:
        img_np  : (H, W) float32 raw MRI
        mask_np : (H, W) int64 ground truth labels
        fname   : filename string for logging
    """
    if not os.path.exists(SELECTED_SLICE_INFO):
        raise FileNotFoundError(
            f"Selected slice info not found: {SELECTED_SLICE_INFO}\n"
            "Run xai_gradcam.py first to generate this file."
        )

    info = {}
    with open(SELECTED_SLICE_INFO, "r") as f:
        for line in f:
            if ":" in line:
                key, val = line.strip().split(":", 1)
                info[key.strip()] = val.strip()

    img_path  = info["Image path"]
    mask_path = info["Mask path"]
    fname     = info["Selected slice"]

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    img_np  = np.load(img_path).astype(np.float32)
    mask_np = np.load(mask_path).astype(np.int64)

    print(f"    Slice    : {fname}")
    print(f"    MYO px   : {int(np.sum(mask_np == MYO_CLASS))}")
    print(f"    Image range: [{img_np.min():.4f}, {img_np.max():.4f}]")

    return img_np, mask_np, fname


# =============================================================================
# SECTION 3 — BOUNDARY EXTRACTION
# =============================================================================

def extract_myo_boundary(mask_np: np.ndarray) -> np.ndarray:
    """
    Extract the ground truth MYO boundary pixels using morphological
    edge detection.

    Method: boundary = MYO_mask − eroded(MYO_mask)
    This gives a 1-pixel-thin ring tracing both the inner edge
    (MYO-LV interface) and the outer edge (MYO-background interface).

    Parameters
    ----------
    mask_np : (H, W) int64 ground truth segmentation mask

    Returns
    -------
    boundary : (H, W) bool array — True at boundary pixels
    """
    myo_binary = (mask_np == MYO_CLASS).astype(bool)

    # 3×3 structuring element for 8-connectivity erosion
    struct = np.ones((3, 3), dtype=bool)
    eroded = binary_erosion(myo_binary, structure=struct)

    boundary = myo_binary & ~eroded
    print(f"    MYO boundary pixels: {int(boundary.sum())}")
    return boundary


def dilate_boundary_for_score(boundary: np.ndarray,
                               radius: int) -> np.ndarray:
    """
    Dilate the boundary by `radius` pixels to create a band around the
    true boundary for the alignment score computation.

    A pixel is 'near the boundary' if it falls within this dilated band.

    Parameters
    ----------
    boundary : (H, W) bool boundary mask
    radius   : dilation radius in pixels

    Returns
    -------
    band : (H, W) bool array
    """
    from scipy.ndimage import binary_dilation
    struct = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    return binary_dilation(boundary, structure=struct)


# =============================================================================
# SECTION 4 — SEG-GRAD-CAM GENERATION (MYO only)
# =============================================================================

def generate_myo_seggradcam(
    model        : torch.nn.Module,
    target_layer : torch.nn.Module,
    input_tensor : torch.Tensor,
) -> np.ndarray:
    """
    Generate Seg-Grad-CAM heatmap for MYO (class 2) at the specified layer.

    Implements Seg-Grad-CAM (Vinogradova et al., 2020):
      - Soft ROI mask: predicted probability for MYO × (argmax == MYO)
      - GradCAM with SemanticSegmentationTarget focuses gradients
        on the predicted MYO region only

    Parameters
    ----------
    model         : nn.Module in eval mode
    target_layer  : Dec3 layer for this model
    input_tensor  : (1, 1, H, W) float tensor on DEVICE

    Returns
    -------
    heatmap : (H, W) float32 array in [0, 1]
    """
    # ── Soft ROI mask (Vinogradova 2020) ─────────────────────────────
    with torch.no_grad():
        probs      = torch.softmax(model(input_tensor), dim=1)  # (1, 4, H, W)
        argmax_np  = probs.argmax(dim=1).squeeze(0).cpu().numpy()
        class_prob = probs[0, MYO_CLASS].cpu().numpy()

    # No np.round() — true soft mask preserves confidence weighting
    soft_mask = (class_prob * (argmax_np == MYO_CLASS)).astype(np.float32)
    target    = SemanticSegmentationTarget(MYO_CLASS, soft_mask)

    # ── GradCAM with segmentation-aware target ────────────────────────
    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        gradcam_out = cam(input_tensor=input_tensor, targets=[target])

    return gradcam_out[0].astype(np.float32)   # (H, W) in [0, 1]


# =============================================================================
# SECTION 5 — BOUNDARY ALIGNMENT SCORE
# =============================================================================

def compute_alignment_score(heatmap: np.ndarray,
                             boundary_band: np.ndarray) -> float:
    """
    Compute the boundary alignment score.

    Definition: mean heatmap value within BOUNDARY_RADIUS_PX pixels
    of the GT MYO boundary.

    Interpretation:
      - Higher score = model attention concentrated near true boundary
      - Lower score  = model attention diffuse / misaligned

    Parameters
    ----------
    heatmap       : (H, W) float32 Seg-Grad-CAM heatmap in [0, 1]
    boundary_band : (H, W) bool — pixels within BOUNDARY_RADIUS_PX of boundary

    Returns
    -------
    score : float in [0, 1]
    """
    band_pixels = heatmap[boundary_band]
    if len(band_pixels) == 0:
        return 0.0
    return float(band_pixels.mean())

def compute_concentration_ratio(heatmap    : np.ndarray,
                                 boundary_band: np.ndarray,
                                 myo_mask   : np.ndarray) -> tuple:
    """
    Compute both the absolute alignment score and the boundary
    concentration ratio.

    Concentration ratio = mean heatmap in boundary band
                          / mean heatmap in full MYO region

    Ratio > 1.0 → model attends MORE to boundary than MYO interior
    Ratio < 1.0 → model attends MORE to MYO interior than boundary
    Ratio = 1.0 → uniform attention across the MYO region

    This removes the normalization bias of the absolute score.

    Returns
    -------
    (abs_score, ratio) : both float values
    """
    boundary_pixels = heatmap[boundary_band]
    myo_pixels      = heatmap[myo_mask]

    abs_score = float(boundary_pixels.mean()) if len(boundary_pixels) > 0 else 0.0
    myo_mean  = float(myo_pixels.mean())      if len(myo_pixels)      > 0 else 1e-8

    ratio = abs_score / (myo_mean + 1e-8)
    return abs_score, ratio


# =============================================================================
# SECTION 6 — FIGURE COMPOSITION
# =============================================================================

def overlay_boundary_on_heatmap(
    img_np   : np.ndarray,
    heatmap  : np.ndarray,
    boundary : np.ndarray,
    linewidth: int = 2,
) -> np.ndarray:
    """
    Create a heatmap overlay image with the GT MYO boundary drawn
    on top as a bright white contour.

    Parameters
    ----------
    img_np   : (H, W) float32 in [0, 1] normalised MRI
    heatmap  : (H, W) float32 in [0, 1] Seg-Grad-CAM heatmap
    boundary : (H, W) bool boundary pixels
    linewidth: thickness of boundary line in pixels

    Returns
    -------
    overlay : (H, W, 3) uint8 RGB image
    """
    rgb     = np.stack([img_np, img_np, img_np], axis=-1).astype(np.float32)
    rgb     = np.clip(rgb, 0.0, 1.0)
    overlay = show_cam_on_image(rgb, heatmap, use_rgb=True).copy()

    # Dilate boundary slightly for visibility, then draw white
    from scipy.ndimage import binary_dilation
    if linewidth > 1:
        struct   = np.ones((linewidth, linewidth), dtype=bool)
        boundary = binary_dilation(boundary, structure=struct)

    overlay[boundary] = [255, 255, 255]    # white boundary line
    return overlay


def compose_edge_map_figure(
    models_data : list,
    img_np      : np.ndarray,
    mask_np     : np.ndarray,
    boundary    : np.ndarray,
    scores      : dict,
    save_path   : str,
) -> None:
    """
    Compose and save the 3-row × 4-column thesis figure.

    Layout
    ------
    Columns:
        0 — Input MRI with GT boundary overlay
        1 — Ground Truth mask (MYO highlighted)
        2 — MYO Seg-Grad-CAM heatmap (no boundary)
        3 — MYO Seg-Grad-CAM heatmap + GT boundary (KEY COLUMN)

    Rows:
        U-Net Baseline | LWEU-Net Base | LWEU-Net V2

    The row label also shows the boundary alignment score.

    Parameters
    ----------
    models_data : list of dicts — one per model:
                  {'name': str, 'heatmap': (H,W) array}
    img_np      : (H, W) float [0,1] normalised MRI
    mask_np     : (H, W) int ground truth mask
    boundary    : (H, W) bool GT MYO boundary pixels
    scores      : {model_name: float} alignment scores
    save_path   : output file path
    """
    n_rows = len(models_data)
    n_cols = 4

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize     = (18, 9),
        gridspec_kw = {"wspace": 0.05, "hspace": 0.18},
    )

    col_titles = [
        "Input MRI + GT Boundary",
        "Ground Truth",
        "MYO Attention",
        "MYO Attention + GT Boundary",
    ]
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(
            title, fontsize=11, fontweight="bold", pad=7
        )

    # Prepare input MRI with boundary drawn for col 0
    input_with_boundary = np.stack(
        [img_np, img_np, img_np], axis=-1
    ).astype(np.float32)
    from scipy.ndimage import binary_dilation
    thick_boundary = binary_dilation(boundary, structure=np.ones((3, 3)))
    input_rgb = (input_with_boundary * 255).astype(np.uint8).copy()
    input_rgb[thick_boundary] = [255, 255, 0]   # yellow boundary on MRI

    for row_idx, mdata in enumerate(models_data):
        model_name = mdata["name"]
        heatmap    = mdata["heatmap"]
        # Row label includes alignment score
        s = scores[model_name]
        row_label = f"{model_name}\nabs={s['abs']:.3f} | ratio={s['ratio']:.2f}"
        axes[row_idx, 0].set_ylabel(
            row_label, fontsize=9, fontweight="bold",
            rotation=90, labelpad=8, va="center"
        )

        # Col 0 — Input MRI with GT boundary (yellow)
        axes[row_idx, 0].imshow(input_rgb)

        # Col 1 — GT mask with MYO highlighted
        axes[row_idx, 1].imshow(
            mask_np, cmap=SEG_CMAP, vmin=0, vmax=3, interpolation="nearest"
        )

        # Col 2 — Pure MYO heatmap (no boundary)
        rgb_plain = np.stack([img_np, img_np, img_np], axis=-1).astype(np.float32)
        rgb_plain = np.clip(rgb_plain, 0.0, 1.0)
        plain_overlay = show_cam_on_image(rgb_plain, heatmap, use_rgb=True)
        axes[row_idx, 2].imshow(plain_overlay)

        # Col 3 — MYO heatmap + GT boundary (white) ← KEY COLUMN
        boundary_overlay = overlay_boundary_on_heatmap(
            img_np   = img_np,
            heatmap  = heatmap,
            boundary = boundary,
            linewidth= 2,
        )
        axes[row_idx, 3].imshow(boundary_overlay)

        # Remove ticks from all panels
        for col_idx in range(n_cols):
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])

    # Legend
    legend_patches = [
        mpatches.Patch(color="black", label="Background"),
        mpatches.Patch(color="red",   label="RV"),
        mpatches.Patch(color="lime",  label="MYO"),
        mpatches.Patch(color="blue",  label="LV"),
        mpatches.Patch(color="white", label="GT MYO Boundary"),
        mpatches.Patch(color="yellow", label="GT MYO Boundary (col 0)"),
    ]
    fig.legend(
        handles         = legend_patches,
        loc             = "lower center",
        ncol            = 6,
        fontsize        = 8,
        frameon         = True,
        bbox_to_anchor  = (0.5, 0.01),
    )

    fig.suptitle(
        "Seg-Grad-CAM MYO Boundary Alignment  "
        "|  Ground Truth Boundary Overlay  "
        "|  Decoder Stage 3 Layer",
        fontsize=12, fontweight="bold", y=0.99,
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Saved: {save_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 62)
    print("  XAI Method 3 — Edge Map Overlay on Seg-Grad-CAM")
    print("  Claim: EnhancedBlock produces sharper boundary")
    print("         attention, supporting MYO HD95 improvement")
    print("=" * 62)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 1: Load the Method 1 slice ───────────────────────────────
    print("\n[1/5] Loading selected slice from Method 1...")
    img_np, mask_np, fname = load_selected_slice()

    # Normalise image to [0, 1] for display and heatmap overlay
    img_min, img_max = img_np.min(), img_np.max()
    img_display = (img_np - img_min) / (img_max - img_min + 1e-8)

    # Input tensor: (1, 1, H, W)
    input_tensor = (
        torch.from_numpy(img_np)
        .unsqueeze(0).unsqueeze(0)
        .float().to(DEVICE)
    )

    # ── Step 2: Extract GT MYO boundary ───────────────────────────────
    print("\n[2/5] Extracting GT MYO boundary...")
    boundary      = extract_myo_boundary(mask_np)
    boundary_band = dilate_boundary_for_score(boundary, BOUNDARY_RADIUS_PX)
    print(f"    Boundary band ({BOUNDARY_RADIUS_PX}px radius): "
          f"{int(boundary_band.sum())} pixels")

    # ── Step 3: Load all models ────────────────────────────────────────
    print("\n[3/5] Loading models...")
    models = {}
    for name, ckpt_path in CHECKPOINTS.items():
        models[name] = load_model(name, ckpt_path)

    # ── Step 4: Generate MYO Seg-Grad-CAM heatmaps ────────────────────
    print("\n[4/5] Generating MYO Seg-Grad-CAM heatmaps (Dec3)...")
    models_data = []
    scores      = {}

    for model_name, model in models.items():
        print(f"\n  Model: {model_name}")

        target_layer = get_dec3_layer(model_name, model)

        heatmap = generate_myo_seggradcam(
            model        = model,
            target_layer = target_layer,
            input_tensor = input_tensor,
        )

        # Boundary alignment score
        #score            = compute_alignment_score(heatmap, boundary_band)
        #scores[model_name] = score

        myo_mask                    = (mask_np == MYO_CLASS)
        abs_score, ratio            = compute_concentration_ratio(
                                  heatmap, boundary_band, myo_mask)
        scores[model_name]          = {"abs": abs_score, "ratio": ratio}

        print(f"    Absolute score  : {abs_score:.4f}")
        print(f"    Concentration   : {ratio:.4f}  "
              f"({'boundary-focused' if ratio >= 1.0 else 'interior-focused'})")

        # Predicted MYO pixel count for logging
        with torch.no_grad():
            pred = model(input_tensor).argmax(dim=1).squeeze(0).cpu().numpy()
        pred_myo = int(np.sum(pred == MYO_CLASS))

        print(f"    Heatmap range   : [{heatmap.min():.4f}, {heatmap.max():.4f}]")
        print(f"    Predicted MYO px: {pred_myo}")

        models_data.append({
            "name"    : model_name,
            "heatmap" : heatmap,
        })

    # ── Step 5: Compose figure and save scores ─────────────────────────
    print("\n[5/5] Composing figure...")

    compose_edge_map_figure(
        models_data = models_data,
        img_np      = img_display,
        mask_np     = mask_np,
        boundary    = boundary,
        scores      = scores,
        save_path   = os.path.join(OUTPUT_DIR, "edge_map_overlay.png"),
    )

    # Save alignment scores
    # Save alignment scores
    scores_path = os.path.join(OUTPUT_DIR, "edge_map_scores.txt")
    with open(scores_path, "w") as f:
        f.write("Boundary Alignment Scores — MYO Seg-Grad-CAM at Dec3\n")
        f.write(f"Boundary radius : {BOUNDARY_RADIUS_PX} pixels\n")
        f.write(f"Slice           : {fname}\n")
        f.write(f"MYO class index : {MYO_CLASS}\n\n")
        f.write(f"{'Model':<20}  {'AbsScore':>9}  {'Ratio':>7}  {'Interpretation'}\n")
        f.write("-" * 65 + "\n")
        for name, s in scores.items():
            boundary_focused = "boundary-focused" if s['ratio'] >= 1.0 else "interior-focused"
            f.write(f"{name:<20}  {s['abs']:>9.4f}  {s['ratio']:>7.4f}  {boundary_focused}\n")
        f.write("\nAbsScore = mean heatmap within boundary band\n")
        f.write("Ratio    = AbsScore / mean heatmap over full MYO region\n")
        f.write("Ratio > 1.0 → boundary-focused | Ratio < 1.0 → interior-focused\n")
    print(f"    Scores saved: {scores_path}")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  Edge Map Overlay complete.")
    print(f"\n  Boundary Alignment Scores:")
    for name, s in scores.items():
        print(f"    {name:<20}  abs={s['abs']:.4f}  ratio={s['ratio']:.4f}")
    print(f"\n  Figure : {OUTPUT_DIR}/edge_map_overlay.png")
    print(f"  Scores : {OUTPUT_DIR}/edge_map_scores.txt")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
