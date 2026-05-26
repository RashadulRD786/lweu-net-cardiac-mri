"""
XAI Method 2 — Direct Context Gate Visualisation
=================================================
Extracts and visualises the GlobalContextGate sigmoid weights from
LWEU-Net V2 across three slice positions (basal, mid-ventricular, apical)
for the same patient to validate Claim 2:

    "Global context gate provides whole-image anatomical shape awareness"

The GlobalContextGate is an SE-style module that outputs a weight in [0,1]
for each feature channel, conditioned on the global average of the entire
slice. If the gate learned anatomy-aware recalibration, channel weights must
shift systematically based on which structures are present in the slice —
not just local pixel patterns.

Expected finding:
    Mid-ventricular → RV-relevant channels have HIGH weights  (RV present)
    Apical          → same channels have LOW weights           (RV absent)

Gate locations visualised:
    Primary   — bottleneck.block.context_gate  (most abstract representation)
    Secondary — encoder.level4.block.context_gate  (early context awareness)

Output figures:
    figures/xai/context_gate_panel.png       3×3 grid: MRI | GT | bar chart
    figures/xai/context_gate_comparison.png  Overlay + difference map
    figures/xai/context_gate_weights.npz     Raw weights for reproducibility

Usage:
    python scripts/xai_context_gate.py

Citations:
    Hu et al. (2018) CVPR          — SE-Net: architectural source of gate design
    Gipiškis et al. (2024) ICT     — Architecture-based XAI as legitimate category
    Sun et al. (2020) MICCAI       — SAUNet: precedent for cardiac attention XAI
"""

import os
import sys
import glob
import re
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.lweunet.lweunet_v2 import LWEUNetV2


# =============================================================================
# CONFIGURATION
# All paths and constants defined here — edit only this section if needed
# =============================================================================

CHECKPOINT_V2 = "checkpoints/lweunet_v2/best_model.pth"
TEST_DATA_DIR = "data/preprocessed/test"
OUTPUT_DIR    = "figures/xai"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Gate locations to extract — keys used as identifiers throughout the script
# path: lambda that returns the Sigmoid module to hook given the model
# label: display name for figures
GATE_LOCATIONS = {
    "bottleneck" : {
        "path"  : lambda m: m.bottleneck.block.context_gate.gate[5],
        "label" : "Bottleneck Gate",
    },
    "enc4" : {
        "path"  : lambda m: m.encoder.level4.block.context_gate.gate[5],
        "label" : "Encoder Level 4 Gate",
    },
}

# Slice position labels and their plot colours
SLICE_LABELS = ["Basal", "Mid-Ventricular", "Apical"]
SLICE_COLORS = ["#E74C3C", "#2ECC71", "#3498DB"]   # red, green, blue

# Minimum pixels per structure to qualify as a mid-ventricular slice
MIN_PIXELS = 100

# 4-class segmentation colormap: BG=black, RV=red, MYO=lime, LV=blue
SEG_CMAP = ListedColormap(["black", "red", "lime", "blue"])


# =============================================================================
# SECTION 1 — MODEL LOADING
# =============================================================================

def load_v2_model(ckpt_path: str) -> torch.nn.Module:
    """
    Load LWEU-Net V2 from checkpoint.
    Supports both checkpoint formats used across the project.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = LWEUNetV2(in_channels=1, num_classes=4)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state)
    model.eval()
    model.to(DEVICE)
    print(f"    Loaded: LWEU-Net V2  ({ckpt_path})")
    return model


# =============================================================================
# SECTION 2 — SLICE SELECTION
# =============================================================================

def parse_filename(fname: str):
    """
    Parse patient ID, phase, and slice index from test filename.
    Expected format: patient138_ED_slice04_img.npy
    Returns: (patient_id, phase, slice_idx) or (None, None, None) on failure
    """
    match = re.match(r"(patient\d+)_(\w+)_slice(\d+)_img\.npy", fname)
    if match:
        return match.group(1), match.group(2), int(match.group(3))
    return None, None, None


def group_slices_by_patient(test_dir: str) -> dict:
    """
    Group all test slices by (patient_id, phase).
    Slices within each group are sorted by slice index.

    Returns
    -------
    dict : {(patient_id, phase): [(slice_idx, img_path, mask_path), ...]}
    """
    groups    = defaultdict(list)
    img_files = sorted(glob.glob(os.path.join(test_dir, "*_img.npy")))

    for img_path in img_files:
        fname      = os.path.basename(img_path)
        patient_id, phase, slice_idx = parse_filename(fname)
        if patient_id is None:
            continue
        mask_path = img_path.replace("_img.npy", "_msk.npy")
        if os.path.exists(mask_path):
            groups[(patient_id, phase)].append((slice_idx, img_path, mask_path))

    for key in groups:
        groups[key].sort(key=lambda x: x[0])

    print(f"    Found {len(groups)} patient-phase groups in {test_dir}")
    return groups


def select_patient(groups: dict, min_slices: int = 4):
    """
    Select a patient-phase with sufficient slices and at least one
    mid-ventricular slice that contains all three cardiac structures.

    Prefers the patient with the most slices — maximum contrast between
    basal, mid-ventricular, and apical positions.

    Returns
    -------
    (patient_key, sorted_slices) : tuple
    """
    candidates = []

    for (patient_id, phase), slices in groups.items():
        if len(slices) < min_slices:
            continue

        # Check at least one slice has all three structures
        for _, _, mask_path in slices:
            mask = np.load(mask_path)
            if (np.sum(mask == 1) >= MIN_PIXELS and
                np.sum(mask == 2) >= MIN_PIXELS and
                np.sum(mask == 3) >= MIN_PIXELS):
                candidates.append(((patient_id, phase), slices))
                break

    if not candidates:
        raise RuntimeError(
            "No patient found with >= min_slices and all three structures. "
            f"Reduce MIN_PIXELS (currently {MIN_PIXELS}) or min_slices."
        )

    # Prefer patient with most slices for clearest anatomical progression
    candidates.sort(key=lambda x: len(x[1]), reverse=True)
    key, slices = candidates[0]
    print(f"    Selected patient : {key[0]}  Phase: {key[1]}  "
          f"Total slices: {len(slices)}")
    return key, slices


def select_three_slices(slices: list) -> list:
    """
    From a sorted patient slice list, select three representative positions:
      - Basal         : first slice (index 0)
      - Mid-Ventricular: most balanced middle slice (all structures present)
      - Apical        : last slice

    Returns
    -------
    list of tuples: [(label, slice_idx, img_path, mask_path), ...]
    """
    # Basal: first slice
    basal  = slices[0]

    # Apical: last slice
    apical = slices[-1]

    # Mid-ventricular: balanced slice excluding first and last
    mid_candidates = []
    for slice_idx, img_path, mask_path in slices[1:-1]:
        mask   = np.load(mask_path)
        rv_px  = int(np.sum(mask == 1))
        myo_px = int(np.sum(mask == 2))
        lv_px  = int(np.sum(mask == 3))

        if (rv_px >= MIN_PIXELS and
            myo_px >= MIN_PIXELS and
            lv_px  >= MIN_PIXELS):
            counts = np.array([rv_px, myo_px, lv_px], dtype=float)
            cv     = counts.std() / counts.mean()
            mid_candidates.append((cv, slice_idx, img_path, mask_path))

    if not mid_candidates:
        # Fallback: use the index-middle slice
        mid_entry = slices[len(slices) // 2]
        mid_candidates = [(0.0, mid_entry[0], mid_entry[1], mid_entry[2])]

    mid_candidates.sort(key=lambda x: x[0])
    _, mid_idx, mid_img, mid_mask = mid_candidates[0]

    selected = [
        ("Basal",            basal[0],  basal[1],  basal[2]),
        ("Mid-Ventricular",  mid_idx,   mid_img,   mid_mask),
        ("Apical",           apical[0], apical[1], apical[2]),
    ]

    print(f"\n    {'Position':18s}  {'SliceIdx':>8}  {'RV':>6}  {'MYO':>6}  {'LV':>6}")
    print(f"    {'-'*52}")
    for label, sidx, _, mpath in selected:
        mask   = np.load(mpath)
        rv_px  = int(np.sum(mask == 1))
        myo_px = int(np.sum(mask == 2))
        lv_px  = int(np.sum(mask == 3))
        print(f"    {label:18s}  {sidx:>8}  {rv_px:>6}  {myo_px:>6}  {lv_px:>6}")

    return selected


# =============================================================================
# SECTION 3 — GATE WEIGHT EXTRACTION
# =============================================================================

def extract_gate_weights(
    model        : torch.nn.Module,
    input_tensor : torch.Tensor,
    gate_locations: dict,
) -> dict:
    """
    Run a forward pass and capture gate weights from all specified locations
    using PyTorch forward hooks registered on the Sigmoid layer of each gate.

    The Sigmoid layer is the final layer in the GlobalContextGate.gate
    Sequential — its output is the raw channel weights vector in [0, 1].

    Parameters
    ----------
    model          : LWEUNetV2 in eval mode
    input_tensor   : (1, 1, H, W) float tensor on DEVICE
    gate_locations : dict from GATE_LOCATIONS config

    Returns
    -------
    dict : {gate_name: (C,) float32 numpy array}
    """
    captured = {}
    hooks    = []

    for gate_name, gate_cfg in gate_locations.items():
        sigmoid_layer = gate_cfg["path"](model)

        def make_hook(name):
            def hook_fn(module, inp, output):
                # output: (B, C) — remove batch dimension
                captured[name] = output.detach().cpu().squeeze(0).numpy()
            return hook_fn

        hooks.append(
            sigmoid_layer.register_forward_hook(make_hook(gate_name))
        )

    with torch.no_grad():
        _ = model(input_tensor)

    for hook in hooks:
        hook.remove()

    return captured


def build_input_tensor(img_path: str) -> torch.Tensor:
    """Load image, build (1, 1, H, W) input tensor."""
    img_np = np.load(img_path).astype(np.float32)
    return (
        torch.from_numpy(img_np)
        .unsqueeze(0)
        .unsqueeze(0)
        .float()
        .to(DEVICE)
    )


# =============================================================================
# SECTION 4 — FIGURE A: PANEL (MRI + GT + gate bar chart per row)
# =============================================================================

def draw_bar_chart(ax, weights, color, subtitle):
    """
    Draw gate weights as a bar chart on the given axes.

    Parameters
    ----------
    ax       : matplotlib Axes
    weights  : (C,) float32 array in [0, 1]
    color    : bar fill colour
    subtitle : axes subtitle text
    """
    C = len(weights)
    ax.bar(
        np.arange(C), weights,
        color     = color,
        alpha     = 0.75,
        width     = 1.0,
        edgecolor = "none",
    )
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
               label="Threshold 0.5")
    ax.set_xlim(0, C)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Channel Index", fontsize=8)
    ax.set_ylabel("Gate Weight [0–1]", fontsize=8)
    ax.set_title(subtitle, fontsize=9, fontweight="bold", pad=4)
    ax.tick_params(labelsize=7)

    # Annotate mean weight
    ax.axhline(y=weights.mean(), color=color, linestyle=":",
               linewidth=1.2, alpha=0.9)
    ax.text(
        C * 0.98, weights.mean() + 0.03,
        f"μ={weights.mean():.2f}",
        ha="right", va="bottom", fontsize=7, color=color, fontweight="bold"
    )


def compose_panel_figure(selected_slices: list, all_weights: dict,
                         save_path: str) -> None:
    """
    Compose the 3-row × (2 + N_gates) panel figure.

    Layout
    ------
    Columns : Input MRI | Ground Truth | [gate bar chart per gate location]
    Rows    : Basal | Mid-Ventricular | Apical

    Parameters
    ----------
    selected_slices : [(label, slice_idx, img_path, mask_path), ...]
    all_weights     : {label: {gate_name: (C,) array}}
    save_path       : output file path
    """
    n_gates = len(GATE_LOCATIONS)
    n_rows  = 3
    n_cols  = 2 + n_gates

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize     = (5.5 * n_cols, 4.2 * n_rows),
        gridspec_kw = {"wspace": 0.38, "hspace": 0.28},
    )

    # Column headers on top row only
    col_headers = ["Input MRI", "Ground Truth"] + [
        cfg["label"] for cfg in GATE_LOCATIONS.values()
    ]
    for col_idx, header in enumerate(col_headers):
        axes[0, col_idx].set_title(
            header, fontsize=11, fontweight="bold", pad=7
        )

    for row_idx, (label, _, img_path, mask_path) in enumerate(selected_slices):
        img_np      = np.load(img_path).astype(np.float32)
        mask_np     = np.load(mask_path).astype(np.int64)
        img_display = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        color       = SLICE_COLORS[row_idx]

        # Row label
        axes[row_idx, 0].set_ylabel(
            label, fontsize=10, fontweight="bold",
            rotation=90, labelpad=8, va="center"
        )

        # Col 0 — Input MRI
        axes[row_idx, 0].imshow(img_display, cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])

        # Col 1 — Ground Truth mask
        axes[row_idx, 1].imshow(
            mask_np, cmap=SEG_CMAP, vmin=0, vmax=3, interpolation="nearest"
        )
        axes[row_idx, 1].set_xticks([])
        axes[row_idx, 1].set_yticks([])

        # Cols 2+ — Gate weight bar charts
        for gate_col, (gate_name, gate_cfg) in enumerate(GATE_LOCATIONS.items()):
            weights = all_weights[label][gate_name]
            draw_bar_chart(
                ax       = axes[row_idx, 2 + gate_col],
                weights  = weights,
                color    = color,
                subtitle = f"{label}  |  {gate_cfg['label']}",
            )

    # Segmentation legend
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
        bbox_to_anchor  = (0.5, 0.0),
    )

    fig.suptitle(
        "Context Gate Weights — LWEU-Net V2 GlobalContextGate  "
        "|  Basal / Mid-Ventricular / Apical",
        fontsize=13, fontweight="bold", y=1.01
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Saved: {save_path}")


# =============================================================================
# SECTION 5 — FIGURE B: COMPARISON OVERLAY + DIFFERENCE MAP
# =============================================================================

def compose_comparison_figure(selected_slices: list, all_weights: dict,
                               save_path: str) -> None:
    """
    Compose the overlay comparison figure showing weight shifts across positions.

    Layout: N_gates rows × 2 columns
      Col 0 — Overlay line plot: basal / mid-ventricular / apical on same axes
      Col 1 — Difference map:   mid-ventricular minus apical
                                 Green bars = channels MORE active at mid (RV-sensitive)
                                 Orange bars = channels MORE active at apical

    The top-5 most RV-sensitive channels are annotated on the difference map.

    Parameters
    ----------
    selected_slices : [(label, slice_idx, img_path, mask_path), ...]
    all_weights     : {label: {gate_name: (C,) array}}
    save_path       : output file path
    """
    n_gates = len(GATE_LOCATIONS)
    fig, axes = plt.subplots(
        n_gates, 2,
        figsize     = (16, 5.5 * n_gates),
        gridspec_kw = {"wspace": 0.32, "hspace": 0.45},
    )

    # Ensure axes is always 2D even when n_gates == 1
    if n_gates == 1:
        axes = axes[np.newaxis, :]

    labels = [s[0] for s in selected_slices]   # ["Basal", "Mid-Ventricular", "Apical"]

    for gate_row, (gate_name, gate_cfg) in enumerate(GATE_LOCATIONS.items()):
        gate_label    = gate_cfg["label"]
        weights_store = {}

        # ── Col 0: Overlay line plot ──────────────────────────────────
        ax_overlay = axes[gate_row, 0]

        for row_idx, label in enumerate(labels):
            weights = all_weights[label][gate_name]
            C       = len(weights)
            weights_store[label] = weights

            ax_overlay.plot(
                np.arange(C),
                weights,
                color     = SLICE_COLORS[row_idx],
                linewidth = 1.0,
                alpha     = 0.85,
                label     = label,
            )

        ax_overlay.axhline(
            y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5
        )
        ax_overlay.set_xlim(0, C)
        ax_overlay.set_ylim(0, 1.08)
        ax_overlay.set_xlabel("Channel Index", fontsize=9)
        ax_overlay.set_ylabel("Gate Weight [0–1]", fontsize=9)
        ax_overlay.set_title(
            f"{gate_label}  —  Weight Profile Across Slice Positions",
            fontsize=10, fontweight="bold"
        )
        ax_overlay.legend(fontsize=9, loc="upper right", framealpha=0.9)
        ax_overlay.tick_params(labelsize=8)

        # ── Col 1: Difference map (mid-ventricular minus apical) ──────
        # Positive (green): channel more active at mid-vent = RV-sensitive
        # Negative (orange): channel more active at apical
        ax_diff = axes[gate_row, 1]
        diff    = weights_store["Mid-Ventricular"] - weights_store["Apical"]
        C       = len(diff)

        bar_colors = ["#27AE60" if d > 0 else "#E67E22" for d in diff]
        ax_diff.bar(
            np.arange(C), diff,
            color     = bar_colors,
            width     = 1.0,
            edgecolor = "none",
            alpha     = 0.85,
        )
        ax_diff.axhline(y=0, color="black", linewidth=0.9)
        ax_diff.set_xlim(0, C)
        ax_diff.set_xlabel("Channel Index", fontsize=9)
        ax_diff.set_ylabel("Weight Difference  (Mid − Apical)", fontsize=9)
        ax_diff.set_title(
            f"{gate_label}  —  RV-Sensitive Channels  (Mid − Apical)",
            fontsize=10, fontweight="bold"
        )
        ax_diff.tick_params(labelsize=8)

        # Annotate top-5 most RV-sensitive channels (largest positive diff)
        top5_idx = np.argsort(diff)[-5:][::-1]
        for ch_idx in top5_idx:
            if diff[ch_idx] > 0:
                ax_diff.annotate(
                    f"ch{ch_idx}",
                    xy        = (ch_idx, diff[ch_idx]),
                    xytext    = (ch_idx, diff[ch_idx] + 0.03),
                    fontsize  = 7,
                    ha        = "center",
                    color     = "#1A5276",
                    fontweight= "bold",
                )

        # Horizontal reference lines for strong suppression / activation
        ax_diff.axhline(y= 0.1, color="#27AE60", linestyle=":",
                        linewidth=0.7, alpha=0.6)
        ax_diff.axhline(y=-0.1, color="#E67E22", linestyle=":",
                        linewidth=0.7, alpha=0.6)

    # Legend for difference map colours
    diff_legend = [
        mpatches.Patch(color="#27AE60",
                       label="Mid > Apical  (RV-sensitive — higher weight when RV present)"),
        mpatches.Patch(color="#E67E22",
                       label="Apical > Mid  (suppressed when RV present)"),
    ]
    fig.legend(
        handles         = diff_legend,
        loc             = "lower center",
        ncol            = 2,
        fontsize        = 9,
        frameon         = True,
        bbox_to_anchor  = (0.5, -0.02),
    )

    fig.suptitle(
        "Context Gate Analysis — LWEU-Net V2  "
        "|  Channel Weight Shift Across Slice Positions",
        fontsize=13, fontweight="bold", y=1.02
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
    print("  XAI Method 2 — Direct Context Gate Visualisation")
    print("  Claim: GlobalContextGate performs anatomy-aware")
    print("         channel recalibration across slice positions")
    print("=" * 62)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 1: Load LWEU-Net V2 ──────────────────────────────────────
    print("\n[1/4] Loading model...")
    model = load_v2_model(CHECKPOINT_V2)

    # ── Step 2: Select patient and three representative slices ────────
    print("\n[2/4] Selecting patient slices...")
    groups              = group_slices_by_patient(TEST_DATA_DIR)
    patient_key, slices = select_patient(groups, min_slices=4)
    selected_slices     = select_three_slices(slices)

    # ── Step 3: Extract gate weights for each slice position ─────────
    print("\n[3/4] Extracting context gate weights...")
    all_weights = {}   # {slice_label: {gate_name: (C,) array}}

    for label, slice_idx, img_path, mask_path in selected_slices:
        input_tensor        = build_input_tensor(img_path)
        weights             = extract_gate_weights(model, input_tensor, GATE_LOCATIONS)
        all_weights[label]  = weights

        for gate_name, w in weights.items():
            gate_label = GATE_LOCATIONS[gate_name]["label"]
            print(f"    [{label:18s}]  {gate_label}: "
                  f"{len(w):4d} channels  "
                  f"mean={w.mean():.3f}  "
                  f"min={w.min():.3f}  "
                  f"max={w.max():.3f}")

    # Save raw weights for reproducibility
    weights_path = os.path.join(OUTPUT_DIR, "context_gate_weights.npz")
    save_dict    = {}
    for label, gate_dict in all_weights.items():
        for gate_name, w in gate_dict.items():
            key = f"{label.replace(' ', '_')}_{gate_name}"
            save_dict[key] = w
    np.savez(weights_path, **save_dict)
    print(f"\n    Raw weights saved: {weights_path}")

    # ── Step 4: Compose and save figures ─────────────────────────────
    print("\n[4/4] Composing figures...")

    compose_panel_figure(
        selected_slices = selected_slices,
        all_weights     = all_weights,
        save_path       = os.path.join(OUTPUT_DIR, "context_gate_panel.png"),
    )

    compose_comparison_figure(
        selected_slices = selected_slices,
        all_weights     = all_weights,
        save_path       = os.path.join(OUTPUT_DIR, "context_gate_comparison.png"),
    )

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  Context Gate Visualisation complete.")
    print(f"  Panel figure   : {OUTPUT_DIR}/context_gate_panel.png")
    print(f"  Comparison fig : {OUTPUT_DIR}/context_gate_comparison.png")
    print(f"  Raw weights    : {OUTPUT_DIR}/context_gate_weights.npz")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
