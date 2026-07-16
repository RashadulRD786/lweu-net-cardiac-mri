"""
Figure 5.6 v3 — Filled Mask Style with Cardiac-Region Cropping
================================================================
Each row is cropped to the cardiac bounding box (from the GT mask),
so all 5 panels per row show the same spatial region at the same scale.
This eliminates floating masks and ensures clean column alignment.

Layout  : 3 rows × 5 columns
Rows    : ED Representative | ED Challenging | ES Challenging
Columns : Original MRI | Ground Truth | Baseline U-Net | LiteU-Net | EnhU-Net

HOW TO RUN:
    cd /home/rashadulnafisriyad/FYP_UNet/LWEU_NET
    conda activate lweunet
    python generate_fig56_v3.py

OUTPUT  : results/figures/figure5_6_final.png
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from pathlib import Path

PROJECT_ROOT = Path('/home/rashadulnafisriyad/FYP_UNet/LWEU_NET')
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / 'data' / 'preprocessed' / 'test'

CKPT_BASELINE = PROJECT_ROOT / 'checkpoints' / 'unet_baseline'  / 'best_model.pth'
CKPT_LITE     = PROJECT_ROOT / 'checkpoints' / 'lweunet_base'   / 'best_model.pth'
CKPT_ENH      = PROJECT_ROOT / 'checkpoints' / 'lweunet_v2'     / 'best_model.pth'

OUTPUT_PATH = PROJECT_ROOT / 'results' / 'figures' / 'figure5_6_final.png'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Slices ─────────────────────────────────────────────────────────────────────
ROWS = [
    ('patient138', 'ED', 4, 'ED — Representative\n(patient138, slice 04)'),
    ('patient112', 'ED', 0, 'ED — Challenging\n(patient112, slice 00)'),
    ('patient144', 'ES', 6, 'ES — Challenging\n(patient144, slice 06)'),
]

# ── Colour map: label → RGB ────────────────────────────────────────────────────
LABEL_RGB = {
    0: [15,  15,  15 ],   # Background — near-black
    1: [0,   187, 221],   # RV         — cyan
    2: [255, 224, 51 ],   # MYO        — yellow
    3: [221, 34,  34 ],   # LV         — red
}

COL_TITLES = [
    'Original MRI',
    'Ground Truth',
    'Baseline U-Net',
    'LiteU-Net\n(Lightweight Baseline)',
    'EnhU-Net\n(Proposed Model)',
]

# ══════════════════════════════════════════════════════════════════════════════
# CROP HELPERS
# ══════════════════════════════════════════════════════════════════════════════

# def get_square_crop(mask: np.ndarray, padding: int = 28):
#     """
#     Return (r0, r1, c0, c1) — a SQUARE bounding box around all
#     non-background labels in `mask`, with `padding` pixels on each side.
#     """
#     fg = mask > 0
#     if not fg.any():
#         H, W = mask.shape
#         return 0, H, 0, W

#     rows = np.where(fg.any(axis=1))[0]
#     cols = np.where(fg.any(axis=0))[0]
#     r0, r1 = rows[0],  rows[-1]
#     c0, c1 = cols[0],  cols[-1]

#     # Add padding
#     H, W = mask.shape
#     r0 = max(0, r0 - padding)
#     r1 = min(H, r1 + padding)
#     c0 = max(0, c0 - padding)
#     c1 = min(W, c1 + padding)

#     # Force square (extend the shorter side equally)
#     h, w = r1 - r0, c1 - c0
#     if h < w:
#         diff = w - h
#         r0 = max(0, r0 - diff // 2)
#         r1 = min(H, r0 + w)
#     elif w < h:
#         diff = h - w
#         c0 = max(0, c0 - diff // 2)
#         c1 = min(W, c0 + h)

#     return r0, r1, c0, c1


# def crop(arr: np.ndarray, r0, r1, c0, c1):
#     return arr[r0:r1, c0:c1]

def get_square_crop(mask: np.ndarray, padding: int = 40, min_size: int = 160):
    """
    Return a square crop around all non-background labels.
    Enforces a minimum crop size of min_size pixels to prevent
    pixelation when cardiac structures are very small (apical slices).
    """
    H, W = mask.shape
    fg = mask > 0

    if not fg.any():
        # No structures — return full image
        return 0, H, 0, W

    rows = np.where(fg.any(axis=1))[0]
    cols = np.where(fg.any(axis=0))[0]
    r0, r1 = int(rows[0]),  int(rows[-1])
    c0, c1 = int(cols[0]),  int(cols[-1])

    # Structure centroid
    ctr_r = (r0 + r1) // 2
    ctr_c = (c0 + c1) // 2

    # Half-size = max of (structure extent + padding) or (min_size / 2)
    struct_h = r1 - r0
    struct_w = c1 - c0
    half = max(
        (max(struct_h, struct_w) // 2) + padding,
        min_size // 2
    )

    # Apply centred square crop
    r0_new = max(0, ctr_r - half)
    r1_new = min(H, ctr_r + half)
    c0_new = max(0, ctr_c - half)
    c1_new = min(W, ctr_c + half)

    return r0_new, r1_new, c0_new, c1_new

def crop(arr: np.ndarray, r0, r1, c0, c1):
    return arr[r0:r1, c0:c1]


# def mask_to_rgb(mask: np.ndarray) -> np.ndarray:


# ══════════════════════════════════════════════════════════════════════════════
# COLOUR CONVERSION
# ══════════════════════════════════════════════════════════════════════════════

def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for lbl, colour in LABEL_RGB.items():
        rgb[mask == lbl] = colour
    return rgb


# ══════════════════════════════════════════════════════════════════════════════
# DATA / MODEL
# ══════════════════════════════════════════════════════════════════════════════

def load_slice(patient, phase, slice_num):
    stem = f'{patient}_{phase}_slice{slice_num:02d}'
    img  = np.load(DATA_DIR / f'{stem}_img.npy').astype(np.float32)
    mask = np.load(DATA_DIR / f'{stem}_msk.npy').astype(np.int32)
    if img.ndim  == 3: img  = img[0]
    if mask.ndim == 3: mask = mask[0]
    return img, mask


def load_model(ckpt_path, model_type):
    if model_type == 'baseline':
        from src.models.unet_baseline import UNetBaseline
        model = UNetBaseline(in_channels=1, num_classes=4,
                             base_filters=64, dropout_p=0.5)
    elif model_type == 'lite':
        from src.models.lweunet.lweunet import LWEUNet
        model = LWEUNet(in_channels=1, num_classes=4,
                        base_filters=32, dropout_p=0.5, use_eca=False)
    elif model_type == 'enhanced':
        from src.models.lweunet.lweunet_v2 import LWEUNetV2
        model = LWEUNetV2(in_channels=1, num_classes=4,
                          base_filters=32, dropout_p=0.5)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    key  = 'model_state' if 'model_state' in ckpt else 'model_state_dict'
    if key in ckpt:
        model.load_state_dict(ckpt[key])
        print(f'  [{model_type:10s}] epoch={ckpt.get("epoch","?")}  '
              f'val_dice={ckpt.get("val_mean_dice",0.0):.4f}')
    else:
        model.load_state_dict(ckpt)
    return model.to(DEVICE).eval()


@torch.no_grad()
def predict(model, img):
    x = torch.tensor(img).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    return model(x).argmax(1).squeeze(0).cpu().numpy().astype(np.int32)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════

def build_figure(models_list, rows_data):
    n_rows  = len(rows_data)
    n_cols  = 5
    cell_sz = 3.2          # inches — each panel cell (square)
    left_w  = 1.9          # width of row-label gutter

    fig_w = left_w + n_cols * cell_sz
    fig_h = 0.55 + n_rows * cell_sz + 0.75   # header + cells + legend

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        facecolor='black',
    )
    fig.patch.set_facecolor('black')

    # Leave room on left for row labels and bottom for legend
    fig.subplots_adjust(
        left   = left_w / fig_w,
        right  = 0.995,
        top    = 1.0 - (0.50 / fig_h),
        bottom = 0.75 / fig_h,
        hspace = 0.04,
        wspace = 0.04,
    )

    # ── Column headers ─────────────────────────────────────────────────────────
    for c, title in enumerate(COL_TITLES):
        # x in figure coordinates: left_w + (c+0.5)*cell_sz, normalised by fig_w
        x = (left_w + (c + 0.5) * cell_sz) / fig_w
        y = 1.0 - (0.06 / fig_h)
        fig.text(x, y, title,
                 ha='center', va='top',
                 fontsize=11, fontweight='bold', color='white',
                 multialignment='center')

    # ── Rows ───────────────────────────────────────────────────────────────────
    for row_idx, (row_label, img, gt_mask) in enumerate(rows_data):

        # Compute square crop from GT mask
        r0, r1, c0, c1 = get_square_crop(gt_mask, padding=55)

        # Run predictions
        preds = [predict(m, img) for m in models_list]

        # Build list of (panel_type, data) for the 5 columns
        panels = [
            ('mri',  img),
            ('mask', gt_mask),
            ('mask', preds[0]),   # Baseline
            ('mask', preds[1]),   # LiteU-Net
            ('mask', preds[2]),   # EnhU-Net
        ]

        # ── Row label (left of grid) ──────────────────────────────────────────
        row_centre_y = 1.0 - (0.50 / fig_h) \
                       - (row_idx + 0.5) * cell_sz / fig_h
        fig.text(
            (left_w - 0.12) / fig_w, row_centre_y,
            row_label,
            ha='right', va='center',
            fontsize=9.5, fontweight='bold', color='white',
            multialignment='center',
        )

        # ── Panels ────────────────────────────────────────────────────────────
        for col_idx, (ptype, data) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor('black')
            ax.axis('off')

            # Crop to cardiac region
            d = crop(data, r0, r1, c0, c1)

            if ptype == 'mri':
                ax.imshow(d, cmap='gray', aspect='equal',
                          vmin=np.percentile(d, 1),
                          vmax=np.percentile(d, 99))
            else:
                ax.imshow(mask_to_rgb(d), aspect='equal',
                          interpolation='nearest')

            # Thin separator border
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('#333333')
                spine.set_linewidth(0.6)

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(color=tuple(c/255 for c in LABEL_RGB[3]),
                       label='LV — Left Ventricle'),
        mpatches.Patch(color=tuple(c/255 for c in LABEL_RGB[2]),
                       label='MYO — Myocardium'),
        mpatches.Patch(color=tuple(c/255 for c in LABEL_RGB[1]),
                       label='RV — Right Ventricle'),
        mpatches.Patch(facecolor='#0F0F0F', edgecolor='white',
                       linewidth=0.5, label='Background'),
    ]
    fig.legend(handles=handles,
               loc='lower center',
               bbox_to_anchor=(0.55, 0.00),
               ncol=4, fontsize=10.5,
               framealpha=0.12, edgecolor='#555555',
               labelcolor='white', facecolor='black')

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print(f'Device : {DEVICE}\n')

    print('Loading models...')
    models_list = [
        load_model(CKPT_BASELINE, 'baseline'),
        load_model(CKPT_LITE,     'lite'),
        load_model(CKPT_ENH,      'enhanced'),
    ]
    print()

    print('Loading slices...')
    rows_data = []
    for patient, phase, slice_num, row_label in ROWS:
        img, gt = load_slice(patient, phase, slice_num)
        rows_data.append((row_label, img, gt))
        r0, r1, c0, c1 = get_square_crop(gt, padding=55)
        print(f'  {patient}_{phase}_slice{slice_num:02d}  '
              f'crop=({r0}:{r1}, {c0}:{c1})  '
              f'size={r1-r0}×{c1-c0}px  '
              f'structures={np.unique(gt[gt>0]).tolist()}')
    print()

    print('Building figure...')
    fig = build_figure(models_list, rows_data)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close(fig)
    print(f'\nSaved → {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
