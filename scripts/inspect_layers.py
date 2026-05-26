# scripts/inspect_layers.py
"""
Layer inspection script for XAI Grad-CAM target layer identification.
Prints named modules for U-Net Baseline, LWEU-Net Base, and LWEU-Net V2.
Run: python scripts/inspect_layers.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

# ── Model imports ─────────────────────────────────────────────────────────────
from src.models.unet_baseline import UNetBaseline
from src.models.lweunet.lweunet import LWEUNet
from src.models.lweunet.lweunet_v2 import LWEUNetV2

# ── Checkpoint paths ──────────────────────────────────────────────────────────
CHECKPOINTS = {
    "UNet Baseline" : "checkpoints/unet_baseline/best_model.pth",
    "LWEU-Net V2"   : "checkpoints/lweunet_v2/best_model.pth",
    "LWEU-Net Base" : "checkpoints/lweunet_base/best_model.pth"
}

# ── Model builders ────────────────────────────────────────────────────────────
def build_model(name):
    if name == "UNet Baseline":
        return UNetBaseline(in_channels=1, num_classes=4)
    elif name == "LWEU-Net V2":
        return LWEUNetV2(in_channels=1, num_classes=4)
    elif name == "LWEU-Net Base":                                   # ← add this
        return LWEUNet(in_channels=1, num_classes=4, use_eca=False)              # ← add this

# ── Load checkpoint ───────────────────────────────────────────────────────────
def load_model(name, path):
    model = build_model(name)
    ckpt = torch.load(path, map_location="cpu")
    # handle both raw state_dict and wrapped checkpoint
    state = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state)
    model.eval()
    print(f"\n✅  Loaded: {name}  ({path})")
    return model

# ── Print named modules ───────────────────────────────────────────────────────
def inspect(name, model):
    print(f"\n{'='*60}")
    print(f"  MODEL: {name}")
    print(f"{'='*60}")
    for module_name, module in model.named_modules():
        if module_name == "":
            continue  # skip root
        indent = "  " * module_name.count(".")
        print(f"{indent}{module_name}  →  {type(module).__name__}")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for model_name, ckpt_path in CHECKPOINTS.items():
        if not os.path.exists(ckpt_path):
            print(f"\n⚠️  Checkpoint not found, skipping: {ckpt_path}")
            continue
        model = load_model(model_name, ckpt_path)
        inspect(model_name, model)

    print("\n\nDONE — copy the bottleneck and decoder layer names from above.")
    print("We are looking for:")
    print("  1. The deepest encoder/bottleneck block")
    print("  2. The third decoder block (Dec3 / decoder3 / up3)")