"""
Statistical Aggregation — LWEU-Net V2 Multi-Seed Runs
======================================================
Loads per-seed evaluation JSON files, extracts ED+ES combined metrics,
computes mean ± std across all seeds, and saves a human-readable summary.

Expected input layout:
    logs/lweunet_v2_seed42/eval_results.json
    logs/lweunet_v2_seed123/eval_results.json
    logs/lweunet_v2_seed456/eval_results.json
    logs/lweunet_v2_seed789/eval_results.json
    logs/lweunet_v2_seed1024/eval_results.json

Each JSON must contain an "ED+ES" key (produced by evaluate_phase.py --phase ALL).

Outputs:
    results/statistical_runs/all_seeds_raw.json     ← per-seed numbers
    results/statistical_runs/summary_mean_std.json  ← mean ± std
    results/statistical_runs/summary_table.txt      ← human-readable table

Usage:
    python scripts/aggregate_stats.py
    python scripts/aggregate_stats.py --seeds 42 123 456  # subset
    python scripts/aggregate_stats.py --log_base logs/lweunet_v2_seed \
                                      --out_dir results/statistical_runs
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Metrics extracted from the ED+ES block of each eval JSON
METRICS = [
    ("mean_dice", "Mean Dice",  ".4f", ""),
    ("dice_lv",   "LV Dice",   ".4f", ""),
    ("dice_rv",   "RV Dice",   ".4f", ""),
    ("dice_myo",  "MYO Dice",  ".4f", ""),
    ("mean_hd95", "Mean HD95", ".2f", "mm"),
    ("hd95_lv",   "LV HD95",   ".2f", "mm"),
    ("hd95_rv",   "RV HD95",   ".2f", "mm"),
    ("hd95_myo",  "MYO HD95",  ".2f", "mm"),
]

DEFAULT_SEEDS   = [42, 123, 456, 789, 1024]
DEFAULT_LOG_BASE = "logs/lweunet_v2_seed"
DEFAULT_OUT_DIR  = "results/statistical_runs"
EVAL_FILENAME    = "eval_results.json"
PHASE_KEY        = "ED+ES"


def load_seed_results(log_base: str, seeds: list) -> dict:
    """
    Load eval JSON for each seed. Returns dict {seed: metrics_dict}.
    Raises FileNotFoundError with a clear message if any file is missing.
    """
    all_data = {}

    for seed in seeds:
        json_path = Path(f"{log_base}{seed}") / EVAL_FILENAME

        if not json_path.exists():
            raise FileNotFoundError(
                f"Missing eval file for seed {seed}: {json_path}\n"
                f"Run evaluate_phase.py with --save_path {json_path} first."
            )

        with open(json_path) as f:
            data = json.load(f)

        if PHASE_KEY not in data:
            raise KeyError(
                f"Key '{PHASE_KEY}' not found in {json_path}.\n"
                f"Available keys: {list(data.keys())}\n"
                f"Was evaluate_phase.py run with --phase ALL?"
            )

        all_data[seed] = data[PHASE_KEY]
        print(f"  Loaded seed {seed:>5} : mean_dice={data[PHASE_KEY]['mean_dice']:.4f}  "
              f"mean_hd95={data[PHASE_KEY]['mean_hd95']:.2f}mm")

    return all_data


def compute_statistics(all_data: dict) -> dict:
    """
    Compute mean, std, min, max for each metric across all seeds.

    Returns
    -------
    summary : dict {metric_key: {'mean': float, 'std': float,
                                  'min': float, 'max': float,
                                  'per_seed': {seed: float}}}
    """
    summary = {}

    for key, label, fmt, unit in METRICS:
        values = {}
        for seed, data in all_data.items():
            if key not in data:
                raise KeyError(f"Metric '{key}' not found in seed {seed} results.")
            values[seed] = float(data[key])

        vals = list(values.values())
        summary[key] = {
            "label"   : label,
            "unit"    : unit,
            "fmt"     : fmt,
            "mean"    : float(np.mean(vals)),
            "std"     : float(np.std(vals, ddof=1)),  # sample std (N-1)
            "min"     : float(np.min(vals)),
            "max"     : float(np.max(vals)),
            "per_seed": values,
        }

    return summary


def print_summary_table(summary: dict, seeds: list) -> str:
    """
    Print and return a formatted summary table.
    """
    n = len(seeds)
    lines = []

    header = (
        f"\n{'=' * 62}\n"
        f"  LWEU-Net V2 — Statistical Runs ({n} seeds: "
        f"{', '.join(str(s) for s in seeds)})\n"
        f"{'=' * 62}"
    )
    lines.append(header)

    col_header = (
        f"  {'Metric':<14} {'Mean':>10} {'±Std':>8} "
        f"{'Min':>8} {'Max':>8}"
    )
    lines.append(col_header)
    lines.append("  " + "-" * 52)

    # Separator between Dice and HD95 groups
    dice_keys = ["mean_dice", "dice_lv", "dice_rv", "dice_myo"]
    hd_keys   = ["mean_hd95", "hd95_lv", "hd95_rv", "hd95_myo"]

    for group_keys in [dice_keys, hd_keys]:
        for key in group_keys:
            s    = summary[key]
            fmt  = s["fmt"]
            unit = s["unit"]
            mean_str = f"{s['mean']:{fmt}}{unit}"
            std_str  = f"±{s['std']:{fmt}}{unit}"
            min_str  = f"{s['min']:{fmt}}{unit}"
            max_str  = f"{s['max']:{fmt}}{unit}"
            line = (
                f"  {s['label']:<14} {mean_str:>10} {std_str:>8} "
                f"{min_str:>8} {max_str:>8}"
            )
            lines.append(line)
        lines.append("  " + "-" * 52)

    lines.append(f"{'=' * 62}")

    # Per-seed breakdown
    lines.append(f"\n  Per-seed breakdown (ED+ES combined):")
    lines.append(
        f"  {'Seed':<8} {'Mean Dice':>10} {'LV':>8} {'RV':>8} "
        f"{'MYO':>8} {'HD95':>8}"
    )
    lines.append("  " + "-" * 52)

    for seed in seeds:
        row = (
            f"  {seed:<8} "
            f"{summary['mean_dice']['per_seed'][seed]:>10.4f} "
            f"{summary['dice_lv']['per_seed'][seed]:>8.4f} "
            f"{summary['dice_rv']['per_seed'][seed]:>8.4f} "
            f"{summary['dice_myo']['per_seed'][seed]:>8.4f} "
            f"{summary['mean_hd95']['per_seed'][seed]:>7.2f}mm"
        )
        lines.append(row)
    lines.append("")

    table_str = "\n".join(lines)
    print(table_str)
    return table_str


def build_output_dicts(summary: dict, seeds: list) -> tuple:
    """
    Build the two output dicts: raw per-seed numbers and mean±std summary.
    """
    raw = {
        "seeds"  : seeds,
        "phase"  : PHASE_KEY,
        "results": {
            str(seed): {
                key: summary[key]["per_seed"][seed]
                for key, *_ in METRICS
            }
            for seed in seeds
        }
    }

    mean_std = {
        "seeds" : seeds,
        "phase" : PHASE_KEY,
        "n_runs": len(seeds),
        "metrics": {
            key: {
                "mean" : summary[key]["mean"],
                "std"  : summary[key]["std"],
                "min"  : summary[key]["min"],
                "max"  : summary[key]["max"],
            }
            for key, *_ in METRICS
        }
    }

    return raw, mean_std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
        help=f"Seeds to aggregate (default: {DEFAULT_SEEDS})"
    )
    parser.add_argument(
        "--log_base", default=DEFAULT_LOG_BASE,
        help=f"Log directory prefix before seed number (default: {DEFAULT_LOG_BASE})"
    )
    parser.add_argument(
        "--out_dir", default=DEFAULT_OUT_DIR,
        help=f"Output directory for results (default: {DEFAULT_OUT_DIR})"
    )
    args = parser.parse_args()

    print("=" * 62)
    print("  LWEU-Net V2 — Statistical Aggregation")
    print(f"  Seeds     : {args.seeds}")
    print(f"  Log base  : {args.log_base}")
    print(f"  Output dir: {args.out_dir}")
    print("=" * 62)

    # ── Step 1: Load all seed results ────────────────────────────
    print(f"\n[1/3] Loading eval results ({PHASE_KEY} phase)...")
    try:
        all_data = load_seed_results(args.log_base, args.seeds)
    except (FileNotFoundError, KeyError) as e:
        print(f"\n✗ {e}")
        sys.exit(1)

    # ── Step 2: Compute statistics ────────────────────────────────
    print(f"\n[2/3] Computing mean ± std across {len(args.seeds)} seeds...")
    summary = compute_statistics(all_data)

    # ── Step 3: Save outputs ──────────────────────────────────────
    print(f"\n[3/3] Saving results to {args.out_dir}/...")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw, mean_std = build_output_dicts(summary, args.seeds)

    raw_path  = out_dir / "all_seeds_raw.json"
    ms_path   = out_dir / "summary_mean_std.json"
    txt_path  = out_dir / "summary_table.txt"

    with open(raw_path, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"    Saved: {raw_path}")

    with open(ms_path, "w") as f:
        json.dump(mean_std, f, indent=2)
    print(f"    Saved: {ms_path}")

    table_str = print_summary_table(summary, args.seeds)

    with open(txt_path, "w") as f:
        f.write(table_str)
    print(f"    Saved: {txt_path}")

    print(f"\n{'=' * 62}")
    print(f"  Done. Thesis-ready numbers:")
    md = summary["mean_dice"]
    hd = summary["mean_hd95"]
    print(f"    Mean Dice : {md['mean']:.4f} ± {md['std']:.4f}")
    print(f"    Mean HD95 : {hd['mean']:.2f} ± {hd['std']:.2f}mm")
    print(f"{'=' * 62}\n")


if __name__ == "__main__":
    main()
