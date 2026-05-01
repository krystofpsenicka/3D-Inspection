"""Re-aggregate per-method mean F1 from the E05 raw runs and print it
alongside the headline table from chap06.tex (Table 6.1) for verification.

Reads `results/e05_visibility_comparison/raw/*.json` directly (no dependency
on the experiment script) so the output is independent verification.
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "results" / "e05_visibility_comparison" / "raw"

MODELS = ["duke_of_lancaster", "cat0", "david0", "wolf0"]
TOSCA_MODELS = ["cat0", "david0", "wolf0"]
METHODS = ["gpu_raycast", "cpu_raycast", "gpu_epsilon", "cpu_epsilon"]
TARGETS = {0.85, 0.90, 0.95, 0.97}
EXPECTED_SEEDS_PER_CELL = 3


def load_runs():
    runs = []
    skipped_target = 0
    skipped_model = 0
    skipped_method = 0
    total = 0
    for path in sorted(RAW_DIR.glob("*.json")):
        total += 1
        with path.open() as f:
            d = json.load(f)
        model = d["model"]
        method = d["method"]
        target = round(float(d["target_coverage"]), 4)
        if model not in MODELS:
            skipped_model += 1
            continue
        if method not in METHODS:
            skipped_method += 1
            continue
        if target not in TARGETS:
            skipped_target += 1
            continue
        runs.append(
            {
                "model": model,
                "method": method,
                "target": target,
                "seed": int(d["seed"]),
                "f1": float(d["mean_f1"]),
            }
        )
    return runs, total, skipped_model, skipped_method, skipped_target


def fmt_cell(values):
    if not values:
        return f"{'-':>16}"
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{m:.4f}±{s:.4f} (n={len(values)})"


def main():
    runs, total, skipped_model, skipped_method, skipped_target = load_runs()

    by_model_method = defaultdict(list)
    by_method_tosca = defaultdict(list)
    cell_counts = defaultdict(int)
    for r in runs:
        by_model_method[(r["model"], r["method"])].append(r["f1"])
        if r["model"] in TOSCA_MODELS:
            by_method_tosca[r["method"]].append(r["f1"])
        cell_counts[(r["model"], r["method"], r["target"])] += 1

    print("=" * 88)
    print("Per-(model, method) mean F1 over targets {0.85, 0.90, 0.95, 0.97} and seeds")
    print("=" * 88)
    header = f"{'method':<14} | " + " | ".join(f"{m:^22}" for m in MODELS)
    print(header)
    print("-" * len(header))
    for method in METHODS:
        row = [f"{method:<14}"]
        for model in MODELS:
            vals = by_model_method.get((model, method), [])
            row.append(f"{fmt_cell(vals):^22}")
        print(" | ".join(row))

    print()
    print("=" * 60)
    print("Headline table (mirrors Table 6.1 in chap06.tex)")
    print("=" * 60)
    print(f"{'Method':<14} | {'Duke':>8} | {'TOSCA avg':>10}")
    print("-" * 40)
    for method in METHODS:
        duke_vals = by_model_method.get(("duke_of_lancaster", method), [])
        tosca_vals = by_method_tosca.get(method, [])
        duke_mean = statistics.mean(duke_vals) if duke_vals else float("nan")
        tosca_mean = statistics.mean(tosca_vals) if tosca_vals else float("nan")
        print(f"{method:<14} | {duke_mean:>8.2f} | {tosca_mean:>10.2f}")

    print()
    print("=" * 60)
    print("Sanity summary")
    print("=" * 60)
    expected_cells = len(MODELS) * len(METHODS) * len(TARGETS)
    expected_files = expected_cells * EXPECTED_SEEDS_PER_CELL
    print(f"Files in raw/                  : {total}")
    print(f"  excluded (model not in table): {skipped_model}")
    print(f"  excluded (method not in table): {skipped_method}")
    print(f"  excluded (target not in set) : {skipped_target}")
    print(f"  included                     : {len(runs)}")
    print(
        f"Expected included              : {expected_files} "
        f"({len(MODELS)} models x {len(METHODS)} methods x {len(TARGETS)} targets x {EXPECTED_SEEDS_PER_CELL} seeds)"
    )

    off_cells = []
    for model in MODELS:
        for method in METHODS:
            for target in TARGETS:
                n = cell_counts.get((model, method, target), 0)
                if n != EXPECTED_SEEDS_PER_CELL:
                    off_cells.append((model, method, target, n))
    if off_cells:
        print(f"Cells with seed count != {EXPECTED_SEEDS_PER_CELL}:")
        for model, method, target, n in off_cells:
            print(f"  ({model}, {method}, target={target}): n={n}")
    else:
        print(f"All {expected_cells} cells have exactly {EXPECTED_SEEDS_PER_CELL} seeds.")


if __name__ == "__main__":
    main()
