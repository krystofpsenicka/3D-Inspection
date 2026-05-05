#!/usr/bin/env python3
"""E04: Curvature Sensitivity - sweep ``curvature_knn_k`` x ``position_weight``.

  curvature_knn_k  {5, 10, 20, 40, 80}
  position_weight  {1.0, 2.5, 5.0, 10.0}
  3 seeds; per combo: sample with curvature weighting -> visibility -> set cover.

    python -m experiments.e04_curvature_sensitivity
    python -m experiments.e04_curvature_sensitivity --plots_only
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.common.config import RESULTS_DIR, SEEDS_3, ModelConfig
from experiments.common.persistence import load_run_result, save_run_result
from experiments.common.plotting import (
    CATEGORICAL_COLORS,
    DOUBLE_COL,
    THESIS_COL,
    heatmap_annotated,
    save_figure,
    setup_thesis_style,
)

# Runtime imports (need isaaclab/CUDA). Plot-only mode skips these.
_RUNTIME_IMPORT_ERROR: ImportError | None = None
try:
    import cupy as cp

    from experiments.common.pipeline_setup import (
        DegenerateNormalsError,
        PipelineContext,
    )
    from experiments.common.runner import free_gpu_memory, set_seed, timed
    from shared.types import Side
    from visibility.set_cover import LazyGreedySetCover

    _RUNTIME_AVAILABLE = True
except ImportError as _e:
    cp = None
    set_seed = timed = free_gpu_memory = None
    PipelineContext = DegenerateNormalsError = None
    Side = None
    LazyGreedySetCover = None
    _RUNTIME_AVAILABLE = False
    _RUNTIME_IMPORT_ERROR = _e

logger = logging.getLogger(__name__)

KNN_K_VALUES = [5, 10, 20, 40, 80]
POSITION_WEIGHTS = [1.0, 2.5, 5.0, 10.0]
N_CANDIDATES = 2000


def run_single(knn_k: int, position_weight: float, seed: int) -> dict:
    set_seed(seed)
    model_cfg = ModelConfig.duke_of_lancaster()
    ctx = PipelineContext(model_cfg)
    ctx.load_mesh()
    target_points, normals = ctx.sample_surface()
    ctx.build_sampling_og()

    sampler = ctx.build_sampler("targeted")
    vis_query = ctx.build_visibility_query("raycast")
    set_seed(seed)  # experiment seed for candidate generation

    with timed() as t_sample:
        pos_gpu, rot_gpu = sampler.sample(
            cp.arange(len(target_points)),
            N_CANDIDATES,
            side=Side.OUTSIDE,
            curvature_weighting=True,
            curvature_knn_k=knn_k,
            position_weight=position_weight,
        )

    with timed() as t_vis:
        V_gpu, _ = vis_query.compute_visibility_batch(pos_gpu, rot_gpu)

    # LazyGreedySetCover is the CPU optimizer; move GPU arrays to host first.
    V = cp.asnumpy(V_gpu)
    pos_np = cp.asnumpy(pos_gpu)
    rot_np = cp.asnumpy(rot_gpu)

    with timed() as t_opt:
        optimizer = LazyGreedySetCover(
            len(target_points),
            pos_np,
            rot_np,
            V,
        )
        opt_result = optimizer.optimize(
            target_coverage=0.95,
            max_viewpoints=1000,
        )

    return {
        "knn_k": knn_k,
        "position_weight": position_weight,
        "seed": seed,
        "num_viewpoints": opt_result.num_viewpoints,
        "coverage": float(opt_result.total_coverage),
        "redundancy": float(opt_result.redundancy),
        "sampling_time": t_sample.elapsed,
        "visibility_time": t_vis.elapsed,
        "optimization_time": t_opt.elapsed,
        "total_time": t_sample.elapsed + t_vis.elapsed + t_opt.elapsed,
    }


def generate_plots(results: list[dict], output_dir: str):
    setup_thesis_style()
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    if not results:
        logger.warning("No results for plotting")
        return

    k_vals = sorted(set(r["knn_k"] for r in results))
    w_vals = sorted(set(r["position_weight"] for r in results))

    k_labels = [str(k) for k in k_vals]
    w_labels = [str(w) for w in w_vals]

    fig, ax = plt.subplots(figsize=(THESIS_COL, 4))
    vp_vals = np.zeros((len(k_vals), len(w_vals)))
    for i, k in enumerate(k_vals):
        for j, w in enumerate(w_vals):
            vals = [
                r["num_viewpoints"]
                for r in results
                if r["knn_k"] == k and r["position_weight"] == w
            ]
            vp_vals[i, j] = np.mean(vals) if vals else 0
    heatmap_annotated(
        ax,
        k_labels,
        w_labels,
        vp_vals,
        fmt=".0f",
        title="Viewpoints Selected",
        xlabel="Position weight",
        ylabel="Curvature knn_k",
        cbar_label="Selected viewpoints",
    )
    save_figure(fig, os.path.join(fig_dir, "e04_heatmap_viewpoints"))

    logger.info("E04 figures saved to %s", fig_dir)


def main():
    p = argparse.ArgumentParser(description="E04: Curvature Sensitivity")
    p.add_argument("--knn_k_values", type=int, nargs="+", default=KNN_K_VALUES)
    p.add_argument("--position_weights", type=float, nargs="+", default=POSITION_WEIGHTS)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_3)
    p.add_argument("--output_dir", default=os.path.join(RESULTS_DIR, "e04_curvature_sensitivity"))
    p.add_argument("--plots_only", action="store_true")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows whose result JSON exists; exit non-zero on CUDA OOM so an outer "
        "restart loop can reclaim GPU memory.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(name)s: %(message)s",
    )

    if not args.plots_only and not _RUNTIME_AVAILABLE:
        raise SystemExit(
            f"Runtime imports unavailable ({_RUNTIME_IMPORT_ERROR}). "
            "Activate the isaaclab env or pass --plots_only."
        )

    raw_dir = os.path.join(args.output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    all_results = []

    if not args.plots_only:
        total = len(args.knn_k_values) * len(args.position_weights) * len(args.seeds)
        run_idx = 0

        for knn_k in args.knn_k_values:
            for pw in args.position_weights:
                for seed in args.seeds:
                    run_idx += 1
                    rpath = os.path.join(
                        raw_dir,
                        f"knn_k={knn_k}_pw={pw}_seed={seed}",
                    )
                    row_desc = f"knn_k={knn_k} pw={pw} seed={seed}"

                    if args.resume and os.path.exists(rpath + ".json"):
                        all_results.append(load_run_result(rpath))
                        logger.info("[%d/%d] [resume] skip %s", run_idx, total, row_desc)
                        continue

                    logger.info(
                        "[%d/%d] knn_k=%d position_weight=%.1f seed=%d",
                        run_idx,
                        total,
                        knn_k,
                        pw,
                        seed,
                    )

                    try:
                        result = run_single(knn_k, pw, seed)
                        all_results.append(result)
                        save_run_result(result, rpath)
                        logger.info(
                            "  vps=%d cov=%.2f%% time=%.1fs",
                            result["num_viewpoints"],
                            result["coverage"] * 100,
                            result["total_time"],
                        )
                    except Exception as e:
                        logger.error("  FAILED [%s]: %s", row_desc, e, exc_info=True)
                    finally:
                        free_gpu_memory()
    else:
        for fname in sorted(os.listdir(raw_dir)):
            if fname.endswith(".json"):
                all_results.append(
                    load_run_result(os.path.join(raw_dir, fname.replace(".json", "")))
                )

    if all_results:
        generate_plots(all_results, args.output_dir)


if __name__ == "__main__":
    main()
