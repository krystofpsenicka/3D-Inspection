#!/usr/bin/env python3
"""E14: scaling and discretization sensitivity along |C| and |P|.

Reviewer R4.21: scaling is shown only along the fleet size K, never along the
candidate count |C| or the surface-point count |P|. R4.22: the fixed 200,000
point discretization is not justified and no sensitivity check is reported.

This experiment sweeps one axis at a time on Duke (all other settings at their
defaults, K=5) and records per-stage timings together with solution quality
(selected viewpoint count, achieved coverage). Two questions are answered:

* runtime scaling of each stage as |C| and |P| grow (R4.21), and
* whether solution quality is stable around the |P|=200k / |C|=1500 defaults,
  i.e. that the choice is on a plateau rather than a cliff (R4.22).

    python -m experiments.e14_cp_scaling --dim P
    python -m experiments.e14_cp_scaling --dim C
    python -m experiments.e14_cp_scaling --dim both --seeds 42 123 7 2024 314
    python -m experiments.e14_cp_scaling --plots_only
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.common.config import RESULTS_DIR, SEEDS_5, ModelConfig
from experiments.common.persistence import load_run_result, save_run_result

_RUNTIME_IMPORT_ERROR: ImportError | None = None
try:
    from experiments.common.pipeline_run import run_pipeline_to_routes
    from experiments.common.pipeline_setup import DegenerateNormalsError, PipelineContext
    from experiments.common.runner import free_gpu_memory, handle_row_exception

    _RUNTIME_AVAILABLE = True
except ImportError as _e:
    _RUNTIME_AVAILABLE = False
    _RUNTIME_IMPORT_ERROR = _e

logger = logging.getLogger(__name__)

FLEET_SIZE = 5
VRP_ALPHA = 0.5

# Sweep grids centred on the Duke defaults (|P|=200k, |C|=1500).
P_GRID = [50_000, 100_000, 200_000, 400_000]
C_GRID = [500, 1000, 1500, 2500, 4000]


def _run_point(ctx, cfg, seed, *, num_candidates=None, num_points=None, with_vrp=False):
    art = run_pipeline_to_routes(
        ctx, cfg, seed, fleet_size=FLEET_SIZE, vrp_alpha=VRP_ALPHA,
        num_candidates=num_candidates, num_points=num_points, do_routing=with_vrp,
    )
    t = art.timings
    return {
        "seed": seed,
        "num_candidates": art.num_candidates,
        "num_points": art.num_points,
        "num_viewpoints": art.num_viewpoints,
        "coverage": art.coverage,
        "t_sample": t.get("t_sample", float("nan")),
        "t_vis": t.get("t_vis", float("nan")),
        "t_opt": t.get("t_opt", float("nan")),
        "t_vrp": t.get("t_vrp", float("nan")),
        "vrp_status": art.vrp_status,
        "vrp_makespan": art.vrp_makespan,
    }


def _sweep(ctx, cfg, seeds, dim, raw_dir, resume, results, with_vrp):
    grid = P_GRID if dim == "P" else C_GRID
    for val in grid:
        for seed in seeds:
            stem = f"dim={dim}_val={val}_seed={seed}"
            rpath = os.path.join(raw_dir, stem)
            if resume and os.path.exists(rpath + ".json"):
                results.append(load_run_result(rpath))
                logger.info("[resume] skip %s", stem)
                continue
            logger.info("Running dim=%s val=%d seed=%d", dim, val, seed)
            try:
                if dim == "P":
                    row = _run_point(ctx, cfg, seed, num_points=val, with_vrp=with_vrp)
                else:
                    row = _run_point(ctx, cfg, seed, num_candidates=val, with_vrp=with_vrp)
            except Exception as exc:  # noqa: BLE001
                handle_row_exception(exc, stem, resume=resume)
                free_gpu_memory()
                continue
            row["dim"] = dim
            row["swept_value"] = val
            results.append(row)
            save_run_result(row, rpath)
            logger.info("  vps=%d cov=%.3f t_vis=%.2fs t_opt=%.2fs t_vrp=%.2fs",
                        row["num_viewpoints"], row["coverage"],
                        row["t_vis"], row["t_opt"], row["t_vrp"])
            free_gpu_memory()


def _print_summary(results):
    for dim, label in (("P", "|P| surface points"), ("C", "|C| candidates")):
        rows = [r for r in results if r.get("dim") == dim]
        if not rows:
            continue
        vals = sorted({r["swept_value"] for r in rows})
        logger.info("\n%s\nE14 scaling along %s (mean over seeds)\n%s", "=" * 74, label, "=" * 74)
        logger.info("%10s %6s %7s %8s %8s %8s %9s", "value", "vps", "cov",
                    "t_vis", "t_opt", "t_vrp", "makespan")
        for v in vals:
            rv = [r for r in rows if r["swept_value"] == v]

            def _m(k):
                xs = [r[k] for r in rv if not (isinstance(r[k], float) and np.isnan(r[k]))]
                return float(np.mean(xs)) if xs else float("nan")

            logger.info("%10d %6.1f %7.3f %8.2f %8.2f %8.2f %9.1f", v,
                        _m("num_viewpoints"), _m("coverage"), _m("t_vis"),
                        _m("t_opt"), _m("t_vrp"), _m("vrp_makespan"))


def main():
    p = argparse.ArgumentParser(description="E14: |C| and |P| scaling / sensitivity")
    p.add_argument("--dim", choices=["C", "P", "both"], default="both")
    p.add_argument("--with_vrp", action="store_true",
                   help="Also run distance-matrix + VRP per point (adds ~3 min/point on "
                        "Duke). Off by default: |C|/|P| scaling is about sampling/visibility/"
                        "set-cover; VRP cost is viewpoint-count-driven (see E07/E08).")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_5)
    p.add_argument("--output_dir", default=os.path.join(RESULTS_DIR, "e14_cp_scaling"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--plots_only", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)-8s %(name)s: %(message)s")

    raw_dir = os.path.join(args.output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    results: list[dict] = []

    if args.plots_only:
        for f in sorted(os.listdir(raw_dir)):
            if f.endswith(".json"):
                results.append(load_run_result(os.path.join(raw_dir, f[:-5])))
        _print_summary(results)
        return

    if not _RUNTIME_AVAILABLE:
        raise SystemExit(f"Runtime imports unavailable ({_RUNTIME_IMPORT_ERROR}).")

    cfg = ModelConfig.duke_of_lancaster()
    ctx = PipelineContext(cfg)
    try:
        ctx.load_mesh()
        ctx.build_sampling_og()
    except DegenerateNormalsError as e:
        raise SystemExit(f"Duke setup failed: {e}")

    dims = ["P", "C"] if args.dim == "both" else [args.dim]
    for dim in dims:
        _sweep(ctx, cfg, args.seeds, dim, raw_dir, args.resume, results, args.with_vrp)

    _print_summary(results)


if __name__ == "__main__":
    main()
