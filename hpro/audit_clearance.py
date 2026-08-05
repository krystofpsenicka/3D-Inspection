"""Independent structure-clearance audit (RESEARCH_PLAN §9.12).

Deliberately a *separate program* from the solver. `joint_pilot.py` reports the
clearance of its own results, which is exactly the arrangement that let a
constraint go unenforced for a session at a time: a solver that samples a
segment at 10 points is not wrong about the 10 points, it is silent about
everything between them, and a self-report inherits that silence. This script
rebuilds the ESDF from the pipeline run on disk and re-measures clearance for

  * the **pipeline's own executed ST-A\\* trajectories** — the number that was
    `NaN` in every stored result, so "we clear the structure better than the
    baseline" was unsupported in both directions; and
  * any chains this repo has dumped (``chains.npz`` next to a
    ``joint_pilot.json``), at whatever resolution is asked for.

Clearance is reported as a *ladder* over samples-per-segment. The minimum of a
continuous quantity along a segment need not land on a sample, so a single
resolution is a claim, not a measurement; the ladder shows where it converges.

Usage::

    ipy_inspection hpro/audit_clearance.py outputs/pilot_baseline \\
        [--chains hpro/results/frontier_clear/frontier/pilot_baseline/chains.npz]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_DIR)
for p in (_DIR, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from joint_pilot import Esdf, segment_samples          # noqa: E402

RUNGS = (10, 40, 160, 640)


def ladder(esdf, chains, rungs=RUNGS):
    """{samples_per_segment: worst clearance} over a list of (T,3) chains."""
    out = {}
    for ps in rungs:
        with torch.no_grad():
            d = torch.cat([segment_samples(c, ps) for c in chains
                           if len(c) > 1])
            out[ps] = float(esdf(d).min())
    return out


def spacing_of(chains):
    """Longest segment in the set — what the ladder's spacing is driven by."""
    return max(float(torch.linalg.norm(c[1:] - c[:-1], dim=1).max())
               for c in chains if len(c) > 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pipeline_dir")
    ap.add_argument("--chains", nargs="*", default=[],
                    help="chains.npz files (or globs) to audit as well.")
    ap.add_argument("--margin", type=float, default=0.50,
                    help="Clearance constraint (robot_radius + margin).")
    ap.add_argument("--calibrate", type=int, default=0,
                    help="Sample N points ON the mesh surface and report what "
                         "the ESDF reads there. Ideally 0; the deviation is "
                         "this field's absolute bias, and no clearance number "
                         "from it means anything to finer precision than "
                         "that. Both arms are measured in the same field, so "
                         "the bias cancels in comparisons but NOT in "
                         "feasibility verdicts.")
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args()

    from VRP.utils.serialization import load_pipeline
    from shared.mesh_loader import load_and_transform_mesh
    from shared.grid_builder_utils import build_occupancy_grid

    device = "cuda"
    data = load_pipeline(args.pipeline_dir)
    # Same mesh, same grid, same resolution as joint_pilot.main — the audit
    # must run against the field the solver was constrained by, or it is
    # measuring a different problem.
    raw_tm = load_and_transform_mesh(data["mesh_path"],
                                     data["mesh_target_length"],
                                     data["mesh_pose"])
    og = build_occupancy_grid(mesh=raw_tm, padding=2.0, inflation_voxels=0,
                              resolution=0.10, fill_interior=True)
    esdf = Esdf(og, device)

    report = {"pipeline_dir": args.pipeline_dir, "margin": args.margin,
              "arms": {}}

    if args.calibrate:
        # A voxel is occupied if the mesh touches it anywhere, and the EDT
        # measures to voxel CENTRES — so the field systematically reads
        # further from the structure than reality. Quantify it instead of
        # calling it "about half a voxel".
        surf, _ = __import__("trimesh").sample.sample_surface(
            raw_tm, args.calibrate)
        with torch.no_grad():
            v = esdf(torch.tensor(np.asarray(surf), dtype=torch.float32,
                                  device=device)).cpu().numpy()
        q = np.percentile(v, [1, 25, 50, 75, 99])
        report["esdf_bias_on_surface"] = dict(
            n=int(args.calibrate), mean=float(v.mean()), std=float(v.std()),
            p1=float(q[0]), p25=float(q[1]), median=float(q[2]),
            p75=float(q[3]), p99=float(q[4]),
            resolution=float(og.resolution))
        print(f"\nESDF calibration on {args.calibrate} true surface points "
              f"(voxel {og.resolution:.2f} m) — a perfect field would read 0:")
        print(f"  mean {v.mean():+.4f} m   std {v.std():.4f}   "
              f"p1 {q[0]:+.4f}   median {q[2]:+.4f}   p99 {q[4]:+.4f}")
        print(f"  => absolute clearance numbers from this field carry "
              f"~{abs(v.mean()):.3f} m of optimism; differences between arms "
              f"measured in it do not.")

    # --- the pipeline's own executed trajectories -------------------------
    trajs = [torch.tensor(np.asarray(t)[:, :3], dtype=torch.float32,
                          device=device)
             for t in data["exec_result"].all_traj_positions]
    lad = ladder(esdf, trajs)
    report["arms"]["pipeline-executed"] = dict(
        ladder={str(k): v for k, v in lad.items()},
        max_segment_m=spacing_of(trajs),
        n_points=[int(len(t)) for t in trajs])

    def show(name, lad, extra=""):
        worst = lad[max(RUNGS)]
        verdict = "OK " if worst >= args.margin else "VIOLATION"
        print(f"{name:<44} " + "  ".join(f"{k}:{v:.4f}" for k, v in lad.items())
              + f"   -> {verdict} {extra}")

    print(f"\nclearance ladder (samples/segment), constraint {args.margin} m")
    print("-" * 108)
    show("pipeline-executed (ST-A*)", lad,
         f"[longest segment {spacing_of(trajs):.2f} m]")

    # --- anything we produced --------------------------------------------
    paths = [p for g in args.chains for p in sorted(glob.glob(g))]
    for path in paths:
        z = np.load(path)
        by_tag = {}
        for k in z.files:                      # keys are "<tag>__r<idx>"
            tag = k.split("__r")[0]
            by_tag.setdefault(tag, []).append(
                torch.tensor(z[k], dtype=torch.float32, device=device))
        for tag, chs in by_tag.items():
            lad = ladder(esdf, chs)
            label = f"{os.path.basename(os.path.dirname(path))}/{tag}"
            report["arms"][label] = dict(
                ladder={str(k): v for k, v in lad.items()},
                max_segment_m=spacing_of(chs))
            show(label, lad, f"[longest segment {spacing_of(chs):.2f} m]")

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nJSON: {args.json_out}")


if __name__ == "__main__":
    main()
