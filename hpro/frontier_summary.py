"""Aggregate the §9 headroom frontier: pipeline operating points vs their
joint-refined (elastic + audit-gate) counterparts.

Reads every ``hpro/results/frontier*/<name>/joint_pilot.json`` produced by
``joint_pilot.py --modes warm`` runs over baselines generated at different
``--target_coverage`` settings, and renders one coverage-vs-makespan frontier
plot plus a table. The question it answers (RESEARCH_PLAN.md §9.3): does the
joint optimizer dominate the pipeline's own coverage/makespan frontier, and
by how much at each operating point?

Directory names encode the operating point and (optionally) the pipeline
seed: ``pilot_tc78`` is seed 42 (the original sweep), ``pilot_s7_tc78`` is
seed 7. With more than one seed present, the per-operating-point deltas are
reported as mean ± std across seeds and the §3.4 rule ("no claim from fewer
than ~5 runs, always mean ± std") is what this table exists to satisfy.
"""

import argparse
import glob
import json
import os
import re
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_DIR = os.path.dirname(os.path.abspath(__file__))
_RESULTS = os.path.join(_DIR, "results")


def parse_name(name):
    """'pilot_s7_tc78' -> (7, 78); 'pilot_tc78' -> (42, 78);
    'pilot_baseline' -> (42, 95)."""
    m = re.match(r"pilot(?:_s(\d+))?_tc(\d+)$", name)
    if m:
        return int(m.group(1) or 42), int(m.group(2))
    m = re.match(r"pilot_s(\d+)_baseline$", name)
    if m:
        return int(m.group(1)), 95
    if name == "pilot_baseline":
        return 42, 95
    return None, None


def load_runs(root=_RESULTS):
    runs = []
    for path in sorted(glob.glob(
            os.path.join(root, "frontier*", "*", "joint_pilot.json"))):
        name = os.path.basename(os.path.dirname(path))
        seed, tc = parse_name(name)
        if seed is None:
            continue
        with open(path) as fh:
            data = json.load(fh)
        by = {r["tag"]: r for r in data["results"]}
        if "pipeline" not in by or "joint-warm" not in by:
            continue
        runs.append(dict(name=name, seed=seed, tc=tc, by=by))
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plot_seed", type=int, default=42,
                    help="Which seed's frontier to draw (table covers all).")
    ap.add_argument("--root", default=_RESULTS,
                    help="Results root holding the frontier*/ groups.")
    ap.add_argument("--compare", default=None,
                    help="A second results root; prints per-run before/after "
                         "deltas against it. Used for the §9.12 clearance "
                         "re-run, where the question is not only 'does the "
                         "constraint hold now' but 'what did holding it "
                         "cost'.")
    args = ap.parse_args()

    runs = load_runs(args.root)
    if not runs:
        raise SystemExit(f"No runs found under {args.root}/frontier*/")
    seeds = sorted({r["seed"] for r in runs})

    if args.compare:
        old = {r["name"]: r for r in load_runs(args.compare)}
        print(f"=== before ({args.compare}) -> after ({args.root}) ===")
        print(f"{'run':<18} {'clearance':>19} {'coverage':>19} "
              f"{'makespan (m)':>21}")
        for r in sorted(runs, key=lambda r: (r["seed"], r["tc"])):
            o = old.get(r["name"])
            if o is None:
                continue
            a, b = o["by"]["joint-warm"], r["by"]["joint-warm"]
            print(f"{r['name']:<18} "
                  f"{a['min_clearance']:>8.4f}->{b['min_clearance']:<8.4f} "
                  f"{a['coverage']:>8.4f}->{b['coverage']:<8.4f} "
                  f"{a['makespan']:>9.2f}->{b['makespan']:<9.2f}")
        print()

    print(f"{'run':<18} {'seed':>4} {'arm':<12} {'cov':>7} {'makespan':>9} "
          f"{'total':>8} {'poses':>6} {'vias':>5} {'clear':>6} {'sep':>6} "
          f"{'wall s':>7}")
    for r in sorted(runs, key=lambda r: (r["seed"], r["tc"])):
        for tag in ("pipeline", "joint-warm"):
            v = r["by"][tag]
            print(f"{r['name']:<18} {r['seed']:>4} {tag:<12} "
                  f"{v['coverage']:>7.4f} {v['makespan']:>8.1f}m "
                  f"{v['total_length']:>7.1f}m {v['n_poses']:>6d} "
                  f"{v.get('n_vias', 0):>5d} {v['min_clearance']:>6.2f} "
                  f"{v.get('min_separation', float('nan')):>6.2f} "
                  f"{v['wall_s']:>7.1f}")
        p, j = r["by"]["pipeline"], r["by"]["joint-warm"]
        print(f"{'':<18} {'':>4} {'Δ':<12} {j['coverage']-p['coverage']:>+7.4f} "
              f"{j['makespan']-p['makespan']:>+8.1f}m")

    # Per-operating-point aggregation across seeds.
    print(f"\n=== across {len(seeds)} pipeline seed(s) {seeds} ===")
    print(f"{'target':>7} {'n':>3} {'V (poses)':>12} {'Δcoverage (pts)':>22} "
          f"{'Δmakespan (%)':>20} {'dominated':>10}")
    agg_rows = []
    for tc in sorted({r["tc"] for r in runs}):
        grp = [r for r in runs if r["tc"] == tc]
        dcov = [100 * (r["by"]["joint-warm"]["coverage"]
                       - r["by"]["pipeline"]["coverage"]) for r in grp]
        dmk = [100 * (r["by"]["joint-warm"]["makespan"]
                      / r["by"]["pipeline"]["makespan"] - 1) for r in grp]
        vs = [r["by"]["pipeline"]["n_poses"] for r in grp]
        dom = sum(1 for c, m in zip(dcov, dmk) if c > 0 and m < 0)
        sd = (lambda x: statistics.stdev(x) if len(x) > 1 else 0.0)
        print(f"{tc/100:>7.2f} {len(grp):>3d} "
              f"{min(vs):>5d}–{max(vs):<6d} "
              f"{statistics.mean(dcov):>+13.2f} ± {sd(dcov):<6.2f} "
              f"{statistics.mean(dmk):>+11.2f} ± {sd(dmk):<6.2f} "
              f"{dom:>5d}/{len(grp):<4d}")
        agg_rows.append((tc, dcov, dmk, dom, len(grp)))

    # Constraint audit: separation and clearance are claimed as constraints,
    # so the worst case over all runs is the number that matters, not a mean.
    seps = [r["by"]["joint-warm"].get("min_separation") for r in runs]
    seps = [s for s in seps if s is not None and s == s]
    base_seps = [r["by"]["pipeline"].get("min_separation") for r in runs]
    base_seps = [s for s in base_seps if s is not None and s == s]
    clears = [r["by"]["joint-warm"]["min_clearance"] for r in runs]
    if seps:
        print(f"\nconstraint audit (worst over {len(seps)} runs): "
              f"min inter-robot separation {min(seps):.2f} m "
              f"(pipeline's own: {min(base_seps):.2f} m), "
              f"min clearance {min(clears):.2f} m")

    all_dcov = [c for _, dc, _, _, _ in agg_rows for c in dc]
    all_dmk = [m for _, _, dm, _, _ in agg_rows for m in dm]
    n_dom = sum(d for _, _, _, d, _ in agg_rows)
    n_all = sum(n for _, _, _, _, n in agg_rows)
    sd = (lambda x: statistics.stdev(x) if len(x) > 1 else 0.0)
    print(f"\noverall: Δcov {statistics.mean(all_dcov):+.2f} ± "
          f"{sd(all_dcov):.2f} pts, Δmakespan "
          f"{statistics.mean(all_dmk):+.2f} ± {sd(all_dmk):.2f} %, "
          f"Pareto-dominated in {n_dom}/{n_all} runs "
          f"(worst Δcov {min(all_dcov):+.2f} pts, "
          f"worst Δmakespan {max(all_dmk):+.2f} %)")

    plot_runs = sorted([r for r in runs if r["seed"] == args.plot_seed],
                       key=lambda r: r["by"]["pipeline"]["n_poses"])
    if not plot_runs:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    px = [r["by"]["pipeline"]["makespan"] for r in plot_runs]
    py = [r["by"]["pipeline"]["coverage"] for r in plot_runs]
    jx = [r["by"]["joint-warm"]["makespan"] for r in plot_runs]
    jy = [r["by"]["joint-warm"]["coverage"] for r in plot_runs]
    ax.plot(px, py, "s-", color="0.35",
            label="pipeline (set cover + VRP + ST-A*)")
    ax.plot(jx, jy, "o-", color="crimson",
            label="joint refinement (elastic band, audit-gated)")
    for r, x0, y0, x1, y1 in zip(plot_runs, px, py, jx, jy):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="0.6", lw=0.8))
        ax.annotate(f"V={r['by']['pipeline']['n_poses']}", (x0, y0),
                    textcoords="offset points", xytext=(4, -10), fontsize=8)
    ax.set_xlabel("makespan (m)")
    ax.set_ylabel("coverage (pipeline's own ray-cast)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("coverage / makespan frontier: pipeline vs joint refinement"
                 + (f"  (seed {args.plot_seed})" if len(seeds) > 1 else ""))
    out = os.path.join(args.root, "frontier", "frontier.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
