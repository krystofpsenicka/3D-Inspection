"""Aggregate the §9 headroom frontier: pipeline operating points vs their
joint-refined (elastic + audit-gate) counterparts.

Reads every ``hpro/results/frontier/<name>/joint_pilot.json`` produced by
``joint_pilot.py --modes warm`` runs over baselines generated at different
``--target_coverage`` settings, and renders one coverage-vs-makespan frontier
plot plus a table. The question it answers (RESEARCH_PLAN.md §9.3): does the
joint optimizer dominate the pipeline's own coverage/makespan frontier, and
by how much at each operating point?
"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTIER = os.path.join(_DIR, "results", "frontier")


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(FRONTIER, "*", "joint_pilot.json"))):
        name = os.path.basename(os.path.dirname(path))
        with open(path) as fh:
            data = json.load(fh)
        by = {r["tag"]: r for r in data["results"]}
        rows.append((name, by))

    rows.sort(key=lambda r: r[1]["pipeline"]["n_poses"])

    print(f"{'baseline':<14} {'arm':<12} {'cov':>7} {'makespan':>9} "
          f"{'total':>8} {'poses':>6} {'vias':>5} {'clear':>6} {'wall s':>7}")
    for name, by in rows:
        for tag in ("pipeline", "joint-warm"):
            r = by.get(tag)
            if r is None:
                continue
            print(f"{name:<14} {tag:<12} {r['coverage']:>7.4f} "
                  f"{r['makespan']:>8.1f}m {r['total_length']:>7.1f}m "
                  f"{r['n_poses']:>6d} {r.get('n_vias', 0):>5d} "
                  f"{r['min_clearance']:>6.2f} {r['wall_s']:>7.1f}")
        p, j = by["pipeline"], by.get("joint-warm")
        if j:
            print(f"{'':<14} {'Δ':<12} {j['coverage']-p['coverage']:>+7.4f} "
                  f"{j['makespan']-p['makespan']:>+8.1f}m")

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    px = [by["pipeline"]["makespan"] for _, by in rows]
    py = [by["pipeline"]["coverage"] for _, by in rows]
    jx = [by["joint-warm"]["makespan"] for _, by in rows if "joint-warm" in by]
    jy = [by["joint-warm"]["coverage"] for _, by in rows if "joint-warm" in by]
    ax.plot(px, py, "s-", color="0.35", label="pipeline (set cover + VRP + ST-A*)")
    ax.plot(jx, jy, "o-", color="crimson",
            label="joint refinement (elastic band, audit-gated)")
    for (name, by), x0, y0, x1, y1 in zip(rows, px, py, jx, jy):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="0.6", lw=0.8))
        ax.annotate(f"V={by['pipeline']['n_poses']}", (x0, y0),
                    textcoords="offset points", xytext=(4, -10), fontsize=8)
    ax.set_xlabel("makespan (m)")
    ax.set_ylabel("coverage (pipeline's own ray-cast)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("coverage / makespan frontier: pipeline vs joint refinement")
    out = os.path.join(FRONTIER, "frontier.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
