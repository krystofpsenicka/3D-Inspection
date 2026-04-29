"""Publication-quality plotting utilities for thesis figures."""

from __future__ import annotations

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Figure dimensions (inches) ──────────────────────────────────────────────

SINGLE_COL = 3.5    # IEEE single-column
DOUBLE_COL = 7.16   # IEEE double-column
THESIS_COL = 5.5    # A4 thesis text width (approx)

# ── Color palettes ──────────────────────────────────────────────────────────

CATEGORICAL_COLORS = plt.cm.tab10.colors[:10]
SEQUENTIAL_CMAP = "viridis"
DIVERGING_CMAP = "RdBu_r"

# ── Strategy display names ─────────────────────────────────────────────────
# Internal data keys → shorter labels used in plot legends/axes.

STRATEGY_DISPLAY_NAMES = {
    "cmaes_100": "cmaes",
    "targeted_100": "targeted",
}


def display_strategy(name: str) -> str:
    """Map internal strategy id to a shorter display label for plots."""
    return STRATEGY_DISPLAY_NAMES.get(name, name)


def setup_thesis_style():
    """Set matplotlib rcParams for publication-quality thesis figures.

    Also suppresses every form of "title above the plot" so the LaTeX
    caption is the only label for each figure. Axis labels, tick labels
    and legends are unaffected.
    """
    plt.rcParams.update({
        "font.size": 10,
        "font.family": "serif",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.4,
        "lines.markersize": 5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    })

    # Suppress every kind of "upper title" — the LaTeX caption below
    # each figure is the canonical label, so any text above the plot
    # area just duplicates it. Patching the methods centrally avoids
    # editing the ~30 set_title / 8 suptitle call sites across the
    # experiment scripts.
    from matplotlib.axes import Axes as _Axes
    from matplotlib.figure import Figure as _Figure
    global _ORIG_AX_SET_TITLE, _ORIG_FIG_SUPTITLE
    if _ORIG_AX_SET_TITLE is None:
        _ORIG_AX_SET_TITLE = _Axes.set_title
        _ORIG_FIG_SUPTITLE = _Figure.suptitle
    _Axes.set_title = lambda self, *args, **kwargs: None
    _Figure.suptitle = lambda self, *args, **kwargs: None


_ORIG_AX_SET_TITLE = None
_ORIG_FIG_SUPTITLE = None


def panel_title(ax, text: str, **kwargs):
    """Set a per-subplot title that bypasses the global suppression patch.

    Per-panel titles in multi-subplot figures label which subplot is which
    (e.g. model name, mesh group); they are NOT redundant with the LaTeX
    figure caption below the figure.
    """
    if _ORIG_AX_SET_TITLE is not None:
        _ORIG_AX_SET_TITLE(ax, text, **kwargs)
    else:
        ax.set_title(text, **kwargs)


def save_figure(fig, path: str, formats: list[str] | None = None):
    """Save figure in multiple formats (default: pdf + png)."""
    if formats is None:
        formats = ["pdf", "png"]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    base, _ = os.path.splitext(path)
    for fmt in formats:
        fig.savefig(f"{base}.{fmt}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Graph helper functions
# ═══════════════════════════════════════════════════════════════════════════


def grouped_bar(ax, data: dict[str, list[float]], x_labels: list[str],
                yerr: dict[str, list[float]] | None = None,
                ylabel: str = "", title: str = "",
                value_labels: bool = False, fmt: str = "%.1f"):
    """Grouped bar chart with optional error bars and value labels.

    Args:
        data: {group_name: [values_per_x]}
        x_labels: category labels on x-axis
        yerr: {group_name: [std_per_x]} for error bars
    """
    groups = list(data.keys())
    n_groups = len(groups)
    n_x = len(x_labels)
    x = np.arange(n_x)
    width = 0.8 / n_groups

    for i, grp in enumerate(groups):
        offset = (i - (n_groups - 1) / 2) * width
        err = yerr[grp] if yerr and grp in yerr else None
        bars = ax.bar(x + offset, data[grp], width, label=grp,
                      color=CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)],
                      yerr=err, capsize=2)
        if value_labels:
            ax.bar_label(bars, padding=2, fmt=fmt, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()


def violin_with_swarm(ax, data: dict[str, list[float]],
                      ylabel: str = "", title: str = ""):
    """Violin plot with individual data points overlaid."""
    labels = list(data.keys())
    values = [np.asarray(data[k]) for k in labels]

    parts = ax.violinplot(values, positions=range(len(labels)),
                          showmedians=True, showextrema=False)
    for pc in parts['bodies']:
        pc.set_alpha(0.4)

    # Overlay strip plot
    for i, vals in enumerate(values):
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   s=12, alpha=0.7, zorder=3,
                   color=CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)])

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)


def stacked_bar(ax, categories: list[str],
                stage_values: dict[str, list[float]],
                stage_colors: list[str] | None = None,
                ylabel: str = "", title: str = ""):
    """Stacked bar chart for timing breakdowns."""
    stages = list(stage_values.keys())
    n = len(categories)
    x = np.arange(n)
    colors = stage_colors or [CATEGORICAL_COLORS[i] for i in range(len(stages))]
    bottom = np.zeros(n)

    for i, stage in enumerate(stages):
        vals = np.array(stage_values[stage])
        ax.bar(x, vals, 0.6, bottom=bottom, label=stage, color=colors[i])
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=7)


def log_log_with_fit(ax, x, y, label: str = "", color=None):
    """Log-log scatter with power-law fit line."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = (x > 0) & (y > 0)
    x, y = x[mask], y[mask]
    c = color or CATEGORICAL_COLORS[0]

    ax.loglog(x, y, "o", color=c, label=label, markersize=5)

    if len(x) >= 2:
        coeffs = np.polyfit(np.log10(x), np.log10(y), 1)
        x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 50)
        y_fit = 10 ** np.polyval(coeffs, np.log10(x_fit))
        ax.loglog(x_fit, y_fit, "--", color=c, alpha=0.5,
                  label=f"slope={coeffs[0]:.2f}")

    ax.legend()


def dual_yaxis(ax, x, y1, y2, label1: str, label2: str,
               ylabel1: str = "", ylabel2: str = "",
               title: str = "", color1=None, color2=None):
    """Dual y-axis line plot."""
    c1 = color1 or CATEGORICAL_COLORS[0]
    c2 = color2 or CATEGORICAL_COLORS[1]

    ax.plot(x, y1, "o-", color=c1, label=label1)
    ax.set_ylabel(ylabel1, color=c1)
    ax.tick_params(axis="y", labelcolor=c1)
    if title:
        ax.set_title(title)

    ax2 = ax.twinx()
    ax2.plot(x, y2, "s--", color=c2, label=label2)
    ax2.set_ylabel(ylabel2, color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)
    return ax2


def convergence_plot(ax, histories: dict[str, list[list[float]]],
                     ylabel: str = "Coverage", title: str = ""):
    """Convergence curves: mean + shaded std across seeds.

    Args:
        histories: {method_name: [seed_1_history, seed_2_history, ...]}
    """
    for i, (name, seed_histories) in enumerate(histories.items()):
        max_len = max(len(h) for h in seed_histories)
        padded = np.full((len(seed_histories), max_len), np.nan)
        for j, h in enumerate(seed_histories):
            padded[j, :len(h)] = h
            padded[j, len(h):] = h[-1]  # pad with final value

        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        x = np.arange(max_len)
        c = CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)]
        ax.plot(x, mean, "-", color=c, label=name)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=c)

    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xlabel("Iteration")
    if title:
        ax.set_title(title)
    ax.legend()


def cdf_plot(ax, data: dict[str, list[float]],
             xlabel: str = "", title: str = ""):
    """Empirical CDF curves for multiple distributions."""
    for i, (name, values) in enumerate(data.items()):
        sorted_v = np.sort(values)
        cdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
        c = CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)]
        ax.step(sorted_v, cdf, where="post", color=c, label=name)

    ax.set_ylabel("CDF")
    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)
    ax.legend()


def heatmap_annotated(ax, row_labels: list[str], col_labels: list[str],
                      values: np.ndarray, fmt: str = ".1f",
                      cmap: str = "viridis", title: str = "",
                      xlabel: str = "", ylabel: str = "",
                      cbar_label: str = "",
                      overlay_mask: np.ndarray | None = None):
    """Annotated 2D heatmap.

    Cell text color is chosen by normalised cell value: dark cells
    (low end of the cmap) get white text, bright cells black. The
    crossover at ~0.55 matches the viridis luminance midpoint.

    ``cbar_label`` labels the colorbar (the third dimension).
    ``overlay_mask`` (same shape as ``values``) draws diagonal hatching
    across every True cell — used to flag e.g. cells whose coverage
    fell below the target.
    """
    from matplotlib.patches import Rectangle
    values = np.asarray(values)
    im = ax.imshow(values, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    finite = values[np.isfinite(values)]
    if finite.size:
        vmin, vmax = float(finite.min()), float(finite.max())
    else:
        vmin, vmax = 0.0, 1.0
    span = (vmax - vmin) or 1.0

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            v = values[i, j]
            norm = (v - vmin) / span
            text_color = "white" if norm < 0.55 else "black"
            if overlay_mask is not None and bool(overlay_mask[i, j]):
                # Diagonal hatching across the whole cell. Hatch colour
                # follows the same dark/bright rule as the text so it
                # stays visible regardless of the underlying viridis hue.
                ax.add_patch(Rectangle(
                    (j - 0.5, i - 0.5), 1.0, 1.0,
                    facecolor="none", edgecolor=text_color,
                    hatch="////", linewidth=0.0))
            ax.text(j, i, f"{v:{fmt}}", ha="center", va="center",
                    fontsize=7, color=text_color, zorder=5)

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    cbar = plt.colorbar(im, ax=ax)
    if cbar_label:
        cbar.set_label(cbar_label)


def pareto_front(ax, x, y, labels: list[str] | None = None,
                 xlabel: str = "", ylabel: str = "", title: str = ""):
    """Scatter with Pareto-dominated points grayed out."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(x)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and x[j] <= x[i] and y[j] <= y[i] and \
               (x[j] < x[i] or y[j] < y[i]):
                dominated[i] = True
                break

    ax.scatter(x[dominated], y[dominated], c="lightgray", marker="o",
               s=30, zorder=1, label="Dominated")
    ax.scatter(x[~dominated], y[~dominated], c=CATEGORICAL_COLORS[0],
               marker="o", s=50, zorder=2, label="Pareto front")

    # Connect Pareto front
    pf_idx = np.where(~dominated)[0]
    pf_order = pf_idx[np.argsort(x[pf_idx])]
    ax.plot(x[pf_order], y[pf_order], "--", color=CATEGORICAL_COLORS[0], alpha=0.5)

    if labels:
        for i in range(n):
            ax.annotate(labels[i], (x[i], y[i]), fontsize=6,
                        textcoords="offset points", xytext=(3, 3))

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=7)


def optimality_gap_bar(ax, solutions: dict[str, float], lower_bound: float,
                       ylabel: str = "Optimality ratio", title: str = ""):
    """Bar chart showing solution / lower_bound ratio for each method."""
    names = list(solutions.keys())
    ratios = [solutions[n] / lower_bound if lower_bound > 0 else 0 for n in names]
    x = np.arange(len(names))
    colors = [CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)] for i in range(len(names))]

    bars = ax.bar(x, ratios, 0.6, color=colors)
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.7, label="LP lower bound")
    ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=7)


def radar_chart(fig, metrics: dict[str, dict[str, float]], title: str = ""):
    """Radar/spider chart for multi-metric comparison.

    Args:
        metrics: {method_name: {metric_name: normalized_value_0_to_1}}
    """
    methods = list(metrics.keys())
    if not methods:
        return
    metric_names = list(metrics[methods[0]].keys())
    n_metrics = len(metric_names)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    ax = fig.add_subplot(111, polar=True)
    for i, method in enumerate(methods):
        values = [metrics[method].get(m, 0) for m in metric_names]
        values += values[:1]
        c = CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)]
        ax.plot(angles, values, "o-", color=c, label=method, linewidth=1.2)
        ax.fill(angles, values, alpha=0.1, color=c)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=8)
    ax.set_ylim(0, 1)
    if title:
        ax.set_title(title, y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=7)
