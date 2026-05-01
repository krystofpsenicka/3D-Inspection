"""Lower-bound sidecar helpers.

Keeps LBs in their own files next to main results:
- CSV experiments (e06): ``lower_bounds.csv`` alongside ``results.csv``.
- JSON experiments (e08 / e09): ``raw_lb/<same_stem>.json``.

Why sidecars? Old runs can be reused  --  we never re-solve the expensive
VRP+MAPF pipeline just to get better LBs. When a tighter LB (e.g. cuOpt
dual bound or a sharper analytical formulation) becomes available we
just rewrite the sidecar and re-plot.

All sidecars carry the same schema (a subset is written per-experiment
depending on whether MAPF ran):

    vrp_makespan_lb_m, vrp_total_cost_lb_m, vrp_objective_lb_m,
    mapf_makespan_time_lb_s, mapf_total_time_lb_s, mapf_objective_time_lb_s,
    vrp_objective_best_bound_m    # cuOpt dual bound (0.0 = unavailable)

Key columns (the columns identifying which main-result row the LB
corresponds to) are experiment-specific and passed in by the caller.
"""

from __future__ import annotations

import csv as _csv
import json as _json
import logging
import os
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

# `cupy` is used by compute_all_lbs/recompute_lbs only. Importing this module
# (e.g., for `load_raw_lb_dir`/`load_lb_csv` in --plots_only mode) must work
# in a vanilla numpy+matplotlib environment, so the cupy dependency is
# deferred until the GPU helpers are actually called.
try:
    import cupy as cp
except ImportError:
    cp = None

from experiments.common.lower_bounds import vrp_lb_meters, vrp_mapf_lb_seconds

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Field definitions
# ═══════════════════════════════════════════════════════════════════════════

_VRP_LB_FIELDS = (
    "vrp_makespan_lb_m",
    "vrp_total_cost_lb_m",
    "vrp_objective_lb_m",
)
_MAPF_LB_FIELDS = (
    "mapf_makespan_time_lb_s",
    "mapf_total_time_lb_s",
    "mapf_objective_time_lb_s",
)
_CUOPT_FIELD = "vrp_objective_best_bound_m"

ALL_LB_FIELDS = _VRP_LB_FIELDS + _MAPF_LB_FIELDS + (_CUOPT_FIELD,)
VRP_ONLY_LB_FIELDS = _VRP_LB_FIELDS + (_CUOPT_FIELD,)


# ═══════════════════════════════════════════════════════════════════════════
# Core computation
# ═══════════════════════════════════════════════════════════════════════════


def compute_all_lbs(
    dist_matrix_cp,
    depot_indices: Sequence[int],
    k: int,
    n_waypoints: int,
    beta: float,
    *,
    include_mapf: bool = True,
    vrp_best_bound_m: float = 0.0,
    vrp_objective_value_m: float | None = None,
    cruise_speed: float | None = None,
    dwell_s: float | None = None,
) -> dict:
    """Compute analytical LBs from a distance matrix + instance params and
    merge with an optional cuOpt dual bound.

    Args:
        dist_matrix_cp: CuPy distance matrix over depots ∪ waypoints.
        depot_indices:  indices of depot rows/cols.
        k:              number of vehicles.
        n_waypoints:    number of inspection waypoints (n - K).
        beta:           VRP objective blend.
        include_mapf:   if True, also compute the seconds-mapped MAPF LBs.
        vrp_best_bound_m: cuOpt dual bound if available (else 0.0).
        vrp_objective_value_m: observed VRP objective in meters (used only
                               for a sanity warning when the bound beats it).
        cruise_speed / dwell_s: override VRP constants (for experiments that
                                want to parameterise). If None, falls back to
                                VRP.core.constants.AUV_CRUISE_SPEED and
                                SPACE_TIME_DWELL_S.

    Returns:
        Dict with keys in ``ALL_LB_FIELDS`` (or ``VRP_ONLY_LB_FIELDS`` when
        ``include_mapf=False``). All in meters / seconds as labelled.
    """
    D_np = cp.asnumpy(dist_matrix_cp).astype(np.float64)
    vrp_m = vrp_lb_meters(D_np, depot_indices, k, beta)

    out: dict = {
        "vrp_makespan_lb_m": float(vrp_m["makespan_lb_m"]),
        "vrp_total_cost_lb_m": float(vrp_m["total_cost_lb_m"]),
        "vrp_objective_lb_m": float(vrp_m["objective_lb_m"]),
        _CUOPT_FIELD: float(vrp_best_bound_m or 0.0),
    }

    if include_mapf:
        from VRP.core.constants import AUV_CRUISE_SPEED, SPACE_TIME_DWELL_S

        cs = AUV_CRUISE_SPEED if cruise_speed is None else cruise_speed
        ds = SPACE_TIME_DWELL_S if dwell_s is None else dwell_s
        mapf_s = vrp_mapf_lb_seconds(
            vrp_m,
            n_waypoints=n_waypoints,
            k=k,
            beta=beta,
            cruise_speed=cs,
            dwell_s=ds,
        )
        out["mapf_makespan_time_lb_s"] = float(mapf_s["makespan_time_lb_s"])
        out["mapf_total_time_lb_s"] = float(mapf_s["total_time_lb_s"])
        out["mapf_objective_time_lb_s"] = float(mapf_s["objective_time_lb_s"])

    # Sanity: cuOpt bound should be <= observed objective.
    if vrp_objective_value_m is not None and out[_CUOPT_FIELD] > vrp_objective_value_m + 1e-6:
        logger.warning(
            "cuOpt best_bound (%.3f) > observed objective (%.3f); reading wrong cuOpt field?",
            out[_CUOPT_FIELD],
            vrp_objective_value_m,
        )
    # Info: analytical LB should be <= cuOpt bound when both present.
    if (
        out[_CUOPT_FIELD] > 0
        and out["vrp_objective_lb_m"] > 0
        and out["vrp_objective_lb_m"] > out[_CUOPT_FIELD] + 1e-6
    ):
        logger.info(
            "Analytical LB (%.3f) beats cuOpt bound (%.3f) — cuOpt likely terminated very early.",
            out["vrp_objective_lb_m"],
            out[_CUOPT_FIELD],
        )
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Short cuOpt solve for dual-bound extraction
# ═══════════════════════════════════════════════════════════════════════════

# Defaults for the bound-only cuOpt run. cuOpt's MIP dual bound is valid
# at any termination, so we just use a very short time_limit and leave
# mip_gap at cuOpt's normal value. Most instances finish the root LP
# relaxation within a second, which gives a useful bound; harder
# instances get whatever B&B work fits. Bump the time_limit if bounds
# come back 0 on Duke-scale instances.
CUOPT_BOUND_TIME_LIMIT = 10
CUOPT_BOUND_MIP_GAP = 0.05


def extract_cuopt_bound(
    dist_matrix_cp,
    depot_indices: Sequence[int],
    k: int,
    beta: float,
    *,
    time_limit: int = CUOPT_BOUND_TIME_LIMIT,
    mip_gap: float = CUOPT_BOUND_MIP_GAP,
) -> float:
    """Run cuOpt briefly to extract a valid dual bound on the β-blended VRP
    objective (meters). Returns 0.0 on failure.

    The solve is intentionally short; the incumbent is discarded  --  we only
    want ``best_bound``.
    """
    try:
        from VRP.core.types import VRPBackend
        from VRP.vrp.vrp_solver import solve_vrp

        vrp_result = solve_vrp(
            dist_matrix=dist_matrix_cp,
            num_vehicles=k,
            depots=list(depot_indices),
            alpha=beta,
            backend=VRPBackend.CUOPT,
            time_limit=time_limit,
            mip_gap=mip_gap,
        )
        return float(vrp_result.best_bound)
    except Exception as exc:
        logger.warning("cuOpt bound extraction failed: %s", exc)
        return 0.0


def recompute_lbs(
    dist_matrix_cp,
    depot_indices: Sequence[int],
    k: int,
    n_waypoints: int,
    beta: float,
    *,
    include_mapf: bool = True,
    include_cuopt: bool = False,
    cuopt_time_limit: int = CUOPT_BOUND_TIME_LIMIT,
    cuopt_mip_gap: float = CUOPT_BOUND_MIP_GAP,
) -> dict:
    """Convenience wrapper used by ``--compute_lbs_only`` paths.

    Always computes analytical LBs. If ``include_cuopt`` is True, also runs
    a short cuOpt solve (``extract_cuopt_bound``) and injects the dual bound
    into the returned dict. No VRP solve when ``include_cuopt=False``.
    """
    best_bound = 0.0
    if include_cuopt:
        best_bound = extract_cuopt_bound(
            dist_matrix_cp,
            depot_indices,
            k,
            beta,
            time_limit=cuopt_time_limit,
            mip_gap=cuopt_mip_gap,
        )
    return compute_all_lbs(
        dist_matrix_cp,
        depot_indices,
        k,
        n_waypoints,
        beta,
        include_mapf=include_mapf,
        vrp_best_bound_m=best_bound,
    )


# ═══════════════════════════════════════════════════════════════════════════
# CSV sidecar (for e06)
# ═══════════════════════════════════════════════════════════════════════════


def lb_csv_fieldnames(key_cols: Sequence[str], *, include_mapf: bool = True):
    """Full column order for an LB CSV sidecar."""
    fields = _VRP_LB_FIELDS
    if include_mapf:
        fields = fields + _MAPF_LB_FIELDS
    fields = fields + (_CUOPT_FIELD,)
    return list(key_cols) + list(fields)


def append_lb_csv_row(
    csv_path: str,
    key_cols: Sequence[str],
    key_values: Mapping[str, object],
    lb: Mapping[str, float],
    *,
    include_mapf: bool = True,
) -> None:
    """Append one row to an LB CSV sidecar, creating the file (with header)
    if it does not exist yet.

    ``key_values`` must cover every column in ``key_cols``.
    """
    fieldnames = lb_csv_fieldnames(key_cols, include_mapf=include_mapf)
    new_file = not os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        row = {c: key_values[c] for c in key_cols}
        for k in fieldnames[len(key_cols) :]:
            row[k] = lb.get(k, 0.0)
        writer.writerow(row)


def write_lb_csv(
    csv_path: str,
    key_cols: Sequence[str],
    rows: Iterable[Mapping[str, object]],
    *,
    include_mapf: bool = True,
) -> None:
    """Rewrite the full CSV (header + rows). Each row must contain every
    column in ``key_cols`` and every LB field (missing fields default to 0).
    """
    fieldnames = lb_csv_fieldnames(key_cols, include_mapf=include_mapf)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            out_row = {c: r[c] for c in key_cols}
            for k in fieldnames[len(key_cols) :]:
                out_row[k] = r.get(k, 0.0)
            writer.writerow(out_row)


def load_lb_csv(
    csv_path: str,
    key_cols: Sequence[str],
    *,
    key_casts: Mapping[str, type] | None = None,
) -> dict:
    """Read a CSV sidecar -> dict keyed by tuple(key_cols).

    Keys are cast via ``key_casts`` (default: ``int`` for every key column).
    LB values are cast to float. Missing file => empty dict.
    """
    if not os.path.exists(csv_path):
        return {}
    key_casts = {c: int for c in key_cols} | dict(key_casts or {})
    out: dict = {}
    with open(csv_path) as f:
        reader = _csv.DictReader(f)
        for row in reader:
            try:
                key = tuple(key_casts[c](row[c]) for c in key_cols)
            except (KeyError, ValueError):
                continue
            lb = {k: float(row[k]) for k in ALL_LB_FIELDS if k in row}
            out[key] = lb
    return out


# ═══════════════════════════════════════════════════════════════════════════
# JSON sidecar (for e08, e09)
# ═══════════════════════════════════════════════════════════════════════════


def save_lb_json(stem: str, lb: Mapping[str, float]) -> None:
    """Save an LB dict to ``<stem>.json``. Creates parent dirs as needed."""
    path = stem if stem.endswith(".json") else stem + ".json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        _json.dump(dict(lb), f, indent=2)


def load_lb_json(stem: str) -> dict:
    """Load an LB JSON, or return an empty dict if missing."""
    path = stem if stem.endswith(".json") else stem + ".json"
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return _json.load(f)


def load_raw_lb_dir(
    raw_lb_dir: str,
    main_raw_dir: str,
) -> dict[str, dict]:
    """Load every ``<stem>.json`` in ``raw_lb_dir`` and key it by the stem.

    The stem is expected to match the main-results JSON filename so
    plotting code can left-join by stem.
    """
    out: dict[str, dict] = {}
    if not os.path.isdir(raw_lb_dir):
        return out
    for fname in sorted(os.listdir(raw_lb_dir)):
        if not fname.endswith(".json"):
            continue
        stem = fname[:-5]
        try:
            with open(os.path.join(raw_lb_dir, fname)) as f:
                out[stem] = _json.load(f)
        except Exception as exc:
            logger.warning("Failed to load LB sidecar %s: %s", fname, exc)
    return out
