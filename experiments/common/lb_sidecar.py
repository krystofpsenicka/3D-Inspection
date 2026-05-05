"""Lower-bound sidecar helpers.

Sidecars keep LBs in their own files next to main results so we can
re-solve them without re-running the
expensive VRP+MAPF pipeline.

- CSV experiments (e06): ``lower_bounds.csv`` next to ``results.csv``.
- JSON experiments (e08/e09): ``raw_lb/<same_stem>.json``.

Schema (subset written per-experiment):

    vrp_makespan_lb_m, vrp_total_cost_lb_m, vrp_objective_lb_m,
    mapf_makespan_time_lb_s, mapf_total_time_lb_s, mapf_objective_time_lb_s,
    vrp_objective_best_bound_m    # cuOpt dual bound (0.0 = unavailable)

Key columns identifying the matching main-result row are passed by the caller.
"""

from __future__ import annotations

import csv as _csv
import json as _json
import logging
import os
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

# cupy used only by compute_all_lbs/recompute_lbs. Importing this module for
# load_*/--plots_only must work without GPU, so cupy is deferred.
try:
    import cupy as cp
except ImportError:
    cp = None

from experiments.common.lower_bounds import vrp_lb_meters, vrp_mapf_lb_seconds

logger = logging.getLogger(__name__)


# ── Field definitions ──────────────────────────────────────────────────────

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


# ── Core computation ───────────────────────────────────────────────────────


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
    """Compute analytical LBs from distance matrix + instance params,
    merged with an optional cuOpt dual bound.

    cruise_speed / dwell_s default to VRP.core.constants when None.
    vrp_objective_value_m is only used for a sanity warning.

    Returns dict with keys in ALL_LB_FIELDS (or VRP_ONLY_LB_FIELDS when
    include_mapf=False).
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
            vrp_m, n_waypoints=n_waypoints, k=k, beta=beta,
            cruise_speed=cs, dwell_s=ds,
        )
        out["mapf_makespan_time_lb_s"] = float(mapf_s["makespan_time_lb_s"])
        out["mapf_total_time_lb_s"] = float(mapf_s["total_time_lb_s"])
        out["mapf_objective_time_lb_s"] = float(mapf_s["objective_time_lb_s"])

    # Sanity: cuOpt bound should be <= observed objective.
    if vrp_objective_value_m is not None and out[_CUOPT_FIELD] > vrp_objective_value_m + 1e-6:
        logger.warning(
            "cuOpt best_bound (%.3f) > observed objective (%.3f); reading wrong cuOpt field?",
            out[_CUOPT_FIELD], vrp_objective_value_m,
        )
    # Info: analytical LB > cuOpt bound means cuOpt terminated very early.
    if (
        out[_CUOPT_FIELD] > 0
        and out["vrp_objective_lb_m"] > 0
        and out["vrp_objective_lb_m"] > out[_CUOPT_FIELD] + 1e-6
    ):
        logger.info(
            "Analytical LB (%.3f) beats cuOpt bound (%.3f) — cuOpt likely terminated very early.",
            out["vrp_objective_lb_m"], out[_CUOPT_FIELD],
        )
    return out


# ── Short cuOpt solve for dual-bound extraction ────────────────────────────

# cuOpt's MIP dual bound is valid at any termination, so we use a very short
# time_limit.
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
    """Brief cuOpt run to extract a valid dual bound (meters). 0.0 on failure."""
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
    """Used by --compute_lbs_only paths. Always computes analytical LBs;
    optionally adds a cuOpt dual bound.
    """
    best_bound = 0.0
    if include_cuopt:
        best_bound = extract_cuopt_bound(
            dist_matrix_cp, depot_indices, k, beta,
            time_limit=cuopt_time_limit, mip_gap=cuopt_mip_gap,
        )
    return compute_all_lbs(
        dist_matrix_cp, depot_indices, k, n_waypoints, beta,
        include_mapf=include_mapf, vrp_best_bound_m=best_bound,
    )


# ── CSV sidecar (for e06) ──────────────────────────────────────────────────


def lb_csv_fieldnames(key_cols: Sequence[str], *, include_mapf: bool = True):
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
    """Append one row, creating file with header if missing.
    key_values must cover every column in key_cols.
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
    """Rewrite full CSV. Each row needs every key_col; missing LB fields => 0."""
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
    """Read sidecar -> dict keyed by tuple(key_cols). Missing file => {}.
    Keys cast via key_casts (default int per key column); LB values cast to float.
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


# ── JSON sidecar (for e08, e09) ────────────────────────────────────────────


def save_lb_json(stem: str, lb: Mapping[str, float]) -> None:
    path = stem if stem.endswith(".json") else stem + ".json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        _json.dump(dict(lb), f, indent=2)


def load_lb_json(stem: str) -> dict:
    path = stem if stem.endswith(".json") else stem + ".json"
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return _json.load(f)


def load_raw_lb_dir(raw_lb_dir: str, main_raw_dir: str) -> dict[str, dict]:
    """Load every <stem>.json keyed by stem (matches main-results filename)."""
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
