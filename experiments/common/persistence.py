"""Save/load experiment results as JSON + NPZ."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np


def _to_serializable(v: Any) -> Any:
    if hasattr(v, "get"):  # CuPy array
        return v.get().tolist()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def save_run_result(result: dict, path: str) -> None:
    """Save dict as JSON + NPZ. Arrays go to {path}.npz, scalars to {path}.json."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    arrays = {}
    scalars = {}
    for k, v in result.items():
        if hasattr(v, "get"):  # CuPy
            arrays[k] = v.get()
        elif isinstance(v, np.ndarray):
            arrays[k] = v
        else:
            scalars[k] = _to_serializable(v)

    if arrays:
        np.savez_compressed(path + ".npz", **arrays)
    with open(path + ".json", "w") as f:
        json.dump(scalars, f, indent=2, default=str)


def load_run_result(path: str) -> dict:
    result = {}
    json_path = path + ".json"
    npz_path = path + ".npz"

    if os.path.exists(json_path):
        with open(json_path) as f:
            result.update(json.load(f))
    if os.path.exists(npz_path):
        with np.load(npz_path) as data:
            for k in data.files:
                result[k] = data[k]
    return result


def aggregate_to_csv(results_dir: str, output_csv: str) -> None:
    """Flatten all per-run JSON files in a directory to CSV."""
    import csv

    rows = []
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(results_dir, fname)) as f:
            rows.append(json.load(f))

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
