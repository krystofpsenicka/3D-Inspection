"""Shared helper functions for VRP solver backends."""

from __future__ import annotations

import cupy as cp
import numpy as np


def compute_route_cost(
    routes: list[list[int]],
    dist_matrix: np.ndarray,
    depot: list[int],
) -> float:
    """Sum travel distances for all vehicles (including return to depot)."""
    total = 0.0
    for v, route in enumerate(routes):
        if not route:
            continue
        d = depot[v]
        full = [d] + list(route) + [d]
        for a, b in zip(full[:-1], full[1:]):
            total += float(dist_matrix[a, b])
    return total


def per_vehicle_costs(
    routes: list[list[int]],
    dist_matrix: np.ndarray,
    depot: list[int],
) -> list[float]:
    """Return per-vehicle travel distances."""
    costs: list[float] = []
    for v, route in enumerate(routes):
        c = 0.0
        if route:
            d = depot[v]
            full = [d] + list(route) + [d]
            for a, b in zip(full[:-1], full[1:]):
                c += float(dist_matrix[a, b])
        costs.append(c)
    return costs


def nearest_neighbor_warmstart(
    dist_matrix: cp.ndarray,
    num_vehicles: int,
    depot: list[int],
) -> list[list[int]]:
    """Greedy nearest-neighbour heuristic for MIP warm-start."""
    N = dist_matrix.shape[0]
    depot_set = set(depot)

    visited = cp.zeros(N, dtype=cp.bool_)
    for d in depot_set:
        visited[d] = True

    routes: list[list[int]] = [[] for _ in range(num_vehicles)]
    current_pos = cp.array(depot, dtype=cp.int32)

    remaining = N - len(depot_set)
    while remaining > 0:
        for v in range(num_vehicles):
            if remaining == 0:
                break
            dists_from_current = dist_matrix[int(current_pos[v])]
            masked = cp.where(visited, cp.inf, dists_from_current)
            nearest = int(cp.argmin(masked))
            routes[v].append(nearest)
            visited[nearest] = True
            current_pos[v] = nearest
            remaining -= 1

    return routes
