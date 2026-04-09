"""Shared helper functions for VRP solver backends."""

from __future__ import annotations

import cupy as cp


def per_vehicle_costs(
    routes: list[list[int]],
    dist_matrix: cp.ndarray,
    depots: list[int],
) -> list[float]:
    """Return per-vehicle travel distances (GPU-vectorized).

    Each vehicle's cost is the sum of arc costs along
    depot -> route[0] -> ... -> route[-1] -> depot.

    Args:
        routes: Per-vehicle customer node lists (excluding depots).
        dist_matrix: (N, N) CuPy distance matrix.
        depots: Per-vehicle depot indices.
    """
    costs: list[float] = []
    for v, route in enumerate(routes):
        if not route:
            costs.append(0.0)
            continue
        d = depots[v]
        full = [d] + list(route) + [d]
        from_nodes = cp.array(full[:-1], dtype=cp.intp)
        to_nodes = cp.array(full[1:], dtype=cp.intp)
        costs.append(float(dist_matrix[from_nodes, to_nodes].sum()))
    return costs


def compute_route_cost(
    routes: list[list[int]],
    dist_matrix: cp.ndarray,
    depots: list[int],
) -> float:
    """Sum travel distances for all vehicles (including return to depot)."""
    return sum(per_vehicle_costs(routes, dist_matrix, depots))


def nearest_neighbor_warmstart(
    dist_matrix: cp.ndarray,
    num_vehicles: int,
    depots: list[int],
) -> list[list[int]]:
    """Greedy nearest-neighbour heuristic for MIP warm-start."""
    N = dist_matrix.shape[0]
    depot_set = set(depots)

    visited = cp.zeros(N, dtype=cp.bool_)
    for d in depot_set:
        visited[d] = True

    routes: list[list[int]] = [[] for _ in range(num_vehicles)]
    current_pos = cp.array(depots, dtype=cp.int32)

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
