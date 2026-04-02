"""Shared helper functions for VRP solver backends."""

from __future__ import annotations

from typing import List, Union

import numpy as np


def normalise_depot(
    depot: int | list[int], num_vehicles: int
) -> list[int]:
    """Return a per-vehicle depot list regardless of input type."""
    if isinstance(depot, (list, tuple)):
        return [int(d) for d in depot]
    return [int(depot)] * num_vehicles


def compute_route_cost(
    routes: list[list[int]],
    dist_matrix: np.ndarray,
    depot: int | list[int] = 0,
) -> float:
    """Sum travel distances for all vehicles (including return to depot)."""
    depots = normalise_depot(depot, len(routes))
    total = 0.0
    for v, route in enumerate(routes):
        if not route:
            continue
        d = depots[v]
        full = [d] + list(route) + [d]
        for a, b in zip(full[:-1], full[1:]):
            total += float(dist_matrix[a, b])
    return total


def per_vehicle_costs(
    routes: list[list[int]],
    dist_matrix: np.ndarray,
    depot: int | list[int] = 0,
) -> list[float]:
    """Return per-vehicle travel distances."""
    depots = normalise_depot(depot, len(routes))
    costs: list[float] = []
    for v, route in enumerate(routes):
        c = 0.0
        if route:
            d = depots[v]
            full = [d] + list(route) + [d]
            for a, b in zip(full[:-1], full[1:]):
                c += float(dist_matrix[a, b])
        costs.append(c)
    return costs


def nearest_neighbor_warmstart(
    dist_matrix: np.ndarray,
    num_vehicles: int,
    depot: int | list[int],
) -> list[list[int]]:
    """Greedy nearest-neighbour heuristic for MIP warm-start. O(N^2)."""
    depots = normalise_depot(depot, num_vehicles)
    depot_set = set(depots)
    customers = [i for i in range(dist_matrix.shape[0]) if i not in depot_set]
    unvisited = set(customers)
    routes: list[list[int]] = [[] for _ in range(num_vehicles)]

    while unvisited:
        for v in range(num_vehicles):
            if not unvisited:
                break
            last = routes[v][-1] if routes[v] else depots[v]
            nearest = min(unvisited, key=lambda j: dist_matrix[last, j])
            routes[v].append(nearest)
            unvisited.remove(nearest)
    return routes
