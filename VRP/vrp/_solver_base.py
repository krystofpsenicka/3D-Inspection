"""Abstract base class for VRP solver backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import cupy as cp

from ..core.types import VRPResult


class VRPSolverBase(ABC):
    """Contract shared by all VRP solver backends."""

    @abstractmethod
    def solve(
        self,
        dist_matrix: cp.ndarray,
        num_vehicles: int,
        depots: list[int],
        alpha: float = 1.0,
        warm_start_routes: Optional[list[list[int]]] = None,
    ) -> VRPResult: ...
