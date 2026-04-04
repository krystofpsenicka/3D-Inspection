"""Base class for iterative set-cover optimizers."""

import logging
import cupy as cp
from abc import ABC, abstractmethod
import time
from typing import List, Optional, Tuple

from ..core.types import OptimizationResult
from ..core.utils import compute_redundancy
from ..core.constants import DEFAULT_TARGET_COVERAGE, DEFAULT_MAX_VIEWPOINTS

logger = logging.getLogger(__name__)


class IterativeSetCoverOptimizer(ABC):
    """Abstract base for iterative set-cover optimization.

    Subclasses implement ``select_next`` and ``commit_selection`` for the
    specific scoring strategy. All arrays exchanged through these methods
    are CuPy arrays on GPU for simplicity (the final pipeline will use gpu 
    implementations).
    """

    @abstractmethod
    def select_next(self) -> Optional[Tuple[cp.ndarray, cp.ndarray, cp.ndarray]]:
        """Select the next best candidate viewpoint.

        Returns ``(position, rotation, visible_indices)`` as CuPy arrays,
        or ``None`` if no candidate provides new coverage.

        Does **not** mutate internal state — call ``commit_selection``
        afterwards to apply the update.
        """

    @abstractmethod
    def commit_selection(self, visible_indices: cp.ndarray):
        """Commit the last selection: update the uncovered mask with
        *visible_indices* and deactivate the selected candidate."""

    def optimize(self, target_coverage: float = DEFAULT_TARGET_COVERAGE,
                 max_viewpoints: int = DEFAULT_MAX_VIEWPOINTS) -> OptimizationResult:
        """Run the full iterative set-cover loop."""
        start_time = time.perf_counter()

        sel_positions: List[cp.ndarray] = []
        sel_rotations: List[cp.ndarray] = []
        sel_vis_rows: List[cp.ndarray] = []  # (M,) uint8 rows for visibility_map

        covered_mask = cp.zeros(self.num_points, dtype=cp.bool_)
        total_covered = 0
        target_count = int(target_coverage * self.num_points)

        while total_covered < target_count and len(sel_positions) < max_viewpoints:
            result = self.select_next()
            if result is None:
                break
            pos, rot, vis = result
            self.commit_selection(vis)

            sel_positions.append(pos)
            sel_rotations.append(rot)

            # Build a (M,) uint8 row for this viewpoint
            row = cp.zeros(self.num_points, dtype=cp.uint8)
            if len(vis) > 0:
                row[vis] = 1
                covered_mask[vis] = True
            sel_vis_rows.append(row)

            total_covered = int(covered_mask.sum())
            coverage = total_covered / self.num_points if self.num_points > 0 else 0.0
            logger.info("  [SetCover] VP %d: +%d pts, coverage=%.1f%%",
                        len(sel_positions), len(vis), coverage * 100)

        optimization_time = time.perf_counter() - start_time
        coverage = int(covered_mask.sum()) / self.num_points if self.num_points > 0 else 0.0

        if sel_positions:
            positions = cp.stack(sel_positions)
            rotations = cp.stack(sel_rotations)
            visibility_map = cp.stack(sel_vis_rows)
        else:
            positions = cp.empty((0, 3), dtype=cp.float32)
            rotations = cp.empty((0, 3, 3), dtype=cp.float32)
            visibility_map = cp.empty((0, self.num_points), dtype=cp.uint8)

        return OptimizationResult(
            positions=positions,
            rotations=rotations,
            visibility_map=visibility_map,
            total_coverage=coverage,
            num_viewpoints=len(sel_positions),
            redundancy=compute_redundancy(visibility_map),
            optimization_time=optimization_time,
        )
