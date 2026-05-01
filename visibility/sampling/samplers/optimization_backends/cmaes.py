"""CMA-ES optimization backend using EvoTorch."""

import logging

import cupy as cp
import torch
from evotorch import Problem
from evotorch.algorithms import CMAES

from ..optimizing import OptimizationBackend

logger = logging.getLogger(__name__)


class _EvoTorchObjective(Problem):
    """Thin EvoTorch adapter wrapping a CuPy-based objective function."""

    def __init__(self, objective_fn, n_dims, popsize):
        self._objective_fn = objective_fn
        super().__init__(
            objective_sense="min",
            solution_length=n_dims,
            initial_bounds=(0.0, 1.0),
            dtype=torch.float32,
            device="cuda",
        )

    def _evaluate_batch(self, solutions):
        vals = solutions.values  # (pop, D) torch CUDA tensor
        vals_gpu = cp.asarray(vals)
        scores_gpu = self._objective_fn(vals_gpu)
        solutions.set_evals(torch.as_tensor(scores_gpu, device="cuda"))


class CMAESBackend(OptimizationBackend):
    """CMA-ES optimization via EvoTorch."""

    def optimize(
        self, objective_fn, n_dims, popsize, maxiter, verbose=False, center_init=None
    ) -> cp.ndarray:
        """Run CMA-ES in [0,1]^n_dims.

        Args:
            objective_fn: callable ``(pop, D) CuPy -> (pop,) CuPy`` scores.
            n_dims:       search dimensions.
            popsize:      population size.
            maxiter:      max generations.
            verbose:      log per-generation progress.
            center_init:  optional ``(D,)`` CuPy array  --  initial centre of the
                          search distribution in [0,1]^D.

        Returns:
            (D,) CuPy array  --  best solution found.
        """
        problem = _EvoTorchObjective(objective_fn, n_dims, popsize)
        cmaes_kwargs = dict(stdev_init=1.0 / 3.0, popsize=popsize)
        if center_init is not None:
            cmaes_kwargs["center_init"] = torch.as_tensor(center_init, device="cuda")
        searcher = CMAES(problem, **cmaes_kwargs)

        for gen_i in range(maxiter):
            searcher.step()
            if verbose and (gen_i + 1) % 5 == 0:
                try:
                    best_val = float(searcher.status["pop_best_eval"])
                    logger.info("  CMA-ES gen %d: best_eval=%.4f", gen_i + 1, best_val)
                except (KeyError, TypeError):
                    pass

        best_torch = searcher.status["pop_best"].values  # CUDA tensor
        return cp.asarray(best_torch)
