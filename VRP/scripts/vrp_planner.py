"""Deprecated — use VRP.core.vrp_orchestrator instead."""

from VRP.core.vrp_orchestrator import VRPFeedbackOrchestrator  # noqa: F401
from VRP.core.geometry import compute_start_grid  # noqa: F401

VRPPipeline = VRPFeedbackOrchestrator
_compute_start_grid = compute_start_grid
