"""Isaac Sim visualization module  --  USD stage-builder counterpart of ``visualization/``.

Every public method adds prims to a caller-provided ``Usd.Stage`` and returns
the list of created prim paths.  No SimulationApp management, no render loops.
"""

from ._helpers import generate_tab20_colors
from .visibility import (
    EsdfVisualizer,
    ModelVisualizer,
    SamplingVisualizer,
    SetCoverVisualizer,
    VisibilityVisualizer,
    add_frustum_lineset,
    add_viewpoint_geometry,
)
from .vrp import ReplayVisualizer
