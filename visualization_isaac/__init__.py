"""Isaac Sim visualization module — USD stage-builder counterpart of ``visualization/``.

Every public method adds prims to a caller-provided ``Usd.Stage`` and returns
the list of created prim paths.  No SimulationApp management, no render loops.
"""

from .frustum_utils import add_frustum_lineset, add_viewpoint_geometry
from ._helpers import generate_tab20_colors
from .model import ModelVisualizer
from .visibility_results import VisibilityVisualizer
from .set_cover import SetCoverVisualizer
from .sampling import SamplingVisualizer
from .esdf import EsdfVisualizer
