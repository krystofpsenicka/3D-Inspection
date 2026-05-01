"""Top-level visualization module  --  one class per visualized domain."""

from ._helpers import generate_tab20_colors, show_geometries
from .esdf import EsdfVisualizer
from .frustum_utils import create_frustum_lineset, create_viewpoint_geometry
from .model import ModelVisualizer
from .sampling import SamplingVisualizer
from .set_cover import SetCoverVisualizer
from .visibility_results import VisibilityVisualizer
