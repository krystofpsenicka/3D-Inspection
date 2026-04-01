"""Top-level visualization module — one class per visualized domain."""

from .frustum_utils import create_frustum_lineset, create_viewpoint_geometry
from ._helpers import generate_tab20_colors, show_geometries
from .model import ModelVisualizer
from .visibility_results import VisibilityVisualizer
from .set_cover import SetCoverVisualizer
from .sampling import SamplingVisualizer
from .esdf import EsdfVisualizer
