"""Shared types used across visibility and VRP packages."""

from enum import Enum


class Side(Enum):
    """Which side of the surface we are inspecting."""

    OUTSIDE = "outside"
    INSIDE = "inside"
