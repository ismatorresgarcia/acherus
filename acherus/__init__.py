"""
Root initialization file for importing Acherus package and modules.
"""

from ._version import __version__
from .config import ConfigOptions
from .data.diagnostics import profiler_log
from .data.store import OutputManager
from .functions.interp_w import compute_ionization
from .mesh.grid import Grid
from .physics.equation import Equation
from .physics.laser import Laser
from .physics.media import Medium, MediumParameters
from .solvers.base import SolverBase
from .solvers.fcn import SolverFCN
from .solvers.sscn import SolverSSCN

__all__ = [
    "__version__",
    "ConfigOptions",
    "profiler_log",
    "OutputManager",
    "Grid",
    "Laser",
    "compute_ionization",
    "Medium",
    "MediumParameters",
    "Equation",
    "SolverBase",
    "SolverSSCN",
    "SolverFCN",
]
