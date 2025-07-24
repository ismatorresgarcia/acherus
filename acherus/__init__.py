"""
Root initialization file for importing Acherus package and modules.
"""

from ._version import __version__
from .config import config_options
from .data.routines import profiler_log
from .data.store import OutputManager
from .functions.ionization import compute_ionization
from .mesh.grid import GridParameters
from .physics.equations import EquationParameters
from .physics.initial_beam import initialize_envelope
from .physics.materials import Material, MaterialParameters
from .physics.optics import LaserParameters
from .solvers.base import SolverBase
from .solvers.fcn import SolverFCN
from .solvers.fss import SolverFSS

__all__ = [
    "__version__",
    "config_options",
    "profiler_log",
    "OutputManager",
    "GridParameters",
    "LaserParameters",
    "initialize_envelope",
    "compute_ionization",
    "Material",
    "MaterialParameters",
    "EquationParameters",
    "SolverBase",
    "SolverFSS",
    "SolverFCN",
]
