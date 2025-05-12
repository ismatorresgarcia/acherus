"""
Root initialization file for importing HALA package and modules.
"""

from ._version import __version__
from .cli import create_cli_arguments
from .core.equations import EquationParameters
from .core.initial import initialize_envelope
from .core.ionization import calculate_ionization
from .core.laser import LaserInputParameters, LaserPulseParameters
from .core.materials import MaterialParameters
from .mesh.grid import GridParameters
from .results.paths import sim_dir
from .results.routines import profiler_report
from .results.store import OutputManager
from .solvers.base import SolverBase
from .solvers.fcn import SolverFCN
from .solvers.fss import SolverFSS

__all__ = [
    "__version__",
    "create_cli_arguments",
    "profiler_report",
    "OutputManager",
    "GridParameters",
    "LaserInputParameters",
    "LaserPulseParameters",
    "initialize_envelope",
    "calculate_ionization",
    "MaterialParameters",
    "EquationParameters",
    "sim_dir",
    "SolverBase",
    "SolverFSS",
    "SolverFCN",
]
