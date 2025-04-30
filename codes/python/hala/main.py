"""Main entry point for the HALA package."""

import cProfile

from ._version import __version__
from .cli import create_cli_arguments
from .core.equations import EquationParameters
from .core.laser import LaserPulseParameters
from .core.materials import MaterialParameters
from .mesh.grid import GridParameters
from .results.routines import profiler_report
from .results.store import OutputManager
from .solvers.fcn import SolverFCN
from .solvers.fss import SolverFSS


def main():
    """Main function."""
    # Initialize CLI arguments
    args = create_cli_arguments()

    # Print package version
    print(f"Running HALA v{__version__}")

    # Initialize classes
    material = MaterialParameters(material_opt=args.material)
    grid = GridParameters()
    laser = LaserPulseParameters(
        material, pulse_opt=args.pulse, gauss_opt=args.gaussian_order
    )
    eqn = EquationParameters(material, laser)

    # Initialize profiler
    profiler = cProfile.Profile()
    profiler.enable()

    # Choose solver
    if args.solver == "fss":
        solver = SolverFSS(material, laser, grid, eqn, method_opt=args.method)
    else:  # fcn
        solver = SolverFCN(material, laser, grid, eqn, method_opt=args.method)
    # ... future solvers to be added in the future!

    # Run simulation
    solver.propagate()

    # Stop profiler
    profiler.disable()

    # Initialize and run data saving class
    output_manager = OutputManager()
    output_manager.save_results(solver, grid)

    # Generate profiler report
    profiler_report(profiler)


if __name__ == "__main__":
    main()
