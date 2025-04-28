"""Main entry point for the HALA package."""

from ._version import __version__
from .cli import create_cli_arguments
from .core.constants import Constants
from .core.equations import NEEParameters
from .core.laser import LaserPulseParameters
from .core.materials import MediumParameters
from .grid.grid import GridParameters
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
    const = Constants()
    medium = MediumParameters(medium_opt=args.medium)
    grid = GridParameters(const)
    laser = LaserPulseParameters(
        const, medium, pulse_opt=args.pulse, gauss_opt=args.gauss_order
    )
    nee = NEEParameters(const, medium, laser)

    # Choose solver
    if args.solver == "fss":
        solver = SolverFSS(const, medium, laser, grid, nee, method_opt=args.method)
    else:  # fcn
        solver = SolverFCN(const, medium, laser, grid, nee, method_opt=args.method)
    # ... future solvers to be added in the future!

    # Run simulation
    solver.propagate()

    # Initialize and run data saving class
    output_manager = OutputManager()
    output_manager.save_results(solver, grid)


if __name__ == "__main__":
    main()
