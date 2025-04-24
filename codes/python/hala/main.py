"""Main entry point for the HALA package."""

from hala import __version__
from hala.cli import create_cli_arguments
from hala.diagnostics.output import OutputManager
from hala.domain.grid import GridParameters
from hala.physics.constants import Constants
from hala.physics.laser import LaserPulseParameters
from hala.physics.medium import MediumParameters
from hala.physics.nee import NEEParameters
from hala.solvers.solver_fcn import SolverFCN
from hala.solvers.solver_fss import SolverFSS


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
    if args.solver.upper() == "FSS":
        solver = SolverFSS(const, medium, laser, grid, nee, method_opt=args.method)
    else:  # fcn
        solver = SolverFCN(const, medium, laser, grid, nee, method_opt=args.method)
    # ... future solvers to be added in the future!

    # Run simulation
    solver.propagate()

    # Initialize and run data saving class
    output_manager = OutputManager()
    output_manager.save_diagnostics(solver, grid)


if __name__ == "__main__":
    main()
