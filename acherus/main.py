"""Main entry point for the Acherus package."""

import cProfile

from ._version import __version__
from .config import config_options
from .data.routines import profiler_log
from .data.store import OutputManager
from .mesh.grid import GridParameters
from .physics.equations import EquationParameters
from .physics.media import MediumParameters
from .physics.optics import LaserParameters
from .solvers.fcn import SolverFCN
from .solvers.fss import SolverFSS


def main():
    """Main function."""
    # Initialize configuration options
    config = config_options()

    # Print package version
    print(f"Running Acherus v{__version__} for Python")

    # Initialize classes
    grid = GridParameters()
    medium = MediumParameters(medium_opt=config["medium"])
    if config["pulse"] == "gaussian":
        laser = LaserParameters(
            medium, pulse_opt=config["pulse"], gauss_opt=config["gauss_n"]
        )
    else:
        raise ValueError(
            f"Invalid pulse type: '{config['pulse']}. "
            f"Choose 'gaussian' or 'to_be_defined'."
        )
    eqn = EquationParameters(medium, laser, grid)

    # Initialize solver
    if config["solver"] == "FSS":
        solver = SolverFSS(
            medium,
            laser,
            grid,
            eqn,
            method_d_opt=config["method_d"],
            method_nl_opt=config["method_nl"],
            ion_model=config["ion_model"],
        )
    elif config["solver"] == "FCN":
        solver = SolverFCN(
            medium,
            laser,
            grid,
            eqn,
            method_d_opt=config["method_d"],
            method_nl_opt=config["method_nl"],
            ion_model=config["ion_model"],
        )
    else:
        raise ValueError(
            f"Not available solver: '{config["solver"]}'. "
            f"Available solvers are: 'FSS' or 'FCN'."
        )
    # Add more solvers here as needed
    # ... future solvers to be added in the future!

    # Initialize profiler
    profiler = cProfile.Profile()
    profiler.enable()

    # Run simulation
    solver.propagate()

    # Stop profiler
    profiler.disable()

    # Initialize and run data saving class
    output_manager = OutputManager()
    output_manager.save_results(solver, grid)

    # Generate profiler report
    profiler_log(profiler)


if __name__ == "__main__":
    main()
