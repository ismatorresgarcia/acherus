"""Main entry point for the AiurPy package."""

import cProfile

from ._version import __version__
from .config import config_options
from .core.equations import EquationParameters
from .core.laser import LaserPulseParameters
from .core.materials import MaterialParameters
from .mesh.grid import GridParameters
from .results.routines import profiler_log
from .results.store import OutputManager
from .solvers.fcn import SolverFCN
from .solvers.fss import SolverFSS


def main():
    """Main function."""
    # Initialize configuration options
    config = config_options()

    # Print package version
    print(f"Running AiurPy v{__version__} simulation ")

    # Initialize classes
    material = MaterialParameters(material_opt=config["material"])
    grid = GridParameters()
    laser = LaserPulseParameters(
        material, pulse_opt=config["pulse"], gauss_opt=config["gaussian_order"]
    )
    eqn = EquationParameters(material, laser)

    # Choose solver
    if config["solver"] == "fss":
        solver = SolverFSS(
            material,
            laser,
            grid,
            eqn,
            method_opt=config["method"],
            ion_model=config["ion_model"],
        )
    elif config["solver"] == "fcn":
        solver = SolverFCN(
            material,
            laser,
            grid,
            eqn,
            method_opt=config["method"],
            ion_model=config["ion_model"],
        )
    else:
        raise ValueError(
            f"Not available solver: '{config["solver"]}'. "
            f"Available solvers are: 'fss' or 'fcn'."
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
