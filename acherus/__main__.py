"""Entry point for running the package with 'python -m acherus'."""

import cProfile

from ._version import __version__
from .config import ConfigOptions
from .data.routines import profiler_log
from .data.store import OutputManager
from .mesh.grid import GridParameters
from .physics.equations import EquationParameters
from .physics.media import MediumParameters
from .physics.optics import LaserParameters
from .solvers.fcn import SolverFCN
from .solvers.sscn import SolverSSCN


def main():
    """Main function."""
    # Print package version
    print(f"Running Acherus v{__version__} for Python")

    # Initialize classes
    config = ConfigOptions(
        medium_name="Oxygen_800",
        pulse_shape="Gaussian",
        gauss_order=2,
        density_method="RK45",
        nonlinear_method="AB2",
        solver_scheme="FCN",
        ionization_model="PPT",
        compute_engine="CPU"
    )

    grid = GridParameters()
    medium = MediumParameters(medium_opt=config.medium_name)
    if config.pulse_shape == "Gaussian":
        laser = LaserParameters(
            medium, pulse_opt=config.pulse_shape, gauss_opt=config.gauss_order
        )
    else:
        raise ValueError(
            f"Invalid pulse type: '{config.pulse_shape}. "
            f"Choose 'Gaussian' or 'to_be_defined'."
        )
    eqn = EquationParameters(medium, laser, grid)

    # Initialize solver
    if config.solver_scheme == "SSCN":
        solver = SolverSSCN(
            config,
            medium,
            laser,
            grid,
            eqn,
        )
    elif config.solver_scheme == "FCN":
        solver = SolverFCN(
            config,
            medium,
            laser,
            grid,
            eqn,
        )
    else:
        raise ValueError(
            f"Not available solver: '{config.solver_scheme}'. "
            f"Available solvers are: 'SSCN' or 'FCN'."
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
