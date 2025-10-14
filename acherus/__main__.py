"""Entry point for running the package with 'python -m acherus'."""

import cProfile

from scipy.fft import set_backend

from ._version import __version__
from .config import ConfigOptions
from .data.routines import profiler_log
from .data.store import OutputManager
from .functions.fft_backend import fft_manager
from .mesh.grid import Grid
from .physics.equation import Equation
from .physics.laser import Laser
from .physics.media import MediumParameters
from .solvers.fcn import SolverFCN
from .solvers.sscn import SolverSSCN


def main():
    """Main function."""
    # Print package version
    print(f"Running Acherus v{__version__} for Python")

    # Initialize classes
    config = ConfigOptions.build(
        medium_name="oxygen_800",
        grid_parameters={
            "space": {
                "nodes": 10000,
                "space_min": 0,
                "space_max": 10e-3
            },
            "axis": {
                "nodes": 4000,
                "axis_min": 0,
                "axis_max": 2.2,
                "snapshots": 1
            },
            "time": {
                "nodes": 4096,
                "time_min": -3e-12,
                "time_max": 3e-12
            },
        },
        pulse_shape="gaussian",
        pulse_parameters={
            "wavelength": 800e-9,
            "waist": 3.57e-3,
            "duration": 1274e-15,
            "energy": 0.1,
            "chirp": 0,
            "focal_length": 2,
            "gauss_order": 2
        },
        density_method="RK45",
        nonlinear_method="AB2",
        solver_scheme="FCN",
        ionization_model="PPT",
        computing_backend="CPU"
    )

    grid = Grid(
        space_par=config.space_grid,
        axis_par=config.axis_grid,
        time_par=config.time_grid
    )
    medium = MediumParameters(medium_opt=config.medium_name)
    laser = Laser(
        medium, grid, pulse_typ=config.pulse_shape, pulse_par=config.pulse_config
    )
    eqn = Equation(medium, laser, grid)

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

    # Initialize FFT algorithm
    fft_manager.set_fft_backend(config.computing_backend)

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
