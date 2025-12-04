"""Entry point for running the package with 'python -m acherus'."""

from ._version import __version__
from .config import ConfigOptions
from .data.store import OutputManager
from .functions.fft_backend import fft_manager
from .mesh.grid import Grid
from .physics.equation import Equation
from .physics.keldysh import KeldyshIonization
from .physics.laser import Laser
from .physics.medium import Medium
from .solvers.nrFCN import nrFCN
from .solvers.nrSSCN import nrSSCN
from .solvers.rFCN import rFCN
from .solvers.rSSCN import rSSCN


def main():
    """Main function."""
    print(f"Running Acherus v{__version__} for Python")

    config = ConfigOptions.build(
        medium_parameters={
            "AIR": {
                "nonlinear_index": 3.0e-23,
                "energy_gap": 12.063,
                "collision_time": 3.5e-15,
                "neutral_density": 0.54e25,
                "initial_density": 1e9,
                "raman_partition": 0.5,
                "raman_response_time": 70e-15,
                "raman_rotational_time": 62.5e-15
            },
        },
        grid_parameters={
            "SPACE": {"nodes": 10000, "space_min": 0, "space_max": 10e-3},
            "AXIS": {"nodes": 4000, "axis_min": 0, "axis_max": 2.2, "snapshots": 3},
            "TIME": {"nodes": 4096, "time_min": -3e-12, "time_max": 3e-12}
        },
        pulse_parameters={
            "GAUSSIAN": {
                "wavelength": 800e-9,
                "waist": 3.57e-3,
                "duration": 1274e-15,
                "energy": 0.1,
                "chirp": 0,
                "focal_length": 2,
                "gauss_order": 2
            },
        },
        density_solver={
            "RK45": {
                "rtol": 1e-9,
                #"atol": 1e-7
            },
        },
        propagation_solver="FCN",
        ionization_model={
            "KELDYSH": {
            },
        },
        computing_backend="CPU"
    )

    grid = Grid(
        space_par=config.space_par,
        axis_par=config.axis_par,
        time_par=config.time_par
    )
    medium = Medium(medium_name=config.medium_name, medium_par=config.medium_par)
    laser = Laser(
        medium, grid, pulse_name=config.pulse_name, pulse_par=config.pulse_par
    )
    eqn = Equation(medium, laser, grid)
    ion = KeldyshIonization(medium, laser, model_name=config.ionization_model, params=config.ionization_model_par)
    output = OutputManager()

    if config.propagation_method == "sscn" and config.medium_par.raman_partition is not None:
        solver = rSSCN(
            config,
            medium,
            laser,
            grid,
            eqn,
            ion,
            output
        )
    elif config.propagation_method == "sscn" and config.medium_par.raman_partition is None:
        solver = nrSSCN(
            config,
            medium,
            laser,
            grid,
            eqn,
            ion,
            output
        )
    elif config.propagation_method == "fcn" and config.medium_par.raman_partition is not None:
        solver = rFCN(
            config,
            medium,
            laser,
            grid,
            eqn,
            ion,
            output
        )
    elif config.propagation_method == "fcn" and config.medium_par.raman_partition is None:
        solver = nrFCN(
            config,
            medium,
            laser,
            grid,
            eqn,
            ion,
            output
        )
    else:
        raise ValueError(
            f"Invalid propagation method: '{config.propagation_method}'. "
        )
    # ... future solvers to be added in the future!

    # Initialize FFT algorithm
    fft_manager.set_fft_backend(config.computing_backend)

    # Run simulation
    solver.propagate()

    # Save final propagation results
    output.save_results(solver, grid)


if __name__ == "__main__":
    main()
