"""Entry point for running the package with 'python -m acherus'."""

from ._version import __version__
from .config import ConfigOptions
from .data.store import OutputManager
from .functions.fft_backend import fft_manager
from .mesh.grid import Grid
from .physics.dispersion import MediumDispersion
from .physics.equation import Equation
from .physics.keldysh import KeldyshIonization
from .physics.laser import Laser
from .physics.media import MediumParameters
from .solvers.fcn import SolverFCN
from .solvers.sscn import SolverSSCN


def main():
    """Main function."""
    print(f"Running Acherus v{__version__} for Python")

    config = ConfigOptions.build(
        medium_name="oxygen_800",
        grid_parameters={
            "space": {"nodes": 10000, "space_min": 0, "space_max": 10e-3},
            "axis": {"nodes": 4000, "axis_min": 0, "axis_max": 2.2, "snapshots": 3},
            "time": {"nodes": 4096, "time_min": -3e-12, "time_max": 3e-12},
        },
        pulse_parameters={
            "gaussian": {
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
                "ini_step": 5e-15,
                "rtol": 1e-9,
                "atol": 1e-6
            },
        },
        propagation_solver="FCN",
        ionization_model={
            "PPTG": {
                "wavelength": 800e-9,
                "energy_gap": 12.063,
            },
        },
        computing_backend="CPU"
    )

    grid = Grid(
        space_par=config.space_par,
        axis_par=config.axis_par,
        time_par=config.time_par
    )
    disp = MediumDispersion(medium_name=config.medium_name)
    medium = MediumParameters(medium_name=config.medium_name)
    laser = Laser(
        disp, medium, grid, pulse_name=config.pulse_name, pulse_par=config.pulse_par
    )
    eqn = Equation(medium, laser, grid)
    ion = KeldyshIonization(disp, model_name=config.ionization_model, params=config.ionization_model_par)
    output = OutputManager()

    if config.propagation_method == "SSCN":
        solver = SolverSSCN(
            config,
            disp,
            medium,
            laser,
            grid,
            eqn,
            ion,
            output
        )
    elif config.propagation_method == "FCN":
        solver = SolverFCN(
            config,
            disp,
            medium,
            laser,
            grid,
            eqn,
            ion,
            output
        )
    else:
        raise ValueError(
            f"Not available propagation method: '{config.propagation_method}'. "
            f"Available methods are: 'SSCN' or 'FCN'."
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
