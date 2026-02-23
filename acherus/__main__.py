"""
Entry point for Acherus
"""

from __future__ import annotations

import argparse
import os
import tomllib
from pathlib import Path

from ._version import __version__
from .config import ConfigOptions
from .data.store import OutputManager
from .functions.fft_backend import fft_manager
from .mesh.grid import Grid
from .physics.equation import Equation
from .physics.keldysh import KeldyshIonization
from .physics.laser import Laser
from .physics.medium import Medium
from .solvers.FCN import FCN
from .solvers.SSCN import SSCN


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the simulation entry point."""
    parser = argparse.ArgumentParser(description="Execute Acherus simulation")
    parser.add_argument("config", type=Path, help="TOML configuration file")
    parser.add_argument("--output", type=Path, help="Choose output directory")
    return parser.parse_args()


def load_config(config_path: Path) -> ConfigOptions:
    """Load and build simulation configuration from TOML file."""
    with open(config_path, "rb") as f:
        base_config = tomllib.load(f)

    return ConfigOptions.build(**base_config)


def init_solver(config, medium, laser, grid, equation, ionization, output):
    """Initialize the solver according to propagation method."""
    method = config.propagation_method.lower()

    solver_map = {
        "sscn": SSCN,
        "fcn": FCN,
    }
    solver_class = solver_map.get(method)
    if solver_class is None:
        raise ValueError(f"Invalid propagation method: '{config.propagation_method}'. ")

    return solver_class(config, medium, laser, grid, equation, ionization, output)


def main():
    """Main function."""
    print(f"Running Acherus v{__version__} for Python")

    args = parse_args()
    config = load_config(args.config)

    output_dir = args.output or config.data_output_path
    if output_dir:
        os.environ["ACHERUS_BASE_DIR"] = str(output_dir)

    grid = Grid(
        space_par=config.space_par, axis_par=config.axis_par, time_par=config.time_par
    )
    medium = Medium(medium_name=config.medium_name, medium_par=config.medium_par)
    laser = Laser(
        medium, grid, pulse_name=config.pulse_name, pulse_par=config.pulse_par
    )
    equation = Equation(medium, laser, grid)
    ionization = KeldyshIonization(
        medium,
        laser,
        model_name=config.ionization_model,
        params=config.ionization_model_par,
    )
    output = OutputManager(save_path=output_dir)

    solver = init_solver(
        config,
        medium,
        laser,
        grid,
        equation,
        ionization,
        output,
    )

    fft_manager.set_fft_backend(config.computing_backend)

    solver.propagate()
    output.save_results(solver, grid)

    print(f"Acherus simulation completed. Results saved to {output.save_path}\n")


if __name__ == "__main__":
    main()
