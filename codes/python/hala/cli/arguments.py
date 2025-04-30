"""Command line argument parser module."""

import argparse

from .. import __version__
from ..core.materials import MaterialParameters


def create_cli_arguments():
    """Parse command line arguments with argparse."""
    parser = argparse.ArgumentParser(
        description="HALA v{__version__} Python package "
        "(2+1)D with cylindrical symmetry numerical solver for "
        "atmospheric lasing and nitrogen filaments. "
    )
    parser.add_argument(
        "-m",
        "--material",
        choices=MaterialParameters.materials_list,
        default="oxygen800",
        help="Propagation material (default: oxygen at 800 nm)",
    )
    parser.add_argument(
        "-p",
        "--pulse",
        choices=["gaussian", "to_be_defined"],
        default="gaussian",
        help="Pulse type (default: gaussian and super-Gaussian pulses)",
    )
    parser.add_argument(
        "-g",
        "--gaussian_order",
        type=int,
        default=2,
        help="Gaussian order parameter (2: regular Gaussian, > 2: super-Gaussian)",
    )
    parser.add_argument(
        "--method",
        choices=["rk4"],
        default="rk4",
        help="Integration method for nonlinear term (default: rk4)",
    )
    parser.add_argument(
        "--solver",
        choices=["fss", "fcn"],
        default="fss",
        help="Solver method (fss: Fourier Split-Step, fcn: Fourier-Crank-Nicolson)",
    )

    return parser.parse_args()
