"""Command line argument parser module."""

import argparse

from hala import __version__


def create_cli_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Cylindrical 2D Fourier Split-step solver "
        "for ultrashort filamentation in transparent media. v{__version__}"
    )
    parser.add_argument(
        "-m",
        "--medium",
        choices=["oxygen800", "airDSR", "water800"],
        default="oxygen800",
        help="Propagation medium (default: oxygen at 800 nm)",
    )
    parser.add_argument(
        "-p",
        "--pulse",
        choices=["gauss", "to_be_defined"],
        default="gauss",
        help="Pulse type (default: gaussian and super-Gaussian pulses)",
    )
    parser.add_argument(
        "-g",
        "--gauss_order",
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
