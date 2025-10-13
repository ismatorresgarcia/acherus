"""Acherus configuration file module."""

from dataclasses import dataclass


@dataclass
class ConfigOptions:
    """
    This class provides different key-string dataclass
    pairs needed for starting the simulation. Each key
    corresponds to a specific option and they must be
    selected by the user manually.

    The default options are provided here, but all 
    the available options are

    Parameters                 Choice
    =========================  ================================================================
     medium_name : str          see "media.py" module for the list
     pulse_shape : str          "gaussian"
     gauss_order : int          any positive number >= 2
     density_method : str       "RK4" | "RK23" | "RK45" | "DOP853" | "Radau" | "BDF" | "LSODA"
     nonlinear_method : str     "AB2" | "RK4"
     solver_scheme : str        "SSCN" | "FCN"
     ionization_model : str     "MPI" | "PPT"
     computing_engine : str     "CPU" | "GPU"
    =========================  ================================================================

    """
    medium_name: str = "oxygen_800"
    pulse_shape: str = "gaussian"
    gauss_order: int = 2
    density_method: str = "RK45"
    nonlinear_method: str = "AB2"
    solver_scheme: str = "FCN"
    ionization_model: str = "PPT"
    computing_engine: str = "CPU"
