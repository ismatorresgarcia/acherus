"""Acherus configuration file module."""

from dataclasses import dataclass
from typing import Dict, Optional, Type


@dataclass
class SpaceGridConfig:
    nodes: int
    space_min: float
    space_max: float

@dataclass
class AxisGridConfig:
    nodes: int
    axis_min: float
    axis_max: float
    snapshots: int

@dataclass
class TimeGridConfig:
    nodes: int
    time_min: float
    time_max: float

@dataclass
class GaussianPulseConfig:
    wavelength: float
    waist: float  # half-width at 1/e^2 of intensity
    duration: float  # half-width at 1/e^2 of intensity
    energy: float
    chirp: float
    focal_length: float
    gauss_order: int

@dataclass
class RK23DensityConfig:
    ini_step: Optional[float] = None
    rtol: float = 1e-3
    atol: float = 1e-6

@dataclass
class RK45DensityConfig:
    ini_step: Optional[float] = None
    rtol: float = 1e-3
    atol: float = 1e-6

@dataclass
class DOP853DensityConfig:
    ini_step: Optional[float] = None
    rtol: float = 1e-3
    atol: float = 1e-6

@dataclass
class RadauDensityConfig:
    ini_step: Optional[float] = None
    rtol: float = 1e-3
    atol: float = 1e-6

@dataclass
class BDFDensityConfig:
    ini_step: Optional[float] = None
    rtol: float = 1e-3
    atol: float = 1e-6

GRID_CONFIG_CLASSES: Dict[str, Type] = {
    "space": SpaceGridConfig,
    "axis": AxisGridConfig,
    "time": TimeGridConfig
}

PULSE_CONFIG_CLASSES: Dict[str, Type] = {
    "gaussian": GaussianPulseConfig,
}

DENSITY_CONFIG_CLASSES: Dict[str, Type] = {
    "RK23": RK23DensityConfig,
    "RK45": RK45DensityConfig,
    "DOP853": DOP853DensityConfig,
    "Radau": RadauDensityConfig,
    "BDF": BDFDensityConfig,
}

@dataclass
class ConfigOptions:
    """
    This class provides different dataclass key-var
    pairs needed for starting the simulation. Each key
    corresponds to a specific option and they must be
    selected by the user manually.

    The default options are provided here, but all
    the available options are

    Parameters                   Choice
    =========================    ================================================================
     medium_name : str            see "media" module for the list
     space_par: Dict              see "classes" above for the list
     axis_par: Dict               see "classes" above for the list
     time_par: Dict               see "classes" above for the list
     pulse_name : str             "gaussian"
     pulse_par : Dict             see "classes" above for the list
     density_method : str         "RK4" | "RK23" | "RK45" | "DOP853" | "Radau" | "BDF" | "LSODA"
     density_method_par : Dict    see "classes" above for the list
     nonlinear_method : str       "AB2" | "RK4"
     solver_scheme : str          "SSCN" | "FCN"
     ionization_model : str       "MPI" | "PPT"
     computing_backend : str      "CPU" | "GPU"
    =========================    ================================================================

    """
    medium_name: str
    space_par: object
    axis_par: object
    time_par: object
    pulse_name: str
    pulse_par: object
    density_method: str
    density_method_par: object
    nonlinear_method: str
    solver_scheme: str
    ionization_model: str
    computing_backend: str

    @staticmethod
    def build(
        medium_name: str,
        grid_parameters: Dict[str, Dict],
        pulse_parameters: Dict[str, Dict],
        density_solver: Dict[str, Dict],
        nonlinear_method: str,
        solver_scheme: str,
        ionization_model: str,
        computing_backend: str
    ) -> "ConfigOptions":

        grid_config = {}
        for grid_name, grid_params in grid_parameters.items():
            if grid_name not in GRID_CONFIG_CLASSES:
                raise ValueError(f"Unknown grid type: {grid_name}")
            grid_class = GRID_CONFIG_CLASSES[grid_name]
            grid_config[grid_name] = grid_class(**grid_params)

        pulse_name = next(iter(pulse_parameters))
        pulse_params = pulse_parameters[pulse_name]

        if pulse_name not in PULSE_CONFIG_CLASSES:
            raise ValueError(f"Invalid pulse type: {pulse_name}.")

        pulse_class = PULSE_CONFIG_CLASSES[pulse_name]
        pulse_config = pulse_class(**pulse_params)

        density_name = next(iter(density_solver))
        density_params = density_solver[density_name]

        if density_name not in DENSITY_CONFIG_CLASSES:
            raise ValueError(f"Invalid density solver: {density_name}.")

        density_class = DENSITY_CONFIG_CLASSES[density_name]
        density_config = density_class(**density_params)

        return ConfigOptions(
            medium_name=medium_name,
            space_par=grid_config["space"],
            time_par=grid_config["time"],
            axis_par=grid_config["axis"],
            pulse_name=pulse_name,
            pulse_par=pulse_config,
            density_method=density_name,
            density_method_par=density_config,
            nonlinear_method=nonlinear_method,
            solver_scheme=solver_scheme,
            ionization_model=ionization_model,
            computing_backend=computing_backend
        )
