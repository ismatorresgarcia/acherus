"""Acherus configuration file module."""

from dataclasses import dataclass
from typing import Dict, Type

# Union will be used for adding more pulse_shapes in the future


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

GRID_CONFIG_CLASSES = {
    "space": SpaceGridConfig,
    "axis": AxisGridConfig,
    "time": TimeGridConfig
}

PULSE_SHAPE_CLASSES: Dict[str, Type] = {
    "gaussian": GaussianPulseConfig,
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

    Parameters                 Choice
    =========================  ================================================================
     medium_name : str          see "media" module for the list
     space_grid: Dict           see "classes" above for the list
     axis_grid: Dict            see "classes" above for the list
     time_grid: Dict            see "classes" above for the list
     pulse_shape : str          "gaussian"
     pulse_config : Dict        see "classes" above for the list
     density_method : str       "RK4" | "RK23" | "RK45" | "DOP853" | "Radau" | "BDF" | "LSODA"
     nonlinear_method : str     "AB2" | "RK4"
     solver_scheme : str        "SSCN" | "FCN"
     ionization_model : str     "MPI" | "PPT"
     computing_backend : str     "CPU" | "GPU"
    =========================  ================================================================

    """
    medium_name: str
    space_grid: object
    axis_grid: object
    time_grid: object
    pulse_shape: str
    pulse_config: object # this is where the Union[shape1, shape2, ...] would be
    density_method: str
    nonlinear_method: str
    solver_scheme: str
    ionization_model: str
    computing_backend: str

    @staticmethod
    def build(
        medium_name: str,
        grid_parameters: Dict[str, Dict],
        pulse_shape: str,
        pulse_parameters: Dict, # this is where the Union[shape1, shape2, ...] would be
        density_method: str,
        nonlinear_method: str,
        solver_scheme: str,
        ionization_model: str,
        computing_backend: str
    ) -> "ConfigOptions":

        grid_config = {}
        for grid_name, params in grid_parameters.items():
            if grid_name not in GRID_CONFIG_CLASSES:
                raise ValueError(f"Unknown grid type: {grid_name}")
            grid_class = GRID_CONFIG_CLASSES[grid_name]
            grid_config[grid_name] = grid_class(**params)

        if pulse_shape not in PULSE_SHAPE_CLASSES:
            raise ValueError(f"Invalid pulse type: {pulse_shape}.")

        pulse_class = PULSE_SHAPE_CLASSES[pulse_shape]
        pulse_config = pulse_class(**pulse_parameters)

        return ConfigOptions(
            medium_name=medium_name,
            space_grid=grid_config["space"],
            time_grid=grid_config["time"],
            axis_grid=grid_config["axis"],
            pulse_shape=pulse_shape,
            pulse_config=pulse_config,
            density_method=density_method,
            nonlinear_method=nonlinear_method,
            solver_scheme=solver_scheme,
            ionization_model=ionization_model,
            computing_backend=computing_backend
        )
