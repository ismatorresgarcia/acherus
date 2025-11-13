"""Acherus configuration file module."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Type


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
class RK4DensityConfig:
    pass

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
class MPIConfig:
    wavelength: float
    linear_index: float
    energy_gap: float
    cross_section: float
    intensity_range: Tuple[float, float] = (1e-1, 1e18)
    num_points: int = 10000

@dataclass
class PPTGasConfig:
    wavelength: float
    energy_gap: float
    tolerance: float = 1e-3
    max_iterations: int = 250
    intensity_range: Tuple[float, float] = (1e-1, 1e18)
    num_points: int = 10000

@dataclass
class PPTCondensedConfig:
    wavelength: float
    energy_gap: float
    neutral_density: float
    reduced_mass: float
    tolerance: float = 1e-3
    max_iterations: int = 250
    intensity_range: Tuple[float, float] = (1e-1, 1e18)
    num_points: int = 10000

GRID_CONFIG_CLASSES: Dict[str, Type] = {
    "space": SpaceGridConfig,
    "axis": AxisGridConfig,
    "time": TimeGridConfig
}

PULSE_CONFIG_CLASSES: Dict[str, Type] = {
    "gaussian": GaussianPulseConfig,
}

DENSITY_CONFIG_CLASSES: Dict[str, Type] = {
    "RK4": RK4DensityConfig,
    "RK23": RK23DensityConfig,
    "RK45": RK45DensityConfig,
    "DOP853": DOP853DensityConfig,
}

IONIZATION_CONFIG_CLASSES: Dict[str, Type] = {
    "MPI": MPIConfig,
    "PPTG": PPTGasConfig,
    "PPTC": PPTCondensedConfig,
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

    Parameters                          Choice
    =============================       ======================================
     medium_name : str                  see "media" module for the list
     space_par: Dict                    see "classes" above for the list
     axis_par: Dict                     see "classes" above for the list
     time_par: Dict                     see "classes" above for the list
     pulse_name : str                   "gaussian"
     pulse_par : Dict                   see "classes" above for the list
     density_method : str               "RK4" | "RK23" | "RK45" | "DOP853"
     density_method_par : Dict          see "classes" above for the list
     propagation_method : str           "SSCN" | "FCN"
     ionization_model : str             "MPI" | "PPTG" | "PPTC"
     ionization_model_par : Dict        see "classes" above for the list
     computing_backend : str            "CPU" | "GPU"
    =============================       ======================================

    """
    medium_name: str
    space_par: object
    axis_par: object
    time_par: object
    pulse_name: str
    pulse_par: object
    density_method: str
    density_method_par: object
    propagation_method: str
    ionization_model: str
    ionization_model_par: object
    computing_backend: str

    @staticmethod
    def build(
        medium_name: str,
        grid_parameters: Dict[str, Dict],
        pulse_parameters: Dict[str, Dict],
        density_solver: Dict[str, Dict],
        propagation_solver: str,
        ionization_model: Dict[str, Dict],
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

        ionization_name = next(iter(ionization_model))
        ionization_params = ionization_model[ionization_name]

        if ionization_name not in IONIZATION_CONFIG_CLASSES:
            raise ValueError(f"Invalid ionization model: {ionization_name}.")

        ionization_class = IONIZATION_CONFIG_CLASSES[ionization_name]
        ionization_config = ionization_class(**ionization_params)

        return ConfigOptions(
            medium_name=medium_name,
            space_par=grid_config["space"],
            time_par=grid_config["time"],
            axis_par=grid_config["axis"],
            pulse_name=pulse_name,
            pulse_par=pulse_config,
            density_method=density_name,
            density_method_par=density_config,
            propagation_method=propagation_solver,
            ionization_model=ionization_name,
            ionization_model_par=ionization_config,
            computing_backend=computing_backend
        )
