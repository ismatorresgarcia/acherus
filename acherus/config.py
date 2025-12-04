"""Acherus configuration file module."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Type


@dataclass
class MediumConfig:
    nonlinear_index: float # [m2 / W]
    energy_gap: float # [eV]
    collision_time: float # [s]
    neutral_density: float # [1 / m3]
    initial_density: float # [1 / m3]
    raman_partition: Optional[float] = None # [-]
    raman_response_time: Optional[float] = None # [s]
    raman_rotational_time: Optional[float] = None # [s]

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
    chirp: Optional[float] = None  # [-]
    focal_length: Optional[float] = None  # [m]
    gauss_order: Optional[int] = 2  # [-]

@dataclass
class RK4DensityConfig:
    pass

@dataclass
class AdaptiveDensityConfig:
    atol: Optional[float] = 1e-6
    rtol: Optional[float] = 1e-3

@dataclass
class MPIConfig:
    cross_section: float # [s^-1 m^(2K)/W^K]
    intensity_range: Optional[Tuple[float, float]] = (1e-1, 1e18) # [W/m^2]
    num_points: Optional[int] = 10000

@dataclass
class KeldyshConfig:
    tolerance: Optional[float] = 1e-3
    max_iterations: Optional[int] = 250
    intensity_range: Optional[Tuple[float, float]] = (1e-1, 1e18) # [W/m^2]
    num_points: Optional[int] = 10000
    reduced_mass: Optional[float] = None # [-], only for condensed media

MEDIUM_CONFIG_CLASSES: Dict[str, Type] = {
    "air": MediumConfig,
    "water": MediumConfig,
    "silica": MediumConfig,
}

GRID_CONFIG_CLASSES: Dict[str, Type] = {
    "space": SpaceGridConfig,
    "axis": AxisGridConfig,
    "time": TimeGridConfig
}

PULSE_CONFIG_CLASSES: Dict[str, Type] = {
    "gaussian": GaussianPulseConfig,
}

DENSITY_CONFIG_CLASSES: Dict[str, Type] = {
    "rk4": RK4DensityConfig,
    "rk23": AdaptiveDensityConfig,
    "rk45": AdaptiveDensityConfig,
    "dop853": AdaptiveDensityConfig,
}

IONIZATION_CONFIG_CLASSES: Dict[str, Type] = {
    "mpi": MPIConfig,
    "keldysh": KeldyshConfig,
}

def _lowercase_dict(d: Dict) -> Dict:
    new_dict = {}
    for k, v in d.items():
        lower_key = k.lower()
        if isinstance(v, dict):
            new_dict[lower_key] = _lowercase_dict(v)
        else:
            new_dict[lower_key] = v
    return new_dict

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
     medium_name : str                  "air" | "water" | "silica"
     medium_par : Dict                  see "classes" above for the list
     space_par : Dict                   see "classes" above for the list
     axis_par : Dict                    see "classes" above for the list
     time_par : Dict                    see "classes" above for the list
     pulse_name : str                   "gaussian"
     pulse_par : Dict                   see "classes" above for the list
     density_method : str               "rk4" | "rk23" | "rk45" | "dop853"
     density_method_par : Dict          see "classes" above for the list
     propagation_method : str           "sscn" | "fcn"
     ionization_model : str             "mpi" | "keldysh"
     ionization_model_par : Dict        see "classes" above for the list
     computing_backend : str            "cpu" | "gpu"
    =============================       ======================================

    """
    medium_name: str
    medium_par: object
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
        medium_parameters: Dict[str, Dict],
        grid_parameters: Dict[str, Dict],
        pulse_parameters: Dict[str, Dict],
        density_solver: Dict[str, Dict],
        propagation_solver: str,
        ionization_model: Dict[str, Dict],
        computing_backend: str
    ) -> "ConfigOptions":

        medium_parameters = _lowercase_dict(medium_parameters)
        grid_parameters = _lowercase_dict(grid_parameters)
        pulse_parameters = _lowercase_dict(pulse_parameters)
        density_solver = _lowercase_dict(density_solver)
        ionization_model = _lowercase_dict(ionization_model)

        medium_name = next(iter(medium_parameters)).lower()
        medium_params = medium_parameters[medium_name]

        if medium_name not in MEDIUM_CONFIG_CLASSES:
            raise ValueError(f"Invalid medium: '{medium_name}'.")

        medium_class = MEDIUM_CONFIG_CLASSES[medium_name]
        medium_config = medium_class(**medium_params)

        grid_config = {}
        for grid_name, grid_params in grid_parameters.items():
            grid_name = grid_name.lower()
            if grid_name not in GRID_CONFIG_CLASSES:
                raise ValueError(f"Unknown grid type: {grid_name}")
            grid_class = GRID_CONFIG_CLASSES[grid_name]
            grid_config[grid_name] = grid_class(**grid_params)

        pulse_name = next(iter(pulse_parameters)).lower()
        pulse_params = pulse_parameters[pulse_name]

        if pulse_name not in PULSE_CONFIG_CLASSES:
            raise ValueError(f"Invalid pulse type: {pulse_name}.")

        pulse_class = PULSE_CONFIG_CLASSES[pulse_name]
        pulse_config = pulse_class(**pulse_params)

        density_name = next(iter(density_solver)).lower()
        density_params = density_solver[density_name]

        if density_name not in DENSITY_CONFIG_CLASSES:
            raise ValueError(f"Invalid density solver: {density_name}.")

        density_class = DENSITY_CONFIG_CLASSES[density_name]
        density_config = density_class(**density_params)

        ionization_name = next(iter(ionization_model)).lower()
        ionization_params = ionization_model[ionization_name]

        if ionization_name not in IONIZATION_CONFIG_CLASSES:
            raise ValueError(f"Invalid ionization model: {ionization_name}.")

        ionization_class = IONIZATION_CONFIG_CLASSES[ionization_name]
        ionization_config = ionization_class(**ionization_params)

        return ConfigOptions(
            medium_name=medium_name,
            medium_par=medium_config,
            space_par=grid_config["space"],
            time_par=grid_config["time"],
            axis_par=grid_config["axis"],
            pulse_name=pulse_name,
            pulse_par=pulse_config,
            density_method=density_name.upper(),
            density_method_par=density_config,
            propagation_method=propagation_solver.lower(),
            ionization_model=ionization_name,
            ionization_model_par=ionization_config,
            computing_backend=computing_backend.lower()
        )
