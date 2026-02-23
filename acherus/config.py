"""Acherus configuration file module."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type


@dataclass
class MediumConfig:
    """Medium properties list of parameters."""

    nonlinear_index: float  # [m2 / W]
    energy_gap: float  # [eV]
    collision_time: float  # [s]
    neutral_density: float  # [1 / m3]
    initial_density: float  # [1 / m3]
    recombination_rate: Optional[float] = None  # [m3 / s]
    raman_partition: Optional[float] = None  # [-]
    raman_response_time: Optional[float] = None  # [s]
    raman_rotational_time: Optional[float] = None  # [s]


@dataclass
class SpaceGridConfig:
    """Spatial propagation grid parameters."""

    nodes: int
    space_min: float
    space_max: float


@dataclass
class AxisGridConfig:
    """Axis grid parameters and number of z snapshots."""

    nodes: int
    axis_min: float
    axis_max: float
    snapshots: int


@dataclass
class TimeGridConfig:
    """Temporal grid parameters."""

    nodes: int
    time_min: float
    time_max: float


@dataclass
class GaussianPulseConfig:
    """Input Gaussian pulse parameters."""

    wavelength: float
    waist: float  # half-width at 1/e^2 of intensity
    duration: float  # half-width at 1/e^2 of intensity
    energy: float
    focal_length: Optional[float] = None  # [m]
    chirp: Optional[float] = None  # [-]
    gauss_order: Optional[int] = 2  # [-]


@dataclass
class RK4DensityConfig:
    """Fixed-step density solver configuration."""


@dataclass
class AdaptiveDensityConfig:
    """Adaptive-step density solver tolerances."""

    atol: Optional[float] = 1e-6
    rtol: Optional[float] = 1e-3


@dataclass
class MPIConfig:
    """Multiphoton ionization (MPI) model parameters."""

    cross_section: float  # [s^-1 m^(2K)/W^K]
    intensity_range: Optional[Tuple[float, float]] = (1e-1, 1e18)  # [W/m^2]
    num_points: Optional[int] = 10000


@dataclass
class KeldyshConfig:
    """Generalized Keldysh model parameters."""

    tolerance: Optional[float] = 1e-3
    max_iterations: Optional[int] = 250
    intensity_range: Optional[Tuple[float, float]] = (1e-1, 1e18)  # [W/m^2]
    num_points: Optional[int] = 10000
    reduced_mass: Optional[float] = None  # [-], only for condensed media


MEDIUM_CONFIG_CLASSES: Dict[str, Type[Any]] = {
    "air": MediumConfig,
    "water": MediumConfig,
    "silica": MediumConfig,
}

GRID_CONFIG_CLASSES: Dict[str, Type[Any]] = {
    "space": SpaceGridConfig,
    "axis": AxisGridConfig,
    "time": TimeGridConfig,
}


PULSE_CONFIG_CLASSES: Dict[str, Type[Any]] = {
    "gaussian": GaussianPulseConfig,
}

DENSITY_CONFIG_CLASSES: Dict[str, Type[Any]] = {
    "rk4": RK4DensityConfig,
    "rk23": AdaptiveDensityConfig,
    "rk45": AdaptiveDensityConfig,
    "dop853": AdaptiveDensityConfig,
}

IONIZATION_CONFIG_CLASSES: Dict[str, Type[Any]] = {
    "mpi": MPIConfig,
    "keldysh": KeldyshConfig,
}


def _lowercase_dict(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively lowercase dictionary keys."""
    lowered_dict = {}
    for key, value in input_dict.items():
        lower_key = key.lower()
        if isinstance(value, dict):
            lowered_dict[lower_key] = _lowercase_dict(value)
        else:
            lowered_dict[lower_key] = value
    return lowered_dict


def _build_single_choice_config(
    options: Dict[str, Dict[str, Any]],
    class_map: Dict[str, Type[Any]],
    invalid_message: str,
) -> tuple[str, Any]:
    """Build config object from a single-key options mapping."""
    option_name = next(iter(options)).lower()
    option_params = options[option_name]

    if option_name not in class_map:
        raise ValueError(invalid_message.format(name=option_name))

    config_class = class_map[option_name]
    return option_name, config_class(**option_params)


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
    data_output_path: Optional[Path] = None
    figure_output_path: Optional[Path] = None

    @staticmethod
    def build(
        medium_parameters: Dict[str, Dict],
        grid_parameters: Dict[str, Dict],
        pulse_parameters: Dict[str, Dict],
        density_solver: Dict[str, Dict],
        propagation_solver: str,
        ionization_model: Dict[str, Dict],
        computing_backend: str,
        data_output_path: Optional[str] = None,
        figure_output_path: Optional[str] = None,
    ) -> "ConfigOptions":
        """Build a normalized `ConfigOptions` instance from raw input mappings."""

        medium_parameters = _lowercase_dict(medium_parameters)
        grid_parameters = _lowercase_dict(grid_parameters)
        pulse_parameters = _lowercase_dict(pulse_parameters)
        density_solver = _lowercase_dict(density_solver)
        ionization_model = _lowercase_dict(ionization_model)

        medium_name, medium_config = _build_single_choice_config(
            medium_parameters,
            MEDIUM_CONFIG_CLASSES,
            "Invalid medium: '{name}'.",
        )

        grid_config = {}
        for grid_name, grid_params in grid_parameters.items():
            grid_name = grid_name.lower()
            if grid_name not in GRID_CONFIG_CLASSES:
                raise ValueError(f"Unknown grid type: {grid_name}")
            grid_class = GRID_CONFIG_CLASSES[grid_name]
            grid_config[grid_name] = grid_class(**grid_params)

        pulse_name, pulse_config = _build_single_choice_config(
            pulse_parameters,
            PULSE_CONFIG_CLASSES,
            "Invalid pulse type: {name}.",
        )

        density_name, density_config = _build_single_choice_config(
            density_solver,
            DENSITY_CONFIG_CLASSES,
            "Invalid density solver: {name}.",
        )

        ionization_name, ionization_config = _build_single_choice_config(
            ionization_model,
            IONIZATION_CONFIG_CLASSES,
            "Invalid ionization model: {name}.",
        )

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
            computing_backend=computing_backend.lower(),
            data_output_path=Path(data_output_path) if data_output_path else None,
            figure_output_path=Path(figure_output_path) if figure_output_path else None,
        )
