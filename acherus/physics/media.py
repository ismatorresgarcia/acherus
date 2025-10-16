"""
Media properties module for laser propagation media.

The units used in the module are

============================  ======================
 refraction_index_linear       [-]
 refraction_index_nonlinear    [m**2 / W]
 constant_k1                   [s / m]
 constant_k2                   [s**2 / m]
 effective_charge              [-]
 constant_mpi                  [s-1 m**(2K) / W**K]
 ionization_energy             [eV]
 drude_time                    [s]
 density_neutral               [m**(-3)]
 density_initial               [m**(-3)]
 raman_rotational_frequency    [Hz]
 raman_response_time           [s]
 raman_partition               [-]
 has_raman                     [-]

"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Medium:
    """Medium parameters available."""

    refraction_index_linear: float
    refraction_index_nonlinear: float
    constant_k1: float
    constant_k2: float
    effective_charge: float
    constant_mpi: float
    ionization_energy: float
    drude_time: float
    density_neutral: float
    density_initial: float
    raman_rotational_frequency: Optional[float] = 0.0
    raman_response_time: Optional[float] = 0.0
    raman_partition: Optional[float] = 0.0
    has_raman: bool = False


# Medium instances defined
MEDIA = {
    "oxygen_800": Medium(
        refraction_index_linear=1.0,
        refraction_index_nonlinear=3.2e-23,
        constant_k1=3.3365e-9,
        constant_k2=0.2e-28,
        effective_charge=0.53,
        constant_mpi=2.81e-128,
        ionization_energy=12.063,
        drude_time=3.5e-13,
        density_neutral=0.54e25,
        density_initial=1e9,
        raman_rotational_frequency=16e12,
        raman_response_time=70e-15,
        raman_partition=0.5,
        has_raman=True,
    ),
    "nitrogen_800": Medium(
        refraction_index_linear=1.0,
        refraction_index_nonlinear=3.2e-23,
        constant_k1=3.3365e-9,
        constant_k2=0.2e-28,
        effective_charge=0.9,
        constant_mpi=6.31e-184,
        ionization_energy=15.576,
        drude_time=3.5e-13,
        density_neutral=2.16e25,
        density_initial=1e9,
        raman_rotational_frequency=16e12,
        raman_response_time=70e-15,
        raman_partition=0.5,
        has_raman=True,
    ),
    "water_800": Medium(
        refraction_index_linear=1.334,
        refraction_index_nonlinear=4.1e-20,
        constant_k1=4.4670e-9,
        constant_k2=248e-28,
        effective_charge=1.0,
        constant_mpi=1.2e-72,
        ionization_energy=6.5,
        drude_time=3e-15,
        density_neutral=6.68e28,
        density_initial=1e9,
        raman_rotational_frequency=0.0,
        raman_response_time=0.0,
        raman_partition=0.0,
        has_raman=False,
    ),
    "water_400": Medium(
        refraction_index_linear=1.34,
        refraction_index_nonlinear=4.1e-20,
        constant_k1=4.4698e-9,
        constant_k2=670e-28,
        effective_charge=1.0,
        constant_mpi=5.4e-41,
        ionization_energy=6.5,
        drude_time=1e-15,
        density_neutral=6.7e28,
        density_initial=1e9,
        raman_rotational_frequency=0.0,
        raman_response_time=0.0,
        raman_partition=0.0,
        has_raman=False,
    ),
    "silica_800": Medium(
        refraction_index_linear=1.453,
        refraction_index_nonlinear=3.2e-20,
        constant_k1=8.908e-9,
        constant_k2=361e-28,
        effective_charge=1.0,
        constant_mpi=1.3e-75,
        ionization_energy=7.6,
        drude_time=6.9e-15,
        density_neutral=2.1e28,
        density_initial=1e9,
        raman_rotational_frequency=0.0,
        raman_response_time=0.0,
        raman_partition=0.0,
        has_raman=False,
    ),
}


class MediumParameters:
    """Medium parameters checking and extraction."""

    def __init__(self, medium_name: str = "oxygen_800"):
        if medium_name not in MEDIA:
            raise ValueError(
                f"Not available medium option: '{medium_name}'. "
                f"Available media are: {', '.join(MEDIA.keys())}"
            )
        self.medium = MEDIA[medium_name]

    def __getattr__(self, name):
        # Extract medium properties as direct attributes
        return getattr(self.medium, name)
