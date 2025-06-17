"""
Material properties module for laser propagation media.

The units used in the module are

============================  ======================
 refraction_index_linear       [-]
 refraction_index_nonlinear    [m**2 / W]
 constant_gvd                  [s**2 / m]
 effective_charge              [-]
 constant_mpi                  [s-1 m**(2K) / W**K]
 ionization_energy             [eV]
 drude_collision_time          [s]
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
class Material:
    """Material parameters available."""

    refraction_index_linear: float
    refraction_index_nonlinear: float
    constant_gvd: float
    effective_charge: float
    constant_mpi: float
    ionization_energy: float
    drude_collision_time: float
    density_neutral: float
    density_initial: float
    raman_rotational_frequency: Optional[float] = 0.0
    raman_response_time: Optional[float] = 0.0
    raman_partition: Optional[float] = 0.0
    has_raman: bool = False


# Material instances defined
MATERIALS = {
    "oxygen800": Material(
        refraction_index_linear=1.0,
        refraction_index_nonlinear=3.2e-23,
        constant_gvd=0.2e-28,
        effective_charge=0.53,
        constant_mpi=2.81e-128,
        ionization_energy=12.063,
        drude_collision_time=3.5e-13,
        density_neutral=0.54e25,
        density_initial=1e15,
        raman_rotational_frequency=16e12,
        raman_response_time=70e-15,
        raman_partition=0.5,
        has_raman=True,
    ),
    "nitrogen800": Material(
        refraction_index_linear=1.0,
        refraction_index_nonlinear=3.2e-23,
        constant_gvd=0.2e-28,
        effective_charge=0.9,
        constant_mpi=6.31e-184,
        ionization_energy=15.576,
        drude_collision_time=3.5e-13,
        density_neutral=2.16e25,
        density_initial=1e15,
        raman_rotational_frequency=16e12,
        raman_response_time=70e-15,
        raman_partition=0.5,
        has_raman=True,
    ),
    "air775_1": Material(
        refraction_index_linear=1.0,
        refraction_index_nonlinear=5.57e-23,
        constant_gvd=2e-28,
        effective_charge=0.11,
        constant_mpi=1.34e-111,
        ionization_energy=11.0,
        drude_collision_time=3.5e-13,
        density_neutral=2.7e25,
        density_initial=1e15,
        raman_rotational_frequency=16e12,
        raman_response_time=77e-15,
        raman_partition=0.5,
        has_raman=True,
    ),
    "water800": Material(
        refraction_index_linear=1.334,
        refraction_index_nonlinear=4.1e-20,
        constant_gvd=248e-28,
        effective_charge=1.0,
        constant_mpi=1.2e-72,
        ionization_energy=6.5,
        drude_collision_time=3e-15,
        density_neutral=6.68e28,
        density_initial=1e21,
        raman_rotational_frequency=0.0,
        raman_response_time=0.0,
        raman_partition=0.0,
        has_raman=False,
    ),
    "silica800": Material(
        refraction_index_linear=1.453,
        refraction_index_nonlinear=3.2e-20,
        constant_gvd=361e-28,
        effective_charge=1.0,
        constant_mpi=1.3e-75,
        ionization_energy=7.6,
        drude_collision_time=3.2e-15,
        density_neutral=2.1e28,
        density_initial=1e21,
        raman_rotational_frequency=0.0,
        raman_response_time=0.0,
        raman_partition=0.0,
        has_raman=False,
    ),
}


class MaterialParameters:
    """Material parameters checking and extraction."""

    def __init__(self, material_opt: str = "oxygen800"):
        if material_opt not in MATERIALS:
            raise ValueError(
                f"Not available material option: '{material_opt}'. "
                f"Available materials are: {', '.join(MATERIALS.keys())}"
            )
        self.material = MATERIALS[material_opt]

    def __getattr__(self, name):
        # Extract material properties as direct attributes
        return getattr(self.material, name)
