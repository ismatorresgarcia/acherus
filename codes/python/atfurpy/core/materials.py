"""
Material properties dictionary for laser propagation media.

The units used for every property in the dictionary are

============================ ======================
 refraction_index_linear      [-]
 refraction_index_nonlinear   [m**2 / W]
 constant_gvd                 [s**2 / m]
 number_photons               [-]
 effective_charge             [-]
 constant_mpi                 [s-1 m**(2K) / W**K]
 ionization_energy            [eV]
 drude_collision_time         [s]
 density_neutral              [m**(-3)]
 raman_rotational_frequency   [Hz]
 raman_response_time          [s]
 raman_partition              [-]
 has_raman                    [-]

"""

from dataclasses import dataclass

import numpy as np


@dataclass
class MaterialParameters:
    "Material parameters to be chosen."

    materials_list = ["oxygen800", "nitrogen800", "air775_1", "water800", "silica800"]

    def __init__(self, material_opt="oxygen800"):
        if material_opt not in self.materials_list:
            material_valid = ", ".join(self.materials_list)
            raise ValueError(
                f"Not available material option: '{material_opt}'. "
                f"Available materials are: {material_valid}"
            )

        if material_opt == "oxygen800":
            self.material_atr = "oxygen800"  # oxygen at 800 nm
        elif material_opt == "nitrogen800":
            self.material_atr = "nitrogen800"  # nitrogen at 800 nm
        elif material_opt == "air775_1":
            self.material_atr = "air775_1"  # one average air at 775 nm
        elif material_opt == "water800":
            self.material_atr = "water800"  # water at 800 nm
        elif material_opt == "silica800":
            self.material_atr = "silica800"  # fused silica at 800 nm

        # Define material properties dictionary
        materials_dict = {
            "oxygen800": {
                "refraction_index_linear": 1.0,
                "refraction_index_nonlinear": 3.2e-23,
                "constant_gvd": 0.2e-28,
                "number_photons": 8.0,
                "effective_charge": 0.53,
                "constant_mpi": 2.81e-128,
                "ionization_energy": 12.06,
                "drude_collision_time": 3.5e-13,
                "density_neutral": 0.54e25,
                "raman_rotational_frequency": 16e12,
                "raman_response_time": 70e-15,
                "raman_partition": 0.5,
                "has_raman": True,
            },
            "nitrogen800": {
                "refraction_index_linear": 1.0,
                "refraction_index_nonlinear": 3.2e-23,
                "constant_gvd": 0.2e-28,
                "number_photons": 11.0,
                "effective_charge": 0.9,
                "constant_mpi": 6.31e-184,
                "ionization_energy": 15.576,
                "drude_collision_time": 3.5e-13,
                "density_neutral": 2.16e25,
                "raman_rotational_frequency": 16e12,
                "raman_response_time": 70e-15,
                "raman_partition": 0.5,
                "has_raman": True,
            },
            "air775_1": {
                "refraction_index_linear": 1.0,
                "refraction_index_nonlinear": 5.57e-23,
                "constant_gvd": 2e-28,
                "number_photons": 7.0,
                "effective_charge": 0.11,
                "constant_mpi": 1.34e-111,
                "ionization_energy": 11.0,
                "drude_collision_time": 3.5e-13,
                "density_neutral": 2.7e25,
                "raman_rotational_frequency": 16e12,
                "raman_response_time": 77e-15,
                "raman_partition": 0.5,
                "has_raman": True,
            },
            "water800": {
                "refraction_index_linear": 1.334,
                "refraction_index_nonlinear": 4.1e-20,
                "constant_gvd": 248e-28,
                "number_photons": 5.0,
                "effective_charge": 1.0,
                "constant_mpa": 1e-61,
                "constant_mpi": 1.2e-72,
                "ionization_energy": 6.5,
                "drude_collision_time": 3e-15,
                "density_neutral": 6.68e28,
                "raman_rotational_frequency": 0.0,
                "raman_response_time": 0.0,
                "raman_partition": 0.0,
                "has_raman": False,
            },
            "silica800": {
                "refraction_index_linear": 1.453,
                "refraction_index_nonlinear": 3.54e-20,
                "constant_gvd": 361e-28,
                "number_photons": 6.0,
                "effective_charge": 1.0,
                "constant_mpi": 1.5e-95,
                "ionization_energy": 9.0,
                "drude_collision_time": 2.33e-14,
                "density_neutral": 2.1e28,
                "raman_rotational_frequency": 0.0,
                "raman_response_time": 0.0,
                "raman_partition": 0.0,
                "has_raman": False,
            },
        }

        # Extract material properties from the dictionary as attributes for
        # later usage in other modules
        material = materials_dict[self.material_atr]
        for key, value in material.items():
            if isinstance(value, float):
                setattr(self, key, np.float64(value))
            else:
                setattr(self, key, value)
