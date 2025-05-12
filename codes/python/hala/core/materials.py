"""material properties dictionary for laser propagation media."""

from dataclasses import dataclass


@dataclass
class MaterialParameters:
    "Materials parameters to be chosen."

    materials_list = ["oxygen800", "nitrogen800", "airdsr", "water800"]

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
        elif material_opt == "airdsr":
            self.material_atr = "airdsr"  # average air at 775 nm
        else:
            self.material_atr = "water800"  # water at 800 nm

        # Define material properties dictionary
        materials_dict = {
            "oxygen800": {
                "refraction_index_linear": 1.0,
                "refraction_index_nonlinear": 3.2e-23,
                "constant_gvd": 0.2e-28,
                "number_photons": 8,
                "effective_charge": 0.53,
                "constant_mpi": 2.81e-128,
                "ionization_energy": 1.932e-18,  # 12.06 eV
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
                "number_photons": 11,
                "effective_charge": 0.9,
                "constant_mpi": 6.31e-184,
                "ionization_energy": 2.495e-18,  # 15.576 eV
                "drude_collision_time": 3.5e-13,
                "density_neutral": 2.16e25,
                "raman_rotational_frequency": 16e12,
                "raman_response_time": 70e-15,
                "raman_partition": 0.5,
                "has_raman": True,
            },
            "airdsr": {
                "refraction_index_linear": 1.0,
                "refraction_index_nonlinear": 5.57e-23,
                "constant_gvd": 2e-28,
                "number_photons": 7,
                "effective_charge": 1.0,
                "constant_mpi": 1.3e-111,
                "ionization_energy": 1.76e-18,  # 11 eV
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
                "number_photons": 5,
                "constant_mpa": 1e-61,
                "constant_mpi": 1.2e-72,
                "ionization_energy": 1.04e-18,  # 6.5 eV
                "drude_collision_time": 3e-15,
                "density_neutral": 6.68e28,
                "raman_rotational_frequency": 0,
                "raman_response_time": 0,
                "raman_partition": 0,
                "has_raman": False,
            },
        }

        # Extract material properties from the dictionary as attributes
        material = materials_dict[self.material_atr]
        for key, value in material.items():
            setattr(self, key, value)
