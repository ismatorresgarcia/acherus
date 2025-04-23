"""Medium parameters for laser propagation."""


class MediumParameters:
    "Medium parameters to be chosen."

    def __init__(self, medium_opt="oxygen800"):
        if medium_opt.upper() == "OXYGEN800":
            self.medium_type = "oxygen800"
        elif medium_opt.upper() == "AIRDSR":
            self.medium_type = "airDSR"
        else:  # water at 800 nm
            self.medium_type = "water800"

        # Define parameter sets
        parameters = {
            "oxygen800": {
                "refraction_index_linear": 1.0,
                "refraction_index_nonlinear": 3.2e-23,
                "constant_gvd": 0.2e-28,
                "number_photons": 8,
                "constant_mpa": 3e-121,
                "constant_mpi": 2.81e-128,
                "ionization_energy": 1.932e-18,  # 12.06 eV
                "drude_collision_time": 3.5e-13,
                "density_neutral": 0.54e25,
                "raman_rotational_frequency": 16e12,
                "raman_response_time": 70e-15,
                "raman_partition": 0.5,
                "has_raman": True,
            },
            "airDSR": {
                "refraction_index_linear": 1.0,
                "refraction_index_nonlinear": 5.57e-23,
                "constant_gvd": 2e-28,
                "number_photons": 7,
                "constant_mpa": 6.5e-104,
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

        # Apply parameters from the dictionary
        medium_params = parameters[self.medium_type]
        for key, value in medium_params.items():
            setattr(self, key, value)
