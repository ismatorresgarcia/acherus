"""ACHERUS configuration file module."""


def config_options(
    material: str = "oxygen800",
    pulse: str = "gaussian",
    gauss_order: int = 2,
    method_density: str = "RK4",
    method_raman: str = "RK4",
    method_nonlin: str = "RK4",
    solver: str = "FSS",
    ion_model: str = "MPI",
) -> dict:
    """
    Available options for launching a simulation.

    It returns a dictionary with different key-string
    pairs needed for starting the simulation. Each key
    corresponds to a specific option and they must be
    selected by the user manually.

    The available options are

    Parameters              Choice
    ======================  ================================================================
     material : str          see "materials.py" module for the list
     pulse : str             "gaussian"
     gauss_order : int       any positive number >= 2
     method_density : str    "RK4" | "RK23" | "RK45" | "DOP853" | "Radau" | "BDF" | "LSODA"
     method_raman : str      "RK4" | "RK23" | "RK45" | "DOP853" | "Radau" | "BDF" | "LSODA"
     method_nonlin : str     "RK4"
     solver : str            "FSS" | "FCN"
     ion_model : str         "MPI" | "PPT"
    ======================  ================================================================

    Returns
    -------
    opt : dict
        Dictionary with the available options for an ACHERUS simulation.
        The keys are the option names and the values are the selected
        options.

    """
    config_list = {
        "material": material,
        "pulse": pulse,
        "gauss_n": gauss_order,
        "method_d": method_density,
        "method_r": method_raman,
        "method_nl": method_nonlin,
        "solver": solver,
        "ion_model": ion_model,
    }

    return config_list
