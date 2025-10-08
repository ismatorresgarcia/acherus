"""Acherus configuration file module."""


def config_options(
    medium: str = "oxygen800",
    pulse: str = "gaussian",
    gauss_order: int = 2,
    method_density: str = "RK45",
    method_nonlinear: str = "AB2",
    solver: str = "FCN",
    ion_model: str = "PPT",
    gpu: bool = False,
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
     medium : str            see "media.py" module for the list
     pulse : str             "gaussian"
     gauss_order : int       any positive number >= 2
     method_density : str    "RK4" | "RK23" | "RK45" | "DOP853" | "Radau" | "BDF" | "LSODA"
     method_nonlin : str     "AB2" | "RK4" 
     solver : str            "SSCN" | "FCN"
     ion_model : str         "MPI" | "PPT"
     gpu : bool              "True" | "False"
    ======================  ================================================================

    Returns
    -------
    opt : dict
        Dictionary with the available options for an Acherus simulation.
        The keys are the option names and the values are the selected
        options.

    """
    config_list = {
        "medium": medium,
        "pulse": pulse,
        "gauss_n": gauss_order,
        "method_d": method_density,
        "method_nl": method_nonlinear,
        "solver": solver,
        "ion_model": ion_model,
        "gpu": gpu,
    }

    return config_list
