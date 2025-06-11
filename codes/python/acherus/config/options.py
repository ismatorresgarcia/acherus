"""ACHERUS configuration file module."""


def config_options(
    material: str = "oxygen800",
    pulse: str = "gaussian",
    gauss_n: int = 2,
    method: str = "rk4",
    solver: str = "fss",
    ion_model: str = "mpi",
) -> dict:
    """
    Available options for launching a simulation.

    It returns a dictionary with different key-string
    pairs needed for starting the simulation. Each key
    corresponds to a specific option and they must be
    selected by the user manually.

    The available options are

    Parameters
    ==================  =======================================================
     material : str      see "materials.py" module for the list
     pulse : str         "gaussian"
     gauss_n : int       any positive number >= 2
     method : str        "rk4"
     solver : str        "fss" | "fcn"
     ion_model : str     "mpi" | "ppt"
    ==================  =======================================================

    Returns
    -------
    opt : dict
        Dictionary with the available options for an ACHERUS simulation.
        The keys are the option names and the values are the selected
        options.

    """
    config_dict = {
        "material": material,
        "pulse": pulse,
        "gauss_n": gauss_n,
        "method": method,
        "solver": solver,
        "ion_model": ion_model,
    }

    return config_dict
