"""ATFURPy configuration file module."""


def config_options():
    """
    Available options for launching a simulation.

    It returns a dictionary with different key-string
    pairs needed for starting the simulation. Each key
    corresponds to a specific option and they must be
    selected by the user manually.

    The available options are

    ================ =======================================================
     material:        "oxygen800" (see "materials.py" module for the list)
     pulse:           "gaussian"
     gaussian_order:  "2" (or any positive integer)
     method:          "rk4"
     solver:          "fss" (or "fcn")
     ion_model:       "mpi" (or "ppt")
    ================ =======================================================

    Returns
    -------
    opt : dict
        Dictionary with the available options for ATFURPy simulation.
        The keys are the option names and the values are the selected
        options.

    """

    return {
        "material": "oxygen800",
        "pulse": "gaussian",
        "gaussian_order": 2,
        "method": "rk4",
        "solver": "fss",
        "ion_model": "mpi",
    }
