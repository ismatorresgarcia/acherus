"""AiurPy configuration file module."""


def config_options():
    """
    Available options for launching AiurPy simulation.

    It returns a dictionary with different key-string
    pairs needed for launching the simulation. Each key
    corresponds to a specific option and they must be
    selected by the user.

    Available options are:
    -> material: The propagating medium. For more information, look in the "materials"
    module.
        * choices: "oxygen800", "nitrogen800", "airdsr", "water800".
        * default: "oxygen800".
    -> pulse: The initial (z = 0) pulse shape. For more information, look in
    the "initial" module.
        * choices: "gaussian".
        * default: "gaussian".
    -> gaussian_order: The super-Gaussian pulse exponent. For more information,
    look in the "initial" module.
        * choices: "n", where n is any positive integer >= 2.
        * default: "2".
    -> method: Numerical method used for envelope equation nonlinear terms integration.
    For more information, look in the "numerical.envelope" submodule.
        * choices: "rk4".
        * default: "rk4".
    -> solver: Numerical scheme used for solving propagation. For more information,
    look in the "solvers.fss/fcn" submodules.
        * choices: "fss", "fcn".
        * default: "fss".
    -> ion_model: Ionization rate model used. For more information, look in the
    "core.ionization" submodule.
        * choices: "mpi", "ppt"
        * default: mpi
    """

    return {
        "material": "oxygen800",
        "pulse": "gaussian",
        "gaussian_order": 2,
        "method": "rk4",
        "solver": "fss",
        "ion_model": "mpi",
    }
