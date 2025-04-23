"""Physical and mathematical constants."""

import numpy as np


class Constants:
    "Physical and mathematical constants."

    def __init__(self):
        self.light_speed_0 = 299792458.0
        self.electric_permittivity_0 = 8.8541878128e-12
        self.electron_mass = 9.1093837139e-31
        self.electron_charge = 1.602176634e-19
        self.planck_bar = 1.05457182e-34
        self.pi = np.pi
        self.imaginary_unit = 1j
