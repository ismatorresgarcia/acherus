# pylint: disable=missing-module-docstring,missing-function-docstring

import math

from acherus.constants import C_LIGHT, E_CHARGE, EPS_0, HBAR, M_E, PI


def test_constants_are_finite_positive():
    constants = (PI, C_LIGHT, E_CHARGE, EPS_0, HBAR, M_E)
    assert all(math.isfinite(value) for value in constants)
    assert all(value > 0 for value in constants)


def test_pi_consistency():
    assert PI == math.pi
