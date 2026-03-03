# pylint: disable=unused-import,import-outside-toplevel,missing-module-docstring,missing-function-docstring

import pytest


def test_dependency_imports():
    import h5py
    import matplotlib
    import numba
    import numpy
    import pyfftw
    import scipy


def test_module_imports():
    from acherus import (
        FCN,
        SSCN,
        ConfigOptions,
        Equation,
        Grid,
        KeldyshIonization,
        Laser,
        Medium,
        OutputManager,
        Shared,
    )
