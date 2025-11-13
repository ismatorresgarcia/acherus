"""Helper module injected into "keldysh.py" for computing series summation."""

import numpy as xp
from scipy.special import dawsn  # pylint: disable=no-name-in-module


def series_sum(alpha, beta, p, medium, tol, max_iter):
    """Compute the probability series for the Keldysh model."""
    n = len(alpha)
    m = max_iter
    if medium == "gas":
        idx_min = xp.ceil(p)
        ids = idx_min[:, None] + xp.arange(m)[None, :]
        args = ids - p[:, None]
        terms = xp.exp(-alpha[:, None] * args) * dawsn(
            xp.sqrt(beta[:, None] * args)
        )
    else:  # condensed
        idx_min = xp.zeros(n, dtype=int)
        ids = idx_min[:, None] + xp.arange(m)[None, :]
        args = xp.floor(p + 1) - p
        terms = xp.exp(-alpha[:, None] * ids) * dawsn(
            xp.sqrt(beta[:, None] * (ids + 2 * args[:, None]))
        )
    sums = xp.cumsum(terms, axis=1)
    stop = terms < tol * sums
    return sums[xp.arange(n), xp.argmax(stop, axis=1)]
