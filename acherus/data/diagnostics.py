"""Diagnosing tools module."""

import sys

import numpy as np


def error_diagnostics(solver, exit_on_error=True, save_on_error=True):
    """
    Check for errors in variables from solver state.

    Parameters
    ----------
    solver : object
        Solver object with data.
    exit_on_error : bool, default: True
        Whether to exit program on validation failure.
    save_on_error : bool, default: True
        Whether to save solver state on validation failure.

    Returns
    -------
    binary : bool
        True if valid, False if invalid (when exit_on_error is False).

    """
    checklist = [solver.envelope_rt, solver.density_rt]
    if any(np.any(~np.isfinite(x)) for x in checklist):
        if exit_on_error:
            print("ERROR: Non-finite values detected in envelope or density")
            if save_on_error and hasattr(solver, "output"):
                print("Saving propagation state before exiting...")
                solver.output.save_results(solver, solver.grid)
            sys.exit(1)
        else:
            print("WARNING: Non-finite values detected in envelope or density")
            return False

    return True


def cheap_diagnostics(solver, step):
    """
    Save memory cheap diagnostics data for current step.

    Parameters
    ----------
    solver : object
        Solver object with data.
    step : integer
        Current propagation step.

    """
    error_diagnostics(solver)

    envelope_rt = solver.envelope_rt
    density_rt = solver.density_rt
    fluence_r = solver.fluence_r

    max_intensity_idx = np.argmax(np.abs(envelope_rt), axis=1)
    max_density_idx = np.argmax(density_rt, axis=1)

    solver.envelope_r0_zt[step, :] = envelope_rt[0]
    solver.envelope_tp_rz[:, step] = envelope_rt[
        np.arange(envelope_rt.shape[0]), max_intensity_idx
    ]
    solver.density_r0_zt[step, :] = density_rt[0]
    solver.density_tp_rz[:, step] = density_rt[
        np.arange(density_rt.shape[0]), max_density_idx
    ]
    solver.fluence_rz[:, step] = fluence_r


def expensive_diagnostics(solver, step):
    """
    Save memory expensive diagnostics data for current step.

    Parameters
    ----------
    solver : object
        Solver object with data.
    step : integer
        Current propagation step.

    """
    solver.envelope_snapshot_rzt[:, step, :] = solver.envelope_rt
    solver.density_snapshot_rzt[:, step, :] = solver.density_rt
    solver.snapshot_z_index[step] = (
        solver.snapshot_z_index[step - 1] + solver.grid.z_steps_per_snapshot
    )
