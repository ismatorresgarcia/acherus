"""Diagnosing tools module."""

import sys

import numpy as np
from h5py import File

from .paths import sim_dir as path

MONITORING_STEPS = 100
monitoring_path = path / "acherus_monitoring.h5"
profiler_path = path / "acherus_log.txt"


def validate_step(solver, exit_on_error=True, save_on_error=True):
    """
    Validate numerical results from solver state.

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
    validate_step(solver)

    envelope_rt = solver.envelope_rt
    density_rt = solver.density_rt
    fluence_r = solver.fluence_r
    radius = solver.radius

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
    solver.radius_z[step] = radius[0]


def monitoring_diagnostics(solver, step):
    """
    Save diagnostics progressively every desired number of steps
    and write them in a HDF5 file on the run.

    Parameters
    ----------
    solver : object
        Solver object with data.
    step : integer
        Current propagation step.

    """
    if step == 1:
        with File(monitoring_path, "w") as f:
            envelope_grp = f.create_group("envelope")
            envelope_grp.create_dataset(
                "peak_rz",
                shape=(solver.grid.r_nodes, solver.grid.z_nodes),
                maxshape=(solver.grid.r_nodes, None),
                dtype=np.complex128,
                compression="gzip",
                chunks=(
                    solver.grid.r_nodes,
                    min(MONITORING_STEPS, solver.grid.z_nodes),
                ),
            )

            coords = f.create_group("coordinates")
            coords.create_dataset("r_min", data=solver.grid.r_min)
            coords.create_dataset("r_max", data=solver.grid.r_max)
            coords.create_dataset("z_min", data=solver.grid.z_min)
            coords.create_dataset("z_max", data=solver.grid.z_max)
            coords.create_dataset("r_grid", data=solver.grid.r_grid)
            coords.create_dataset("z_grid", data=solver.grid.z_grid)

            # Add metadata
            meta = f.create_group("metadata")
            meta.create_dataset("last_step", data=0, dtype=np.int16)

    # Update data
    if step % MONITORING_STEPS == 0 or step == solver.grid.z_nodes - 1:
        with File(monitoring_path, "r+") as f:
            step_idx = step + 1
            last_step = f["metadata/last_step"][()]
            last_step_idx = last_step + 1

            if step > last_step:
                f["envelope/peak_rz"][:, last_step_idx : step_idx] = (
                    solver.envelope_tp_rz[:, last_step_idx : step_idx]
                )
                f["envelope/peak_rz"][:, last_step_idx : step_idx] = (
                    solver.envelope_tp_rz[:, last_step_idx : step_idx]
                )
                f["metadata/last_step"][()] = step


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