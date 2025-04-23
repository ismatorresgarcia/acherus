"""Diagnosing tools module."""

import os
import sys

import h5py
import numpy as np

from . import DEFAULT_SAVE_PATH, DIAGNOSE_SAVE_INTERVAL


def validate_step(solver, exit_on_error=True):
    """
    Validate numerical results from solver state.

    Parameters:
    - solver: Solver instance to validate
    - exit_on_error: Whether to exit program on validation failure (default: True)

    Returns:
    - bool: True if valid, False if invalid (when exit_on_error is False)
    """
    # Check envelope for non-finite values
    if np.any(~np.isfinite(solver.envelope_rt)):
        if exit_on_error:
            print("ERROR: Non-finite values detected in envelope")
            sys.exit(1)
        else:
            print("WARNING: Non-finite values detected in envelope")
            return False

    # Check density for non-finite values
    if np.any(~np.isfinite(solver.density_rt)):
        if exit_on_error:
            print("ERROR: Non-finite values detected in density")
            sys.exit(1)
        else:
            print("WARNING: Non-finite values detected in density")
            return False

    return True


def cheap_diagnostics(solver, step):
    """Save memory cheap diagnostics data for current step.

    Parameters:
    - solver: Solver instance containing the data to save
    - step: Current propagation step
    """
    # Validate current solver state
    validate_step(solver)

    node_r0 = solver.grid.node_r0
    envelope_rt = solver.envelope_rt
    density_rt = solver.density_rt

    axis_data_envelope = envelope_rt[node_r0]
    axis_data_density = density_rt[node_r0]
    axis_data_intensity = np.abs(axis_data_envelope)

    peak_node_intensity = np.argmax(axis_data_intensity)
    peak_node_density = np.argmax(axis_data_density)

    solver.envelope_r0_zt[step] = axis_data_envelope
    solver.envelope_tp_rz[:, step] = envelope_rt[:, peak_node_intensity]
    solver.density_r0_zt[step] = axis_data_density
    solver.density_tp_rz[:, step] = density_rt[:, peak_node_density]
    solver.fluence_rz[:, step] = solver.fluence_r
    solver.radius_z[step] = solver.radius[0]

    # Save intermediate diagnostics
    intermediate_diagnostics(solver, step)


def expensive_diagnostics(solver, step):
    """Save memory expensive diagnostics data for current step.

    Parameters:
    - solver: Solver instance containing the data to save
    - step: Current propagation step index for snapshots (1-based)
    """
    solver.envelope_snapshot_rzt[:, step, :] = solver.envelope_rt
    solver.density_snapshot_rzt[:, step, :] = solver.density_rt
    solver.snapshot_z_index[step] = (
        solver.snapshot_z_index[step - 1] + solver.grid.steps_per_snapshot
    )


def intermediate_diagnostics(solver, step):
    """Save diagnostics progressively every desired number of steps.

    Parameters:
    - solver: Solver instance containing the data to save
    - step: Current propagation step
    """
    temp_diagnostic = f"{DEFAULT_SAVE_PATH}/temp_diagnostic.h5"

    if step == 1:
        os.makedirs(DEFAULT_SAVE_PATH, exist_ok=True)
        with h5py.File(temp_diagnostic, "w") as f:
            envelope_grp = f.create_group("envelope")
            envelope_grp.create_dataset(
                "peak_rz",
                shape=(solver.grid.nodes_r, solver.grid.number_steps + 1),
                maxshape=(solver.grid.nodes_r, None),
                dtype=complex,
                compression="gzip",
                chunks=(solver.grid.nodes_r, min(100, solver.grid.number_steps + 1)),
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
            meta.create_dataset("last_step", data=0, dtype=int)

    # Update data
    if step % DIAGNOSE_SAVE_INTERVAL == 0 or step == solver.grid.number_steps:
        with h5py.File(temp_diagnostic, "r+") as f:
            last_step = f["metadata/last_step"][()]

            if step > last_step:
                f["envelope/peak_rz"][:, last_step + 1 : step + 1] = (
                    solver.envelope_tp_rz[:, last_step + 1 : step + 1]
                )
                f["metadata/last_step"][()] = step
