"""Final results data saving module."""

import os

import h5py

from .config import DEFAULT_SAVE_PATH as path


class OutputManager:
    """Handles data storage from the final simulation results."""

    def __init__(self, save_path=path, compression="gzip"):
        """Initialize output manager.

        Parameters:
        - save_path: Directory where data files will be stored
        - compression: Compression method for HDF5 files
        """
        self.save_path = save_path
        self.compression = compression

        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

    def save_snapshots(self, solver):
        """Save full snapshot data to HDF5 file.

        Parameters:
        - solver: Solver instance containing snapshot data
        """
        with h5py.File(f"{self.save_path}/snapshots.h5", "w") as f:
            f.create_dataset(
                "envelope_snapshot_rzt",
                data=solver.envelope_snapshot_rzt,
                compression=self.compression,
                chunks=True,
            )
            f.create_dataset(
                "density_snapshot_rzt",
                data=solver.density_snapshot_rzt,
                compression=self.compression,
                chunks=True,
            )
            f.create_dataset(
                "snap_z_idx", data=solver.snapshot_z_index, compression=self.compression
            )

    def save_diagnostics(self, solver, grid):
        """Save diagnostic data to HDF5 file.

        Parameters:
        - solver: Solver instance containing diagnostic data
        - grid: Grid instance containing coordinate information
        """
        with h5py.File(f"{self.save_path}/final_diagnostic.h5", "w") as f:
            # Envelope data
            envelope_grp = f.create_group("envelope")
            envelope_grp.create_dataset(
                "axis_zt", data=solver.envelope_r0_zt, compression=self.compression
            )
            envelope_grp.create_dataset(
                "peak_rz", data=solver.envelope_tp_rz, compression=self.compression
            )

            # Density data
            density_grp = f.create_group("density")
            density_grp.create_dataset(
                "axis_zt", data=solver.density_r0_zt, compression=self.compression
            )
            density_grp.create_dataset(
                "peak_rz", data=solver.density_tp_rz, compression=self.compression
            )

            # Pulse characteristics
            pulse_grp = f.create_group("pulse")
            pulse_grp.create_dataset(
                "fluence_rz", data=solver.fluence_rz, compression=self.compression
            )
            pulse_grp.create_dataset(
                "radius_z", data=solver.radius_z, compression=self.compression
            )

            # Coordinate information
            coords_grp = f.create_group("coordinates")
            coords_grp.create_dataset("r_min", data=grid.r_min)
            coords_grp.create_dataset("r_max", data=grid.r_max)
            coords_grp.create_dataset("z_min", data=grid.z_min)
            coords_grp.create_dataset("z_max", data=grid.z_max)
            coords_grp.create_dataset("t_min", data=grid.t_min)
            coords_grp.create_dataset("t_max", data=grid.t_max)

    def save_all_results(self, solver, grid):
        """Save all simulation results.

        Parameters:
        - solver: Solver instance containing all data
        - grid: Grid instance containing coordinate information
        """
        self.save_snapshots(solver)
        self.save_diagnostics(solver, grid)
