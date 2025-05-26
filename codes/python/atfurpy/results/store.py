"""Final results data saving module."""

import h5py

from .paths import sim_dir as path

snapshots_path = path / "snapshots.h5"
diagnostic_path = path / "final_diagnostic.h5"


class OutputManager:
    """Handles data storage from the final simulation results."""

    def __init__(self, save_path=path, compression="gzip", compression_opts=9):
        """Initialize output manager.

        Parameters
        ----------
        save_path : str
            Directory where data files will be stored.
        compression : str, default: "gzip"
            Compression method for HDF5 files.
        compression_opts : integer, default: 9
            Compression level chosen.

        """
        self.save_path = save_path
        self.compression = compression
        self.compression_opts = compression_opts

    def save_snapshots(self, solver):
        """Save full snapshot data to HDF5 file.

        Parameters
        ----------
        solver : object
            Solver object with data.

        """
        with h5py.File(snapshots_path, "w") as f:
            f.create_dataset(
                "envelope_snapshot_rzt",
                data=solver.envelope_snapshot_rzt,
                compression=self.compression,
                compression_opts=self.compression_opts,
                chunks=True,
                shuffle=True,
            )
            f.create_dataset(
                "density_snapshot_rzt",
                data=solver.density_snapshot_rzt,
                compression=self.compression,
                compression_opts=self.compression_opts,
                chunks=True,
                shuffle=True,
            )
            f.create_dataset(
                "snap_z_idx", data=solver.snapshot_z_index, compression=self.compression
            )

    def save_diagnostics(self, solver, grid):
        """Save diagnostic data to HDF5 file.

        Parameters
        ----------
        solver : object
            Solver object with data.
        grid : object
            Contains the grid input parameters.

        """
        with h5py.File(diagnostic_path, "w") as f:
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
            coords_grp.create_dataset("r_min", data=grid.rd.r_min)
            coords_grp.create_dataset("r_max", data=grid.rd.r_max)
            coords_grp.create_dataset("z_min", data=grid.zd.z_min)
            coords_grp.create_dataset("z_max", data=grid.zd.z_max)
            coords_grp.create_dataset("t_min", data=grid.td.t_min)
            coords_grp.create_dataset("t_max", data=grid.td.t_max)

    def save_results(self, solver, grid):
        """Save all simulation results.

        Parameters
        ----------
        solver : object
            Solver object with data.
        grid : object
            Contains the grid input parameters.

        """
        self.save_snapshots(solver)
        self.save_diagnostics(solver, grid)
