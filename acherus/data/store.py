"""Final results data saving module."""

from pathlib import Path

from h5py import File

from .paths import get_base_dir

MONITORING_STEPS = 100


class OutputManager:
    """Handles simulation data storage."""

    def __init__(self, save_path=None, compression="gzip", compression_opts=9):
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
        self.save_path = Path(save_path) if save_path else get_base_dir()
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.compression = compression
        self.compression_opts = compression_opts

        self.snapshots_path = self.save_path / "acherus_snapshots.h5"
        self.diagnostics_path = self.save_path / "acherus_diagnostics.h5"
        self.monitoring_path = self.save_path / "acherus_monitoring.h5"

    def save_snapshots(self, solver):
        """Save full snapshot data to HDF5 file.

        Parameters
        ----------
        solver : object
            Solver object with data.

        """
        with File(self.snapshots_path, "w") as f:
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
        with File(self.diagnostics_path, "w") as f:
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

            # Fluence data
            fluence_grp = f.create_group("fluence")
            fluence_grp.create_dataset(
                "fluence_rz", data=solver.fluence_rz, compression=self.compression
            )

            # Coordinate information
            coords_grp = f.create_group("coordinates")
            coords_grp.create_dataset("r_min", data=grid.r_min)
            coords_grp.create_dataset("r_max", data=grid.r_max)
            coords_grp.create_dataset("z_min", data=grid.z_min)
            coords_grp.create_dataset("z_max", data=grid.z_max)
            coords_grp.create_dataset("t_min", data=grid.t_min)
            coords_grp.create_dataset("t_max", data=grid.t_max)

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

    def monitoring_diagnostics(self, solver, step):
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
            with File(self.monitoring_path, "w") as f:
                envelope_grp = f.create_group("envelope")
                envelope_grp.create_dataset(
                    "peak_rz",
                    shape=(solver.grid.r_nodes, solver.grid.z_nodes),
                    maxshape=(solver.grid.r_nodes, None),
                    dtype=solver.envelope_tp_rz.dtype,
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
                meta.create_dataset("last_step", data=0, dtype="uint32")

        # Update data
        if step % MONITORING_STEPS == 0 or step == solver.grid.z_nodes - 1:
            with File(self.monitoring_path, "r+") as f:
                step_idx = step + 1
                last_step = f["metadata/last_step"][()]
                last_step_idx = last_step + 1

                if step > last_step:
                    f["envelope/peak_rz"][:, last_step_idx:step_idx] = (
                        solver.envelope_tp_rz[:, last_step_idx:step_idx]
                    )
                    f["metadata/last_step"][()] = step
