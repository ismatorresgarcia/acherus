"""Shared solver module."""

import numpy as np

from ..data.routines import (
    cheap_diagnostics,
    expensive_diagnostics,
    monitoring_diagnostics,
)
from ..mathematics.shared.fluence import compute_fluence
from ..mathematics.shared.radius import compute_radius
from ..physics.initial_beam import initialize_envelope


class SolverBase:
    """Base solver class."""

    def __init__(
        self,
        material,
        laser,
        grid,
        eqn,
        method_d_opt="RK4",
        method_r_opt="RK4",
        method_nl_opt="RK4",
        ion_model="MPI",
    ):
        """Initialize solver with common parameters.

        Parameters
        ----------
        material : object
            Contains the chosen medium parameters.
        laser : object
            Contains the laser input parameters.
        grid : object
            Contains the grid input parameters.
        eqn : object
            Contains the equation parameters.
        method_d_opt : str, default: "RK4"
            Density solver method chosen.
        method_r_opt : str, default: "RK4"
            Raman solver method chosen.
        method_nl_opt : str, default: "RK4"
            Nonlinear solver method chosen.
        ion_model : str, default: "MPI"
            Ionization model chosen.

        """
        self.material = material
        self.laser = laser
        self.grid = grid
        self.eqn = eqn
        self.method_d = method_d_opt
        self.method_r = method_r_opt
        self.method_nl = method_nl_opt
        self.ion_model = ion_model

        # Initialize frequent arguments
        self.r_nodes = grid.r_nodes
        self.z_nodes = grid.z_nodes
        self.t_nodes = grid.t_nodes
        self.z_snapshots = grid.z_snapshots
        self.z_steps_per_snapshot = grid.z_steps_per_snapshot
        self.r_res = grid.r_res
        self.z_res = grid.z_res
        self.t_res = grid.t_res
        self.r_grid = grid.r_grid
        self.z_grid = grid.z_grid
        self.t_grid = grid.t_grid
        self.number_photons = self.eqn.n_k
        self.hydrogen_f0 = self.eqn.hyd_f0
        self.hydrogen_nc = self.eqn.hyd_nc
        self.keldysh_c = self.eqn.keld_c
        self.index_c = self.eqn.idx_c
        self.ppt_c = self.eqn.ppt_c
        self.mpi_c = self.eqn.mpi_c
        self.w_0 = self.laser.frequency_0
        self.k_n = self.laser.wavenumber
        self.k_pp = self.material.constant_gvd
        self.density_n = self.material.density_neutral
        self.density_ini = self.material.density_initial
        self.avalanche_c = eqn.ava_c
        self.plasma_c = eqn.plasma_c
        self.mpa_c = eqn.mpa_c
        self.kerr_c = eqn.kerr_c
        self.raman_c = eqn.raman_c
        self.raman_c1 = self.eqn.raman_c1
        self.raman_c2 = self.eqn.raman_c2

        # Set up flags
        self.use_raman = material.has_raman

        # Initialize simulation arrays
        self._init_simulation_arrays()

    # Set up (pre-allocate) arrays
    def _init_simulation_arrays(self):
        """Initialize arrays for simulation."""
        shape_r = (self.r_nodes,)
        shape_rt = (self.r_nodes, self.t_nodes)
        shape_rzt = (
            self.r_nodes,
            self.z_snapshots + 1,
            self.t_nodes,
        )
        shape_zt = (self.z_nodes, self.t_nodes)
        shape_rz = (self.r_nodes, self.z_nodes)

        # Initialize envelope-related arrays
        self.envelope_rt = np.empty(shape_rt, dtype=np.complex128)
        self.envelope_next_rt = np.empty_like(self.envelope_rt)
        self.envelope_snapshot_rzt = np.empty(shape_rzt, dtype=np.complex128)
        self.envelope_r0_zt = np.empty(shape_zt, dtype=np.complex128)
        self.envelope_tp_rz = np.empty(shape_rz, dtype=np.complex128)
        self.intensity_rt = np.empty(shape_rt, dtype=np.float64)

        # Initialize density arrays
        self.density_rt = np.empty(shape_rt, dtype=np.float64)
        self.density_snapshot_rzt = np.empty(shape_rzt, dtype=np.float64)
        self.density_r0_zt = np.empty(shape_zt, dtype=np.float64)
        self.density_tp_rz = np.empty(shape_rz, dtype=np.float64)

        # Initialize fluence and radius arrays
        self.fluence_r = np.empty(shape_r, dtype=np.float64)
        self.fluence_rz = np.empty(shape_rz, dtype=np.float64)
        self.radius = np.empty(1, dtype=np.float64)
        self.radius_z = np.empty(self.z_nodes, dtype=np.float64)

        # Initialize Raman arrays
        self.raman_rt = np.empty(shape_rt, dtype=np.float64)
        self.draman_rt = np.empty_like(self.raman_rt)

        # Initialize nonlinearities array
        self.nonlinear_rt = np.empty_like(self.envelope_rt)

        # Initialize ionization arrays
        self.ionization_rate = np.empty_like(self.density_rt)
        self.ionization_sum = np.empty_like(self.density_rt)

        # Initialize tracking variable
        self.snapshot_z_index = np.zeros(self.z_snapshots + 1, dtype=np.int16)

    def set_initial_conditions(self):
        """Set initial conditions."""
        self.envelope_rt[:] = initialize_envelope(self.grid, self.laser)
        self.density_rt[:, 0] = self.density_ini
        self.fluence_rz[:, 0] = compute_fluence(self.envelope_rt, t_g_a=self.t_grid)
        self.radius_z[0] = compute_radius(self.fluence_rz[:, 0], r_g_a=self.r_grid)

        # Store initial values for diagnostics
        self.envelope_snapshot_rzt[:, 0, :] = self.envelope_rt
        self.envelope_r0_zt[0, :] = self.envelope_rt[0, :]
        self.envelope_tp_rz[:, 0] = self.envelope_rt[
            np.arange(self.r_nodes), np.argmax(np.abs(self.envelope_rt), axis=1)
        ]
        self.density_snapshot_rzt[:, 0, :] = self.density_ini
        self.density_r0_zt[0, :] = self.density_rt[0, :]
        self.density_tp_rz[:, 0] = self.density_rt[
            np.arange(self.r_nodes), np.argmax(self.density_rt, axis=1)
        ]

    # Methods that should exist in all solvers
    def set_operators(self):
        """Set numerical operators.

        Look for the function in the FCN and FSS modules.
        """
        raise NotImplementedError("Modules must include set_operators()")

    def solve_step(self):
        """Perform one propagation step.

        Look for the function in the FCN and FSS modules.
        """
        raise NotImplementedError("Module must include solve_step()")

    # Propagation method
    def propagate(self):
        """Propagate beam through all steps."""
        z_spsnap = self.z_steps_per_snapshot
        z_snaps = self.z_snapshots

        for snap_idx in range(1, z_snaps + 1):
            for steps_snap_idx in range(1, z_spsnap + 1):
                step_idx = (snap_idx - 1) * z_spsnap + steps_snap_idx
                self.solve_step()
                cheap_diagnostics(self, step_idx)
                monitoring_diagnostics(self, step_idx)
            expensive_diagnostics(self, snap_idx)
