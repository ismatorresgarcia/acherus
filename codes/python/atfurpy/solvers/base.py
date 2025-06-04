"""Shared solver module."""

import numpy as np

from ..core.initial import initialize_envelope
from ..numerical.shared.fluence import compute_fluence
from ..numerical.shared.radius import compute_radius
from ..results.routines import (
    cheap_diagnostics,
    expensive_diagnostics,
    inter_diagnostics,
)


class SolverBase:
    """Base solver class."""

    def __init__(self, material, laser, grid, eqn, method_opt="rk4", ion_model="mpi"):
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
        method_opt : str, default: "rk4"
            Nonlinear solver method chosen.
        ion_model : str, default: "mpi"
            Ionization model chosen.

        """
        self.material = material
        self.laser = laser
        self.grid = grid
        self.eqn = eqn
        self.method = method_opt
        self.ion_model = ion_model

        # Initialize simulation arrays
        self._init_sim_arrays()

        # Initialize frequent arguments
        self.r_nodes = grid.r_nodes
        self.z_nodes = grid.z_nodes
        self.t_nodes = grid.t_nodes
        self.r_res = grid.r_res
        self.z_res = grid.z_res
        self.t_res = grid.t_res
        self.r_grid = grid.r_grid
        self.z_grid = grid.z_grid
        self.t_grid = grid.t_grid
        self.density_neutral = self.material.density_neutral
        self.coefficient_ava = eqn.coefficient_ava
        self.coefficient_plasma = eqn.coefficient_plasma
        self.coefficient_mpa = eqn.coefficient_mpa
        self.coefficient_kerr = eqn.coefficient_kerr
        self.coefficient_raman = eqn.coefficient_raman

        # Set up flags
        self.use_raman = material.has_raman

    # Set up (pre-allocate) arrays
    def _init_sim_arrays(self):
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

        # Initialize envelope arrays
        self.envelope_rt = np.zeros(shape_rt, dtype=np.complex128)
        self.envelope_next_rt = np.zeros_like(self.envelope_rt)
        self.envelope_snapshot_rzt = np.zeros(shape_rzt, dtype=np.complex128)
        self.envelope_r0_zt = np.zeros(shape_zt, dtype=np.complex128)
        self.envelope_tp_rz = np.zeros(shape_rz, dtype=np.complex128)

        # Initialize density arrays
        self.density_rt = np.zeros(shape_rt, dtype=np.float64)
        self.density_snapshot_rzt = np.zeros(shape_rzt, dtype=np.float64)
        self.density_r0_zt = np.zeros(shape_zt, dtype=np.float64)
        self.density_tp_rz = np.zeros(shape_rz, dtype=np.float64)

        # Initialize fluence and radius arrays
        self.fluence_r = np.zeros(shape_r, dtype=np.float64)
        self.fluence_rz = np.zeros(shape_rz, dtype=np.float64)
        self.radius = np.zeros(1, dtype=np.float64)
        self.radius_z = np.zeros(self.z_nodes, dtype=np.float64)

        # Initialize Raman arrays
        self.raman_rt = np.zeros(shape_rt, dtype=np.complex128)
        self.draman_rt = np.zeros_like(self.raman_rt)

        # Initialize nonlinearities array
        self.nonlinear_rt = np.zeros(shape_rt, dtype=np.complex128)

        # Initialize RK4 integration arrays
        self.envelope_rk4_stage = np.zeros(self.r_nodes, dtype=np.complex128)
        self.density_rk4_stage = np.zeros(self.r_nodes, dtype=np.float64)
        self.raman_rk4_stage = np.zeros(self.r_nodes, dtype=np.complex128)
        self.draman_rk4_stage = np.zeros_like(self.raman_rk4_stage)

        # Initialize ionization arrays
        self.ionization_rate = np.zeros(shape_rt, dtype=np.float64)
        self.ionization_sum = np.zeros(shape_rt, dtype=np.float64)

        # Initialize tracking variable
        self.snapshot_z_index = np.zeros(self.grid.z_snapshots + 1, dtype=np.int16)

    def set_initial_conditions(self):
        """Set initial conditions."""
        self.envelope_rt[:] = initialize_envelope(self.grid, self.laser)
        self.fluence_rz[:, 0] = compute_fluence(self.envelope_rt, dt=self.t_res)
        self.radius_z[0] = compute_radius(self.fluence_rz[:, 0], r_g=self.r_grid)

        # Store initial values for diagnostics
        self.envelope_snapshot_rzt[:, 0, :] = self.envelope_rt
        self.envelope_r0_zt[0, :] = self.envelope_rt[0, :]
        self.envelope_tp_rz[:, 0] = self.envelope_rt[
            np.arange(self.r_nodes), np.argmax(np.abs(self.envelope_rt), axis=1)
        ]
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
        z_spsnap = self.grid.z_steps_per_snapshot
        z_snaps = self.grid.z_snapshots

        for snap_idx in range(1, z_snaps + 1):
            for steps_snap_idx in range(1, z_spsnap + 1):
                step_idx = (snap_idx - 1) * z_spsnap + steps_snap_idx
                self.solve_step()
                cheap_diagnostics(self, step_idx)
                inter_diagnostics(self, step_idx)
            expensive_diagnostics(self, snap_idx)
