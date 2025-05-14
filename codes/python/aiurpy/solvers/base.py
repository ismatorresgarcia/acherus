"""Shared solver module."""

import numpy as np

from ..core.initial import initialize_envelope
from ..numerical.shared.fluence import calculate_fluence
from ..numerical.shared.radius import calculate_radius
from ..results.routines import (
    cheap_diagnostics,
    expensive_diagnostics,
    inter_diagnostics,
)


class SolverBase:
    """Base solver class."""

    def __init__(self, material, laser, grid, eqn, method_opt="rk4", ion_model="mpi"):
        """Initialize solver with common parameters.

        Parameters:
        -> material: MaterialParameters object with material properties
        -> laser: LaserPulseParameters object with laser properties
        -> grid: GridParameters object with grid definition
        -> eqn: EquationParameters object with equation parameters
        -> method_opt: Nonlinear solver method (default: "rk4")
        -> ion_model: Ionization model to use (default: "mpi")
        """
        self.material = material
        self.laser = laser
        self.grid = grid
        self.eqn = eqn
        self.method = method_opt
        self.ion_model = ion_model

        # Setup arrays
        self._init_arrays()

        # Compute Runge-Kutta constants
        self.del_z = grid.del_z
        self.del_z_2 = self.del_z * 0.5
        self.del_z_6 = self.del_z / 6
        self.del_t = grid.del_t
        self.del_t_2 = self.del_t * 0.5
        self.del_t_6 = self.del_t / 6

        self.envelope_arguments = (
            self.material.density_neutral,
            eqn.coefficient_plasma,
            eqn.coefficient_mpa,
            eqn.coefficient_kerr,
            eqn.coefficient_raman,
        )
        self.density_arguments = (
            self.material.density_neutral,
            eqn.coefficient_ava,
        )

        # Setup flags
        self.use_raman = material.has_raman

        # Setup tracking variables
        self.snapshot_z_index = np.empty(self.grid.zd.z_snapshots + 1, dtype=int)

    # Setup (pre-allocate) arrays
    def _init_arrays(self):
        """Initialize arrays for simulation."""
        shape_r = (self.grid.r_nodes,)
        shape_rt = (self.grid.r_nodes, self.grid.td.t_nodes)
        shape_rzt = (
            self.grid.r_nodes,
            self.grid.zd.z_snapshots + 1,
            self.grid.td.t_nodes,
        )
        shape_zt = (self.grid.zd.z_steps + 1, self.grid.td.t_nodes)
        shape_rz = (self.grid.r_nodes, self.grid.zd.z_steps + 1)

        # Initialize envelope arrays
        self.envelope_rt = np.empty(shape_rt, dtype=complex)
        self.envelope_next_rt = np.empty_like(self.envelope_rt)
        self.envelope_snapshot_rzt = np.empty(shape_rzt, dtype=complex)
        self.envelope_r0_zt = np.empty(shape_zt, dtype=complex)
        self.envelope_tp_rz = np.empty(shape_rz, dtype=complex)

        # Initialize density arrays
        self.density_rt = np.empty(shape_rt)
        self.density_snapshot_rzt = np.empty(shape_rzt)
        self.density_r0_zt = np.empty(shape_zt)
        self.density_tp_rz = np.empty(shape_rz)

        # Initialize fluence and radius arrays
        self.fluence_r = np.empty(shape_r)
        self.fluence_rz = np.empty(shape_rz)
        self.radius = np.empty(1)
        self.radius_z = np.empty(self.grid.zd.z_steps + 1)

        # Initialize Raman arrays
        self.raman_rt = np.empty(shape_rt, dtype=complex)
        self.draman_rt = np.empty_like(self.raman_rt)
        self.nonlinear_rt = np.empty_like(self.envelope_rt)

        # Initialize RK4 integration arrays
        self.envelope_rk4_stage = np.empty(self.grid.r_nodes, dtype=complex)
        self.density_rk4_stage = np.empty(self.grid.r_nodes)
        self.raman_rk4_stage = np.empty(self.grid.r_nodes, dtype=complex)
        self.draman_rk4_stage = np.empty_like(self.raman_rk4_stage)

        # Initialize ionization arrays
        self.ionization_rate = np.empty_like(self.density_rt)
        self.ionization_sum = np.empty_like(self.density_rt)

    def setup_initial_condition(self):
        """Setup initial conditions."""
        # Initial conditions
        self.envelope_rt[:] = initialize_envelope(self.grid, self.laser)
        self.density_rt[:, 0] = 0
        self.fluence_rz[:, 0] = calculate_fluence(self.envelope_rt, dt=self.grid.del_t)
        self.radius_z[0] = calculate_radius(self.fluence_rz[:, 0], r_g=self.grid.r_grid)

        # Store initial values for diagnostics
        self.envelope_snapshot_rzt[:, 0, :] = self.envelope_rt
        self.envelope_r0_zt[0, :] = self.envelope_rt[0, :]
        self.envelope_tp_rz[:, 0] = self.envelope_rt[:, self.grid.t0_node]
        self.density_snapshot_rzt[:, 0, :].fill(0)
        self.density_r0_zt[0, :] = self.density_rt[0, :]
        self.density_tp_rz[:, 0] = self.density_rt[:, self.grid.t0_node]
        self.snapshot_z_index[0] = 0

    # Method that should exist in all solvers
    def setup_operators(self):
        """Setup numerical operators.

        Look for the function in the FCN and FSS modules.
        """
        raise NotImplementedError("Modules must include setup_operators()")

    # Method that should exist in all solvers
    def solve_step(self):
        """Perform one propagation step.

        Look for the function in the FCN and FSS modules.
        """
        raise NotImplementedError("Module must include solve_step()")

    def propagate(self):
        """Propagate beam through all steps."""
        z_spsnap = self.grid.z_steps_per_snapshot
        z_snaps = self.grid.zd.z_snapshots

        for snap_idx in range(1, z_snaps + 1):
            for steps_snap_idx in range(1, z_spsnap + 1):
                step_idx = (snap_idx - 1) * z_spsnap + steps_snap_idx
                self.solve_step()
                cheap_diagnostics(self, step_idx)
                inter_diagnostics(self, step_idx)
            expensive_diagnostics(self, snap_idx)
