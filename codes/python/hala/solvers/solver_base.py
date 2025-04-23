"""Base solver module."""

import numpy as np
from tqdm import tqdm

from ..diagnostics.utilities import cheap_diagnostics, expensive_diagnostics
from ..domain.ini_envelope import initialize_envelope
from ..methods.common.fluence import calculate_fluence
from ..methods.common.radius import calculate_radius


class SolverBase:
    """Base solver class."""

    def __init__(self, const, medium, laser, grid, nee, method_opt="rk4"):
        """Initialize solver with common parameters.

        Parameters:
        - const: Constants object with physical constants
        - medium: MediumParameters object with medium properties
        - laser: LaserPulseParameters object with laser properties
        - grid: GridParameters object with grid definition
        - nee: NEEParameters object with equation parameters
        - method_opt: Nonlinear solver method (default: "rk4")
        """
        self.const = const
        self.medium = medium
        self.laser = laser
        self.grid = grid
        self.nee = nee

        self.method = "rk4" if method_opt.upper() == "RK4" else "to_be_defined"

        # Setup arrays
        self._initialize_arrays()

        # Compute Runge-Kutta constants
        self.del_z = grid.del_z
        self.del_z_2 = self.del_z * 0.5
        self.del_z_6 = self.del_z / 6
        self.del_t = grid.del_t
        self.del_t_2 = self.del_t * 0.5
        self.del_t_6 = self.del_t / 6

        self.envelope_arguments = (
            self.medium.number_photons,
            self.medium.density_neutral,
            nee.coefficient_plasma,
            nee.coefficient_mpa,
            nee.coefficient_kerr,
            nee.coefficient_raman,
        )
        self.density_arguments = (
            self.medium.number_photons,
            self.medium.density_neutral,
            nee.coefficient_ofi,
            nee.coefficient_ava,
        )

        # Setup flags
        self.use_raman = medium.has_raman

        # Setup tracking variables
        self.snapshot_z_index = np.empty(self.grid.number_snapshots + 1, dtype=int)

    # Setup (pre-allocate) arrays
    def _initialize_arrays(self):
        """Initialize arrays for simulation."""
        shape_r = (self.grid.nodes_r,)
        shape_rt = (self.grid.nodes_r, self.grid.nodes_t)
        shape_rzt = (
            self.grid.nodes_r,
            self.grid.number_snapshots + 1,
            self.grid.nodes_t,
        )
        shape_zt = (self.grid.number_steps + 1, self.grid.nodes_t)
        shape_rz = (self.grid.nodes_r, self.grid.number_steps + 1)

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
        self.radius_z = np.empty(self.grid.number_steps + 1)

        # Initialize Raman arrays
        self.raman_rt = np.empty(shape_rt, dtype=complex)
        self.draman_rt = np.empty_like(self.raman_rt)
        self.nonlinear_rt = np.empty_like(self.envelope_rt)

        # Initialize arrays for RK4 integration
        self.envelope_rk4_stage = np.empty(self.grid.nodes_r, dtype=complex)
        self.density_rk4_stage = np.empty(self.grid.nodes_r)
        self.raman_rk4_stage = np.empty(self.grid.nodes_r, dtype=complex)
        self.draman_rk4_stage = np.empty_like(self.raman_rk4_stage)

    def setup_initial_condition(self):
        """Setup initial conditions."""
        # Initial conditions
        self.envelope_rt[:] = initialize_envelope(self.const, self.grid, self.laser)
        self.density_rt[:, 0] = 0
        self.fluence_rz[:, 0] = calculate_fluence(self.envelope_rt, dt=self.grid.del_t)
        self.radius_z[0] = calculate_radius(self.fluence_rz[:, 0], r_g=self.grid.r_grid)

        # Store initial values for diagnostics
        self.envelope_snapshot_rzt[:, 0, :] = self.envelope_rt
        self.envelope_r0_zt[0, :] = self.envelope_rt[self.grid.node_r0, :]
        self.envelope_tp_rz[:, 0] = self.envelope_rt[:, self.grid.node_t0]
        self.density_snapshot_rzt[:, 0, :].fill(0)
        self.density_r0_zt[0, :] = self.density_rt[self.grid.node_r0, :]
        self.density_tp_rz[:, 0] = self.density_rt[:, self.grid.node_t0]
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
        steps = self.grid.number_steps
        steps_snap = self.grid.steps_per_snapshot
        n_snaps = self.grid.number_snapshots

        with tqdm(total=steps, desc="Progress") as pbar:
            for snap_idx in range(1, n_snaps + 1):
                for steps_snap_idx in range(1, steps_snap + 1):
                    step_idx = (snap_idx - 1) * steps_snap + steps_snap_idx
                    self.solve_step()
                    cheap_diagnostics(self, step_idx)
                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "snap": snap_idx,
                            "step_per_snap": steps_snap_idx,
                            "step": step_idx,
                        }
                    )
                expensive_diagnostics(self, snap_idx)
