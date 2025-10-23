"""Shared solver module."""

import numpy as np

from ..data.diagnostics import (
    cheap_diagnostics,
    expensive_diagnostics,
    monitoring_diagnostics,
)
from ..functions.fluence import compute_fluence
from ..functions.radius import compute_radius
from ..physics.keldysh import compute_ppt_rate


class SolverBase:
    """Base solver class."""

    def __init__(
        self,
        config,
        medium,
        laser,
        grid,
        eqn,
        output
    ):
        """Initialize solver with common parameters.

        Parameters
        ----------
        config: object
            Contains the simulation options.
        medium : object
            Contains the chosen medium parameters.
        laser : object
            Contains the laser input parameters.
        grid : object
            Contains the grid input parameters.
        eqn : object
            Contains the equation parameters.
        output : object
            Contains the output manager methods.

        """
        self.medium = medium
        self.laser = laser
        self.grid = grid
        self.eqn = eqn
        self.output = output
        self.medium_n = config.medium_name
        self.dens_meth = config.density_method
        self.dens_meth_ini_step = config.density_method_par.ini_step
        self.dens_meth_rtol = config.density_method_par.rtol
        self.dens_meth_atol = config.density_method_par.atol
        self.nlin_meth = config.nonlinear_method
        self.ion_model = config.ionization_model

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
        self.w_grid = grid.w_grid
        self.number_photons = self.eqn.n_k
        self.mpi_c = self.eqn.mpi_c
        self.w_0 = self.laser.frequency_0
        self.k_0 = self.laser.wavenumber_0
        self.k_1 = self.medium.constant_k1
        self.k_2 = self.medium.constant_k2
        self.density_n = self.medium.density_neutral
        self.density_ini = self.medium.density_initial
        self.avalanche_c = eqn.ava_c
        self.plasma_c = eqn.plasma_c
        self.mpa_c = eqn.mpa_c
        self.kerr_c = eqn.kerr_c
        self.raman_c = eqn.raman_c
        self.raman_ode1 = eqn.raman_ode1
        self.raman_ode2 = eqn.raman_ode2

        # Set up flags
        self.use_raman = medium.has_raman

        # Initialize simulation arrays
        self.init_simulation_arrays()

    # Set up (pre-allocate) arrays
    def init_simulation_arrays(self):
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
        self.envelope_rt = np.zeros(shape_rt, dtype=np.complex128)
        self.envelope_next_rt = np.zeros_like(self.envelope_rt)
        self.envelope_snapshot_rzt = np.zeros(shape_rzt, dtype=np.complex128)
        self.envelope_r0_zt = np.zeros(shape_zt, dtype=np.complex128)
        self.envelope_tp_rz = np.zeros(shape_rz, dtype=np.complex128)
        self.intensity_rt = np.zeros(shape_rt, dtype=np.float64)

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
        self.raman_rt = np.zeros(shape_rt, dtype=np.float64)
        self.raman_aux = np.zeros(shape_rt, dtype=np.complex128)

        # Initialize nonlinearities array
        self.nonlinear_rt = np.zeros_like(self.envelope_rt)

        # Initialize ionization arrays
        self.ionization_rate = np.zeros_like(self.density_rt)

        # Initialize tracking variable
        self.snapshot_z_index = np.zeros(self.z_snapshots + 1, dtype=np.int16)

        # Initialize PPT rate arrays
        if self.ion_model == "PPT":
            self.peak_intensity, self.ppt_rate, self.i_factor = compute_ppt_rate(
                self.medium, self.laser
            )

    def set_initial_conditions(self):
        """Set initial conditions."""
        self.envelope_rt[:] = self.laser.init_envelope()
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

    def solve_step(self, step):
        """Perform one propagation step.

        Look for the function in the FCN and SSCN modules.
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
                self.solve_step(step_idx)
                cheap_diagnostics(self, step_idx)
                monitoring_diagnostics(self, step_idx)
            expensive_diagnostics(self, snap_idx)
