"""Module for shared variables and methods used by the solvers."""

import numpy as np

from ..data.diagnostics import cheap_diagnostics, expensive_diagnostics
from ..functions.density import compute_density_nr, compute_density_r
from ..functions.fluence import compute_fluence


class Shared:
    """Class for shared variables and methods used by the different solvers."""

    def __init__(self, config, medium, laser, grid, eqn, ion, output):
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
        ion : object
            Contains the ionization model parameters.
        output : object
            Contains the output manager methods.

        """
        self.medium = medium
        self.laser = laser
        self.grid = grid
        self.eqn = eqn
        self.ion = ion
        self.output = output
        self.medium_n = config.medium_name
        self.dens_meth = config.density_method
        self.dens_meth_rtol = config.density_method_par.rtol
        self.dens_meth_atol = config.density_method_par.atol
        self.recomb_c = config.medium_par.recombination_rate

        if self.recomb_c is not None:
            self._compute_density = self._compute_density_recombination
        else:
            self._compute_density = self._compute_density_no_recombination

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
        self.number_photons = eqn.n_k
        self.w_0 = laser.frequency_0
        self.k_0 = laser.wavenumber_0
        _, _, self.k_1 = medium.dispersion_properties(self.w_0)
        self.density_n = medium.neutral_density
        self.density_ini = medium.initial_density
        self.avalanche_c = eqn.ava_c
        self.plasma_c = eqn.plasma_c
        self.mpa_c = eqn.mpa_c
        self.kerr_c = eqn.kerr_c

        # Set up array shapes
        self.shape_r = (self.r_nodes,)
        self.shape_rt = (self.r_nodes, self.t_nodes)
        self.shape_rzt = (
            self.r_nodes,
            self.z_snapshots + 1,
            self.t_nodes,
        )
        self.shape_zt = (self.z_nodes, self.t_nodes)
        self.shape_rz = (self.r_nodes, self.z_nodes)

        # Initialize simulation arrays
        self.init_simulation_arrays()

    # Set up (pre-allocate) arrays
    def init_simulation_arrays(self):
        """Initialize arrays for simulation."""
        self.envelope_rt = np.zeros(self.shape_rt, dtype=np.complex128)
        self.envelope_snapshot_rzt = np.empty(self.shape_rzt, dtype=np.complex128)
        self.envelope_r0_zt = np.empty(self.shape_zt, dtype=np.complex128)
        self.envelope_tp_rz = np.empty(self.shape_rz, dtype=np.complex128)
        self.intensity_rt = np.zeros(self.shape_rt, dtype=np.float64)

        self.density_rt = np.zeros(self.shape_rt, dtype=np.float64)
        self.density_snapshot_rzt = np.empty(self.shape_rzt, dtype=np.float64)
        self.density_r0_zt = np.empty(self.shape_zt, dtype=np.float64)
        self.density_tp_rz = np.empty(self.shape_rz, dtype=np.float64)

        self._dens_rhs_buf = np.empty(self.shape_r, dtype=np.float64)
        self._dens_tmp_buf = np.empty(self.shape_r, dtype=np.float64)
        self._dens_init_buf = np.empty(self.shape_r, dtype=np.float64)

        self.fluence_r = np.zeros(self.shape_r, dtype=np.float64)
        self.fluence_rz = np.zeros(self.shape_rz, dtype=np.float64)

        self.nonlinear_rt = np.zeros(self.shape_rt, dtype=np.complex128)

        self.ionization_rate = np.zeros(self.shape_rt, dtype=np.float64)
        self.intensity_to_rate = self.ion.intensity_to_rate

        self.snapshot_z_index = np.zeros(self.z_snapshots + 1, dtype=np.uint32)

    def _compute_density_no_recombination(self):
        """Compute density without recombination term."""
        compute_density_nr(
            self.intensity_rt[:-1, :],
            self.density_rt[:-1, :],
            self.ionization_rate[:-1, :],
            self.t_grid,
            self.density_n,
            self._dens_init_buf[:-1],
            self.avalanche_c,
            self.dens_meth,
            self.dens_meth_rtol,
            self.dens_meth_atol,
            self._dens_rhs_buf[:-1],
            self._dens_tmp_buf[:-1],
        )

    def _compute_density_recombination(self):
        """Compute density with recombination term."""
        compute_density_r(
            self.intensity_rt[:-1, :],
            self.density_rt[:-1, :],
            self.ionization_rate[:-1, :],
            self.t_grid,
            self.density_n,
            self._dens_init_buf[:-1],
            self.avalanche_c,
            self.recomb_c,
            self.dens_meth,
            self.dens_meth_rtol,
            self.dens_meth_atol,
            self._dens_rhs_buf[:-1],
            self._dens_tmp_buf[:-1],
        )

    def set_initial_conditions(self):
        """Set initial conditions."""
        self.envelope_rt[:] = self.laser.init_envelope()
        self.density_rt[:, 0] = self.density_ini
        self._dens_init_buf[:] = self.density_ini
        self.fluence_rz[:, 0] = compute_fluence(self.envelope_rt, t_g_a=self.t_grid)

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

    def propagate(self):
        """Propagate beam through all steps."""
        z_spsnap = self.z_steps_per_snapshot
        z_snaps = self.z_snapshots

        for snap_idx in range(1, z_snaps + 1):
            for steps_snap_idx in range(1, z_spsnap + 1):
                step_idx = (snap_idx - 1) * z_spsnap + steps_snap_idx
                self.solve_step(step_idx)
                cheap_diagnostics(self, step_idx)
                self.output.monitoring_diagnostics(self, step_idx)
            expensive_diagnostics(self, snap_idx)
