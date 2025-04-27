"""Grid parameters for the cylindrical domain."""

import numpy as np
from scipy.fft import fftfreq


class GridParameters:
    "Spatial and temporal grid parameters."

    def __init__(self, const):
        # Radial domain
        self.r_min = 0
        self.r_max = 5e-3
        self.nodes_r_i = 10000

        # Distance domain
        self.z_min = 0
        self.z_max = 4
        self.number_steps = 4000
        self.number_snapshots = 5

        # Time domain
        self.t_min = -250e-15
        self.t_max = 250e-15
        self.nodes_t = 8192

        # Initialize derived parameters functions
        self._setup_derived_parameters()
        self._setup_arrays(const)

    @property
    def nodes_r(self):
        "Total number of radial nodes for boundary conditions."
        return self.nodes_r_i + 2

    @property
    def steps_per_snapshot(self):
        "Number of propagation steps between saved snapshots."
        return self.number_steps // self.number_snapshots

    def _setup_derived_parameters(self):
        "Setup domain parameters."
        # Calculate steps
        self.del_r = (self.r_max - self.r_min) / (self.nodes_r - 1)
        self.del_z = (self.z_max - self.z_min) / self.number_steps
        self.del_t = (self.t_max - self.t_min) / (self.nodes_t - 1)
        self.del_w = 2 * np.pi / (self.nodes_t * self.del_t)

        # Calculate nodes for r = 0 and t = 0
        self.node_r0 = int(-self.r_min / self.del_r)
        self.node_t0 = self.nodes_t // 2

    def _setup_arrays(self, const):
        "Setup grid arrays."
        # 1D
        self.r_grid = np.linspace(self.r_min, self.r_max, self.nodes_r)
        self.z_grid = np.linspace(self.z_min, self.z_max, self.number_steps + 1)
        self.t_grid = np.linspace(self.t_min, self.t_max, self.nodes_t)
        self.w_grid = 2 * const.pi * fftfreq(self.nodes_t, self.del_t)

        # 2D
        self.r_grid_2d, self.t_grid_2d = np.meshgrid(
            self.r_grid, self.t_grid, indexing="ij"
        )
