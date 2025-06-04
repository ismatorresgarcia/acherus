"""Grid parameters for the cylindrical domain."""

from dataclasses import dataclass

import numpy as np
from scipy.fft import fftfreq


@dataclass
class GridParameters:
    """Fixed mesh for cylindrical propagation."""

    # Radial grid initialization
    r_min: float = 0
    r_max: float = 5e-3
    r_i_nodes: int = 10000

    # Axial grid initialization
    z_min: float = 0
    z_max: float = 4
    z_steps: int = 4000
    z_snapshots: int = 5

    # Temporal grid initialization
    t_min: float = -250e-15
    t_max: float = 250e-15
    t_nodes: int = 8192

    def post__init__(self):
        """Post-initialization after defining basic grid parameters."""
        self._init_grid_resolution()
        self._init_grid_arrays()

    @property
    def r_nodes(self):
        """Total number of radial nodes including boundary ones."""
        return self.r_i_nodes + 2

    @property
    def z_nodes(self):
        """Total number of axial nodes."""
        return self.z_steps + 1

    @property
    def z_steps_per_snapshot(self):
        """Number of propagation steps between saved snapshots."""
        return self.z_steps // self.z_snapshots

    def _init_grid_resolution(self):
        """Set grid resolution."""
        self.r_res = np.float64((self.r_max - self.r_min) / (self.r_nodes - 1))
        self.z_res = np.float64((self.z_max - self.z_min) / (self.z_nodes - 1))
        self.t_res = np.float64((self.t_max - self.t_min) / (self.t_nodes - 1))
        self.w_res = np.float64(2 * np.pi / (self.t_nodes * self.t_res))

    def _init_grid_arrays(self):
        """Set 1D and 2D grid arrays."""
        self.r_grid = np.linspace(
            self.r_min, self.r_max, self.r_nodes, dtype=np.float64
        )
        self.z_grid = np.linspace(
            self.z_min, self.z_max, self.z_nodes, dtype=np.float64
        )
        self.t_grid = np.linspace(
            self.t_min, self.t_max, self.t_nodes, dtype=np.float64
        )
        self.w_grid = 2 * np.pi * fftfreq(self.t_nodes, self.t_res, dtype=np.float64)

        self.r_grid_2d, self.t_grid_2d = np.meshgrid(
            self.r_grid, self.t_grid, indexing="ij", dtype=np.float64
        )
