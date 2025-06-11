"""Grid parameters for the cylindrical domain."""

from dataclasses import dataclass

import numpy as np


@dataclass
class GridParameters:
    """
    Fixed mesh for cylindrical propagation.

    The following variables must be given in SI units,
    i.e., meters and seconds for the fixed mesh.
    """

    # Radial grid initialization
    r_min: float = 0
    r_max: float = 10e-3
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

    def __post_init__(self):
        """Post-initialization after defining basic grid parameters."""
        assert self.r_max > self.r_min, "r_max must be greater than r_min!"
        assert self.z_max > self.z_min, "z_max must be greater than z_min!"
        self.r_nodes = self.r_i_nodes + 2
        self.z_nodes = self.z_steps + 1
        self.z_steps_per_snapshot = self.z_steps // self.z_snapshots
        self._init_grid_resolution()
        self._init_grid_arrays()

    def _init_grid_resolution(self):
        """Set grid resolution."""
        self.r_res = (self.r_max - self.r_min) / (self.r_nodes - 1)
        self.z_res = (self.z_max - self.z_min) / (self.z_nodes - 1)
        self.t_res = (self.t_max - self.t_min) / (self.t_nodes - 1)
        self.w_res = 2 * np.pi / (self.t_nodes * self.t_res)  # in rad/s

    def _init_grid_arrays(self):
        """Set 1D grid arrays."""
        self.r_grid = np.linspace(
            self.r_min, self.r_max, self.r_nodes, dtype=np.float64
        )
        self.z_grid = np.linspace(
            self.z_min, self.z_max, self.z_nodes, dtype=np.float64
        )
        self.t_grid = np.linspace(
            self.t_min, self.t_max, self.t_nodes, dtype=np.float64
        )
