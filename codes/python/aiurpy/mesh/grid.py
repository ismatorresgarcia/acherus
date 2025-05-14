"""Grid parameters for the cylindrical domain."""

from dataclasses import dataclass

import numpy as np
from scipy.fft import fftfreq


@dataclass
class RadialGrid:
    """Radial (r) grid properties."""

    r_min: float = 0
    r_max: float = 5e-3
    r_i_nodes: int = 10000


@dataclass
class AxialGrid:
    """Axial (z) grid properties."""

    z_min: float = 0
    z_max: float = 4
    z_steps: int = 4000
    z_snapshots: int = 5


@dataclass
class TemporalGrid:
    """Temporal (t) grid properties."""

    t_min: float = -250e-15
    t_max: float = 250e-15
    t_nodes: int = 8192


class GridParameters:
    "Grid parameters calculations."

    def __init__(self):
        """Initialize grid properties."""
        self.rd = RadialGrid()
        self.zd = AxialGrid()
        self.td = TemporalGrid()

        # Initialize functions
        self._init_grid_resolution()
        self._init_grid_arrays()

    @property
    def r_nodes(self):
        "Total number of radial nodes including boundary ones."
        return self.rd.r_i_nodes + 2

    @property
    def t0_node(self):
        "Time node for which t = 0."
        return self.td.t_nodes // 2

    @property
    def z_steps_per_snapshot(self):
        "Number of propagation steps between saved snapshots."
        return self.zd.z_steps // self.zd.z_snapshots

    def _init_grid_resolution(self):
        "Setup domain parameters."
        # Calculate steps
        self.del_r = (self.rd.r_max - self.rd.r_min) / (self.r_nodes - 1)
        self.del_z = (self.zd.z_max - self.zd.z_min) / self.zd.z_steps
        self.del_t = (self.td.t_max - self.td.t_min) / (self.td.t_nodes - 1)
        self.del_w = 2 * np.pi / (self.td.t_nodes * self.del_t)

    def _init_grid_arrays(self):
        "Setup 1D and 2D grid arrays."
        self.r_grid = np.linspace(self.rd.r_min, self.rd.r_max, self.r_nodes)
        self.z_grid = np.linspace(self.zd.z_min, self.zd.z_max, self.zd.z_steps + 1)
        self.t_grid = np.linspace(self.td.t_min, self.td.t_max, self.td.t_nodes)
        self.w_grid = 2 * np.pi * fftfreq(self.td.t_nodes, self.del_t)

        self.r_grid_2d, self.t_grid_2d = np.meshgrid(
            self.r_grid, self.t_grid, indexing="ij"
        )
