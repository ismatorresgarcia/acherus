"""Grid for the cylindrical domain."""

import numpy as np


class Grid:
    """
    Fixed mesh for cylindrical propagation.

    The following variables must be given in SI units,
    i.e., meters and seconds for the fixed mesh.
    """

    def __init__(self, space_par, axis_par, time_par):
        # Initialize parameters
        self.r_nodes = space_par.nodes
        self.r_min = space_par.space_min
        self.r_max = space_par.space_max
        self.z_nodes = axis_par.nodes
        self.z_min = axis_par.axis_min
        self.z_max = axis_par.axis_max
        self.z_snapshots = axis_par.snapshots
        self.t_nodes = time_par.nodes
        self.t_min = time_par.time_min
        self.t_max = time_par.time_max
        self.z_steps_per_snapshot = (self.z_nodes - 1) // self.z_snapshots

        self._init_grid_resolution()
        self._init_grid_arrays()

    def _init_grid_resolution(self):
        """Set grid resolution."""
        self.r_res = (self.r_max - self.r_min) / (self.r_nodes - 1)
        self.z_res = (self.z_max - self.z_min) / (self.z_nodes - 1)
        self.t_res = (self.t_max - self.t_min) / (self.t_nodes - 1)

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
