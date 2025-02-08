"""
Grid Configuration Module.

This module defines the spatial grid configuration for the beam propagation
simulation. It handles:
    - Radial grid parameters (range and resolution)
    - Propagation distance parameters
    - Grid step sizes and derived quantities
    - Special grid points (like axis node)
"""

from dataclasses import dataclass


@dataclass
class GridConfig:
    """
    Configuration class for simulation spatial grid.

    This class handles the initialization and calculation of grid parameters
    for both radial and propagation distance coordinates.

    Attributes:
        ini_radi (float): Initial radius in meters. Defaults to 0
        fin_radi (float): Final radius in meters. Defaults to 20mm
        radi_nodes (int): Number of radial nodes. Defaults to 1000
        ini_dist (float): Initial propagation distance in meters. Defaults to 0
        fin_dist (float): Final propagation distance in meters. Defaults to 3m
        dist_steps (int): Number of propagation steps. Defaults to 1000
        n_radi_nodes (int): Total number of radial nodes including boundaries
        radi_step (float): Radial step size in meters
        dist_step (float): Propagation step size in meters
        axis_node (int): Index of the beam axis node
    """

    ini_radi: float = 0
    fin_radi: float = 2e-2
    radi_nodes: int = 1000
    ini_dist: float = 0
    fin_dist: float = 3
    dist_steps: int = 1000

    def __post_init__(self):
        """
        Initialize derived grid parameters after instance creation.

        Computes:
            - Total number of radial nodes
            - Radial step size
            - Propagation step size
            - Beam axis node index
        """
        self.n_radi_nodes = self.radi_nodes + 2
        self.radi_step = (self.fin_radi - self.ini_radi) / (self.n_radi_nodes - 1)
        self.dist_step = self.fin_dist / self.dist_steps
        self.axis_node = int(-self.ini_radi / self.radi_step)
