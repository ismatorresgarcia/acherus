"""
Gaussian Beam Simulation Module.

This module implements the numerical simulation of an ultra-intense and ultra-short
laser pulse propagation using the Unidirectional Pulse Propagation Equation (UPPE).
It handles both numerical propagation using Crank-Nicolson method and analytical
solutions for comparison.

The simulation includes:
    - Diffraction (for the transverse direction)
    - Finite Differences Method (FDM) with Crank-Nicolson scheme
    - Gaussian initial condition
    - Neumann-Dirichlet boundary conditions
"""

import numpy as np

from .beam_config import BeamConfig
from .crank_nicolson import CrankNicolsonSolver
from .grid_config import GridConfig


class GaussianBeamSimulation:
    """
    A class to simulate Gaussian beam propagation using UPPE.

    This class handles both the numerical simulation using Crank-Nicolson method
    and the analytical solution for comparison purposes.

    Attributes:
        beam (BeamConfig): Beam parameters configuration
        grid (GridConfig): Grid parameters configuration
        solver (CrankNicolsonSolver): Solver for numerical propagation
        envelope (np.ndarray): Numerical solution array
        envelope_s (np.ndarray): Analytical solution array
        radi_array (np.ndarray): Radial coordinates array
        dist_array (np.ndarray): Distance coordinates array
        radi_2d_array (np.ndarray): 2D radial coordinates mesh
        dist_2d_array (np.ndarray): 2D distance coordinates mesh
    """

    def __init__(self, beam_config: BeamConfig, grid_config: GridConfig):
        """
        Initialize simulation with beam and grid configurations.

        Args:
            beam_config (BeamConfig): Configuration object containing beam parameters
            grid_config (GridConfig): Configuration object containing grid parameters
        """
        self.beam = beam_config
        self.grid = grid_config
        self.setup_grids()
        self.solver = CrankNicolsonSolver(grid_config, beam_config)

        # Initialize simulation results arrays
        self.envelope = np.empty(
            (self.grid.n_radi_nodes, self.grid.dist_steps + 1), dtype=complex
        )
        self.envelope_s = np.empty_like(self.envelope, dtype=complex)

    def calculate_analytical_solution(self):
        """
        Calculate the analytical solution for a Gaussian beam.

        This method implements the analytical solution for a Gaussian beam
        propagation through a lens, including effects such as:
            - Beam waist evolution
            - Radius of curvature
            - Gouy phase shift

        Returns:
            np.ndarray: Complex array containing the analytical solution
        """
        # Initialize empty array for analytical solution
        envelope_s = np.empty_like(self.envelope, dtype=complex)

        # Calculate beam parameters
        rayleigh_len = 0.5 * self.beam.wavenumber * self.beam.waist**2
        lens_dist = self.beam.focal_length / (
            1 + (self.beam.focal_length / rayleigh_len) ** 2
        )

        # Calculate beam evolution
        beam_waist = self.beam.waist * np.sqrt(
            (1 - self.dist_array / self.beam.focal_length) ** 2
            + (self.dist_array / rayleigh_len) ** 2
        )

        beam_radius = (
            self.dist_array
            - lens_dist
            + (lens_dist * (self.beam.focal_length - lens_dist))
            / (self.dist_array - lens_dist)
        )

        gouy_phase = np.arctan(
            (self.dist_array - lens_dist)
            / np.sqrt(self.beam.focal_length * lens_dist - lens_dist**2)
        )

        # Calculate terms for the analytical solution
        ratio_term = self.beam.waist / beam_waist[np.newaxis, :]
        decay_exp_term = (self.radi_array[:, np.newaxis] / beam_waist) ** 2
        prop_exp_term = (
            0.5
            * 1j
            * self.beam.wavenumber
            * self.radi_array[:, np.newaxis] ** 2
            / beam_radius
        )
        gouy_exp_term = 1j * gouy_phase[np.newaxis, :]

        # Compute final solution
        envelope_s = (
            self.beam.amplitude
            * ratio_term
            * np.exp(-decay_exp_term + prop_exp_term - gouy_exp_term)
        )

        return envelope_s

    def run_simulation(self):
        """
        Run the complete beam propagation simulation.

        This method executes the following steps:
        1. Sets up the initial Gaussian beam condition
        2. Propagates the field using Crank-Nicolson method
        3. Calculates the analytical solution for comparison
        """
        # Create initial condition
        initial_field = self.initial_condition(self.radi_array)

        # Propagate the field
        self.envelope = self.solver.propagate(initial_field)

        # Calculate analytical solution
        self.envelope_s = self.calculate_analytical_solution()

    def setup_grids(self):
        """
        Set up the computational grids for the simulation.

        Creates the following grid arrays:
        - 1D radial coordinates array
        - 1D propagation distance array
        - 2D meshgrid for both coordinates
        """
        self.radi_array = np.linspace(
            self.grid.ini_radi, self.grid.fin_radi, self.grid.n_radi_nodes
        )
        self.dist_array = np.linspace(
            self.grid.ini_dist, self.grid.fin_dist, self.grid.dist_steps + 1
        )
        self.radi_2d_array, self.dist_2d_array = np.meshgrid(
            self.radi_array, self.dist_array, indexing="ij"
        )

    def initial_condition(self, r):
        """
        Set the post-lens chirped Gaussian beam initial condition.

        Args:
            r (np.ndarray): Radial coordinates array

        Returns:
            np.ndarray: Complex array containing the initial field distribution
        """
        return self.beam.amplitude * np.exp(
            -((r / self.beam.waist) ** 2)
            - 0.5 * 1j * self.beam.wavenumber * r**2 / self.beam.focal_length
        )
