"""
Crank-Nicolson Solver Module.

This module implements the Crank-Nicolson finite difference method for solving
the paraxial wave equation in cylindrical coordinates. Features include:

* Implicit finite difference scheme implementation
* Sparse matrix operations for efficient computation
* Boundary conditions:
  - Neumann at r=0 (beam axis)
  - Dirichlet at r=R (computational boundary)
* Progressive field propagation with progress tracking
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import diags_array
from scipy.sparse.linalg import spsolve
from tqdm import tqdm


@dataclass
class CrankNicolsonSolver:
    """
    Solver implementing the Crank-Nicolson method for beam propagation.

    This class handles the numerical solution of the paraxial wave equation
    using an implicit finite difference scheme with sparse matrices.

    Attributes:
        grid: Grid parameters configuration
        beam: Beam parameters configuration
        eu_cyl: Coordinate system flag (1 for cylindrical)
        delta_r: Normalized step size parameter
        matrix_cnt: Matrix construction parameter
        left_matrix: Left-hand side CN matrix (sparse)
        right_matrix: Right-hand side CN matrix (sparse)
    """

    def __init__(self, grid_config, beam_config):
        """
        Initialize solver with grid and beam configurations.

        Args:
            grid_config: Configuration object for spatial grid
            beam_config: Configuration object for laser beam
        """
        self.grid = grid_config
        self.beam = beam_config
        self.eu_cyl = 1  # Cylindrical coordinate system
        self._setup_matrices()

    def _setup_matrices(self):
        """
        Initialize matrices needed for calculations.

        Computes:
            * Normalized step size parameter (delta_r)
            * Matrix construction parameter (matrix_cnt)
            * Left and right Crank-Nicolson matrices
        """
        self.delta_r = (
            0.25 * self.grid.dist_step / (self.beam.wavenumber * self.grid.radi_step**2)
        )
        self.matrix_cnt = 1j * self.delta_r

        # Create CN matrices
        self.left_matrix = self._create_cn_matrix("LEFT")
        self.right_matrix = self._create_cn_matrix("RIGHT", negative=True)

    def _create_cn_diagonals(self, n, pos, coef):
        """
        Generate the three diagonals for Crank-Nicolson array.

        Implements the finite difference stencil for the paraxial equation
        in cylindrical coordinates, including boundary conditions.

        Args:
            n: Number of grid points
            pos: Position indicator ('LEFT' or 'RIGHT')
            coef: Coefficient for matrix elements

        Returns:
            tuple: Three arrays (lower, main, upper diagonals)
        """
        mcf = 1.0 + 2.0 * coef
        ind = np.arange(1, n - 1)

        diag_m1 = -coef * (1 - 0.5 * self.eu_cyl / ind)
        diag_0 = np.full(n, mcf)
        diag_p1 = -coef * (1 + 0.5 * self.eu_cyl / ind)

        diag_m1 = np.append(diag_m1, [0.0])
        diag_p1 = np.insert(diag_p1, 0, [0.0])

        # Set boundary conditions
        if self.eu_cyl == 1:
            if pos == "LEFT":
                diag_0[0], diag_0[-1] = mcf, 1.0
                diag_p1[0] = -2.0 * coef
            else:  # RIGHT
                diag_0[0], diag_0[-1] = mcf, 0.0
                diag_p1[0] = -2.0 * coef

        return diag_m1, diag_0, diag_p1

    def _create_cn_matrix(self, position, negative=False):
        """
        Create Crank-Nicolson sparse matrix.

        Args:
            position: Matrix position ('LEFT' or 'RIGHT')
            negative: Whether to negate the coefficient (default: False)

        Returns:
            scipy.sparse.csr_matrix: Sparse matrix in CSR format
        """
        coef = -self.matrix_cnt if negative else self.matrix_cnt
        diags = self._create_cn_diagonals(self.grid.n_radi_nodes, position, coef)
        return diags_array(diags, offsets=[-1, 0, 1], format="csr")

    def propagate(self, initial_field):
        """
        Propagate the field using Crank-Nicolson method.

        Implements step-by-step field propagation using the implicit scheme:
        1. Apply right-hand side matrix to current field
        2. Solve linear system with left-hand side matrix
        3. Store result and continue to next step

        Args:
            initial_field: Initial field distribution (complex array)

        Returns:
            np.ndarray: Propagated field at all distances (2D complex array)
        """
        envelope = np.empty(
            (self.grid.n_radi_nodes, self.grid.dist_steps + 1), dtype=complex
        )
        envelope[:, 0] = initial_field

        b_array = np.empty(self.grid.n_radi_nodes, dtype=complex)

        for k in tqdm(range(self.grid.dist_steps)):
            b_array = self.right_matrix @ envelope[:, k]
            envelope[:, k + 1] = spsolve(self.left_matrix, b_array)

        return envelope
