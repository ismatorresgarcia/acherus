"""
Beam Plotting Module.

This module handles the visualization of beam propagation results, including:
    - Intensity profile plots
    - Comparison between numerical and analytical solutions
    - Custom styling and unit conversions
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


class BeamPlotter:
    """
    Class for creating beam propagation visualization plots.

    This class handles the creation of publication-quality plots
    for comparing numerical and analytical beam propagation results.

    Attributes:
        sim (GaussianBeamSimulation): Simulation object with results
        radi_factor (float): Conversion factor to mm for radial coordinates
        dist_factor (float): Conversion factor to cm for propagation distance
        area_factor (float): Conversion factor to cm² for beam area
        cmap (matplotlib.colors.Colormap): Color map for intensity plots
        figsize (tuple): Figure size in inches (width, height)
    """

    def __init__(self, simulation):
        """
        Initialize plotter with simulation results.

        Args:
            simulation (GaussianBeamSimulation): Simulation object containing
                grid parameters and results to be plotted
        """
        self.sim = simulation
        self.setup_plot_parameters()

    def setup_plot_parameters(self):
        """
        Set up plotting parameters and style.

        Configures:
            - Unit conversion factors
            - Dark background style
            - Color map selection
            - Figure size
        """
        self.radi_factor = 1000  # mm
        self.dist_factor = 100  # cm
        self.area_factor = 1e-4  # cm²
        plt.style.use("dark_background")
        self.cmap = mpl.colormaps["plasma"]
        self.figsize = (13, 7)

    def plot_intensity_profiles(self, envelope, envelope_s):
        """
        Create intensity profile comparison plots.

        Generates a figure with three subplots:
            1. Numerical solution intensity profile
            2. Analytical solution intensity profile
            3. Difference between solutions

        Args:
            envelope (np.ndarray): Numerical solution field
            envelope_s (np.ndarray): Analytical solution field
        """
        # Set up plotting grid (mm, cm)
        new_radi_2d_array = self.radi_factor * self.sim.radi_2d_array
        new_dist_2d_array = self.dist_factor * self.sim.dist_2d_array
        new_radi_array = new_radi_2d_array[:, 0]
        new_dist_array = new_dist_2d_array[0, :]

        # Set up intensities (W/cm^2)
        plot_intensity = (
            self.area_factor
            * self.sim.beam.media["WATER"]["INT_FACTOR"]
            * np.abs(envelope) ** 2
        )
        plot_intensity_s = (
            self.area_factor
            * self.sim.beam.media["WATER"]["INT_FACTOR"]
            * np.abs(envelope_s) ** 2
        )

        ## Set up figure 1
        fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize)
        # Subplot 1
        intensity_list = [
            (
                plot_intensity_s[:, 0],
                "#FF00FF",
                "-",
                r"Analytical solution at beginning $z$ step",
            ),
            (
                plot_intensity_s[:, -1],
                "#FFFF00",
                "-",
                r"Analytical solution at final $z$ step",
            ),
            (
                plot_intensity[:, 0],
                "#32CD32",
                "--",
                r"Numerical solution at beginning $z$ step",
            ),
            (
                plot_intensity[:, -1],
                "#1E90FF",
                "--",
                r"Numerical solution at final $z$ step",
            ),
        ]

        for data, color, style, label in intensity_list:
            ax1.plot(
                new_radi_array, data, color, linestyle=style, linewidth=2, label=label
            )
        ax1.set(xlabel=r"$r$ ($\mathrm{mm}$)", ylabel=r"$I(r)$ ($\mathrm{W/{cm}^2}$)")
        ax1.legend(facecolor="black", edgecolor="white")

        # Subplot 2
        ax2.plot(
            new_dist_array,
            plot_intensity_s[self.sim.grid.axis_node, :],
            "#FF00FF",
            linestyle="-",
            linewidth=2,
            label="On-axis analytical solution",
        )
        ax2.plot(
            new_dist_array,
            plot_intensity[self.sim.grid.axis_node, :],
            "#32CD32",
            linestyle="--",
            linewidth=2,
            label="On-axis numerical solution",
        )
        ax2.set(xlabel=r"$z$ ($\mathrm{cm}$)", ylabel=r"$I(z)$ ($\mathrm{W/{cm}^2}$)")
        ax2.legend(facecolor="black", edgecolor="white")

        fig1.tight_layout()
        plt.show()

        ## Set up figure 2
        fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=self.figsize)
        # Subplot 1
        fig2_1 = ax3.pcolormesh(
            new_radi_2d_array, new_dist_2d_array, plot_intensity, cmap=self.cmap
        )
        fig2.colorbar(fig2_1, ax=ax3)
        ax3.set(xlabel=r"$r$ ($\mathrm{mm}$)", ylabel=r"$z$ ($\mathrm{cm}$)")
        ax3.set_title("Numerical solution in 2D")

        # Subplot 2
        fig2_2 = ax4.pcolormesh(
            new_radi_2d_array, new_dist_2d_array, plot_intensity_s, cmap=self.cmap
        )
        fig2.colorbar(fig2_2, ax=ax4)
        ax4.set(xlabel=r"$r$ ($\mathrm{mm}$)", ylabel=r"$z$ ($\mathrm{cm}$)")
        ax4.set_title("Analytical solution in 2D")

        fig2.tight_layout()
        plt.show()

        ## Set up figure 3
        fig3, (ax5, ax6) = plt.subplots(
            1, 2, figsize=self.figsize, subplot_kw={"projection": "3d"}
        )
        # Subplot 1
        ax5.plot_surface(
            new_radi_2d_array,
            new_dist_2d_array,
            plot_intensity,
            cmap=self.cmap,
            linewidth=0,
            antialiased=False,
        )
        ax5.set(
            xlabel=r"$r$ ($\mathrm{mm}$)",
            ylabel=r"$z$ ($\mathrm{cm}$)",
            zlabel=r"$I(r,z)$ ($\mathrm{W/{cm}^2}$)",
        )
        ax5.set_title("Numerical solution")

        # Subplot 2
        ax6.plot_surface(
            new_radi_2d_array,
            new_dist_2d_array,
            plot_intensity_s,
            cmap=self.cmap,
            linewidth=0,
            antialiased=False,
        )
        ax6.set(
            xlabel=r"$r$ ($\mathrm{mm}$)",
            ylabel=r"$z$ ($\mathrm{cm}$)",
            zlabel=r"$I(r,z)$ ($\mathrm{W/{cm}^2}$)",
        )
        ax6.set_title("Analytical solution")

        fig3.tight_layout()
        plt.show()
