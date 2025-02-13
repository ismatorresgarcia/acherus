"""
Python script for plotting NumPy arrays saved during the simulations.
This script uses the matplotlib library to plot the results.
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class PlotConfig:
    """Configuration for plot styling"""

    style: str = "dark_background"
    cmap: str = "plasma"
    figsize: Tuple[int, int] = (13, 7)
    colors: Dict[str, str] = None

    def __post_init__(self):
        if self.colors is None:
            self.colors = {
                "green": "#32CD32",  # Lime green
                "blue": "#1E90FF",  # Electric Blue
                "yellow": "#FFFF00",  # Pure yellow
            }
        self.cmap = mpl.colormaps[self.cmap]


class PhysicalConstants:
    """Physical constants used in calculations"""

    ELEC_PERMITTIVITY_0: float = 8.8541878128e-12
    LIGHT_SPEED_0: float = 299792458

    # Conversion factors
    INT_FACTOR: float = 1
    RADI_FACTOR: float = 1e6
    DIST_FACTOR: float = 100
    TIME_FACTOR: float = 1e15
    AREA_FACTOR: float = 1e-4


class ComputationalDomain:
    """Handles computational domain setup and calculations"""

    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.setup_domain_limits()
        self.calculate_nodes()
        self.setup_arrays()

    def setup_domain_limits(self):
        """Set up the computational domain limits"""
        self.radi_limits = (self.data["INI_RADI_COOR"], self.data["FIN_RADI_COOR"] / 20)
        self.dist_limits = (self.data["INI_DIST_COOR"], self.data["FIN_DIST_COOR"])
        self.time_limits = (-100e-15, 100e-15)

    def calculate_nodes(self):
        """Calculate node positions"""
        # Calculate dimensions
        self.n_radi = self.data["e"].shape[0]
        self.n_dist = self.data["e_axis"].shape[0]
        self.n_time = self.data["e_axis"].shape[1]

        # Calculate float positions
        self.nodes = {}
        for dim, (start, end, n_nodes, ini, fin) in {
            "radi": (
                *self.radi_limits,
                self.n_radi,
                self.data["INI_RADI_COOR"],
                self.data["FIN_RADI_COOR"],
            ),
            "dist": (
                *self.dist_limits,
                self.n_dist,
                self.data["INI_DIST_COOR"],
                self.data["FIN_DIST_COOR"],
            ),
            "time": (
                *self.time_limits,
                self.n_time,
                self.data["INI_TIME_COOR"],
                self.data["FIN_TIME_COOR"],
            ),
        }.items():
            start_val = (start - ini) * (n_nodes - 1) / (fin - ini)
            end_val = (end - ini) * (n_nodes - 1) / (fin - ini)
            self.nodes[dim] = (int(start_val), int(end_val) + 1)

        self.peak_node = -int(
            self.time_limits[0]
            * (self.n_time - 1)
            // (self.data["FIN_TIME_COOR"] - self.data["INI_TIME_COOR"])
        )

    def setup_arrays(self):
        """Set up computational arrays"""
        # Create slice objects
        self.slices = {
            "r": slice(*self.nodes["radi"]),
            "z": slice(*self.nodes["dist"]),
            "t": slice(*self.nodes["time"]),
        }

        # Create base arrays
        self.arrays = {
            "radi": np.linspace(
                self.data["INI_RADI_COOR"], self.data["FIN_RADI_COOR"], self.n_radi
            )[self.slices["r"]],
            "dist": np.linspace(
                self.data["INI_DIST_COOR"], self.data["FIN_DIST_COOR"], self.n_dist
            )[self.slices["z"]],
            "time": np.linspace(
                self.data["INI_TIME_COOR"], self.data["FIN_TIME_COOR"], self.n_time
            )[self.slices["t"]],
        }

        # Create 2D meshgrids
        self.arrays["dist_2d_1"], self.arrays["time_2d_1"] = np.meshgrid(
            self.arrays["dist"], self.arrays["time"], indexing="ij"
        )
        self.arrays["radi_2d_2"], self.arrays["time_2d_2"] = np.meshgrid(
            self.arrays["radi"], self.arrays["time"], indexing="ij"
        )


class Plotter:
    """Handles all plotting functionality"""

    def __init__(
        self,
        domain: ComputationalDomain,
        config: PlotConfig,
        constants: PhysicalConstants,
    ):
        self.domain = domain
        self.config = config
        self.constants = constants
        plt.style.use(self.config.style)
        self.setup_scaled_arrays()

    def calculate_intensities(self, envelope, envelope_axis):
        """Calculate intensities for plotting"""
        return (
            self.constants.AREA_FACTOR
            * self.constants.INT_FACTOR
            * np.abs(envelope) ** 2,
            self.constants.AREA_FACTOR
            * self.constants.INT_FACTOR
            * np.abs(envelope_axis) ** 2,
        )

    def setup_scaled_arrays(self):
        """Set up scaled arrays for plotting"""
        self.scaled_arrays = {
            "time": self.constants.TIME_FACTOR * self.domain.arrays["time"],
            "dist": self.constants.DIST_FACTOR * self.domain.arrays["dist"],
            "dist_2d_1": self.constants.DIST_FACTOR * self.domain.arrays["dist_2d_1"],
            "time_2d_1": self.constants.TIME_FACTOR * self.domain.arrays["time_2d_1"],
            "radi_2d_2": self.constants.RADI_FACTOR * self.domain.arrays["radi_2d_2"],
            "time_2d_2": self.constants.TIME_FACTOR * self.domain.arrays["time_2d_2"],
        }

    def plot_temporal_evolution(self, plot_int_axis):
        """Create temporal evolution plot"""
        fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=self.config.figsize)

        # First subplot
        ax1.plot(
            self.scaled_arrays["time"],
            plot_int_axis[0, :],
            color=self.config.colors["green"],
            linestyle="--",
            label=r"On-axis numerical solution at beginning $z$ step",
        )
        ax1.plot(
            self.scaled_arrays["time"],
            plot_int_axis[-1, :],
            color=self.config.colors["blue"],
            linestyle="-",
            label=r"On-axis numerical solution at final $z$ step",
        )
        ax1.set(xlabel=r"$t$ ($\mathrm{fs}$)", ylabel=r"$I(t)$ ($\mathrm{W/{cm}^2}$)")
        ax1.legend(facecolor="black", edgecolor="white")
        # Second subplot
        ax2.plot(
            self.scaled_arrays["dist"],
            plot_int_axis[:, self.domain.peak_node],
            color=self.config.colors["yellow"],
            linestyle="-",
            label="On-axis peak time numerical solution",
        )
        ax2.set(xlabel=r"$z$ ($\mathrm{cm}$)", ylabel=r"$I(z)$ ($\mathrm{W/{cm}^2}$)")
        ax2.legend(facecolor="black", edgecolor="white")

        fig1.tight_layout()
        plt.show()

    def plot_2d_solutions(self, plot_int_axis, plot_int_fin):
        """Create 2D solution plots"""
        fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=self.config.figsize)

        # First subplot
        fig2_1 = ax3.pcolormesh(
            self.scaled_arrays["dist_2d_1"],
            self.scaled_arrays["time_2d_1"],
            plot_int_axis,
            cmap=self.config.cmap,
        )
        fig2.colorbar(fig2_1, ax=ax3)
        ax3.set(xlabel=r"$z$ ($\mathrm{cm}$)", ylabel=r"$t$ ($\mathrm{fs}$)")
        ax3.set_title("On-axis solution in 2D")
        # Second subplot
        fig2_2 = ax4.pcolormesh(
            self.scaled_arrays["radi_2d_2"],
            self.scaled_arrays["time_2d_2"],
            plot_int_fin,
            cmap=self.config.cmap,
        )
        fig2.colorbar(fig2_2, ax=ax4)
        ax4.set(xlabel=r"$r$ ($\mathrm{\mu m}$)", ylabel=r"$t$ ($\mathrm{fs}$)")
        ax4.set_title(r"Final step solution in 2D")

        fig2.tight_layout()
        plt.show()

    def plot_3d_solutions(self, plot_int_axis, plot_int_fin):
        """Create 3D solution plots"""
        _, (ax5, ax6) = plt.subplots(
            1, 2, figsize=self.config.figsize, subplot_kw={"projection": "3d"}
        )

        # First subplot
        ax5.plot_surface(
            self.scaled_arrays["dist_2d_1"],
            self.scaled_arrays["time_2d_1"],
            plot_int_axis,
            cmap=self.config.cmap,
            linewidth=0,
            antialiased=False,
        )
        ax5.set(
            xlabel=r"$z$ ($\mathrm{cm}$)",
            ylabel=r"$t$ ($\mathrm{fs}$)",
            zlabel=r"$I(z,t)$ ($\mathrm{W/{cm}^2}$)",
        )
        ax5.set_title("On-axis solution in 3D")

        # Second subplot
        ax6.plot_surface(
            self.scaled_arrays["radi_2d_2"],
            self.scaled_arrays["time_2d_2"],
            plot_int_fin,
            cmap=self.config.cmap,
            linewidth=0,
            antialiased=False,
        )
        ax6.set(
            xlabel=r"$r$ ($\mathrm{\mu m}$)",
            ylabel=r"$t$ ($\mathrm{fs}$)",
            zlabel=r"$I(r,t)$ ($\mathrm{W/{cm}^2}$)",
        )
        ax6.set_title(r"Final step solution in 3D")

        # fig3.tight_layout()
        plt.show()


def main():
    """Main execution function"""
    # Load data
    data = np.load(
        "/Users/ytoga/projects/phd_thesis/phd_coding/python/storage/pruebilla_scn_2.npz"
    )

    # Initialize classes
    config = PlotConfig()
    constants = PhysicalConstants()
    domain = ComputationalDomain(data)
    plotter = Plotter(domain, config, constants)

    # Calculate intensities
    plot_int_fin, plot_int_axis = plotter.calculate_intensities(
        data["e"][domain.slices["r"], domain.slices["t"]],
        data["e_axis"][domain.slices["z"], domain.slices["t"]],
    )

    # Create plots
    plotter.plot_temporal_evolution(plot_int_axis)
    plotter.plot_2d_solutions(plot_int_axis, plot_int_fin)
    plotter.plot_3d_solutions(plot_int_axis, plot_int_fin)


if __name__ == "__main__":
    main()
