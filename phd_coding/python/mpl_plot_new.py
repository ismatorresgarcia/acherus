"""
Python script for plotting NumPy arrays saved during the simulations.
The script uses the matplotlib library to plot the results.
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class VisualizationConfig:
    """Configuration for plot styling."""

    style: str = "dark_background"
    figsize: Tuple[int, int] = (13, 7)
    colors: Dict[str, str] = None
    cmaps: Dict[str, str] = None

    def __post_init__(self):
        if self.colors is None:
            self.colors = {
                "green": "#32CD32",  # Lime green
                "blue": "#1E90FF",  # Electric Blue
                "yellow": "#FFFF00",  # Pure yellow
                "magenta": "#FF00FF",  # Magenta
            }
        if self.cmaps is None:
            self.cmaps = {
                "intensity": mpl.colormaps["plasma"],
                "density": mpl.colormaps["viridis"],
            }


class UniversalConstants:
    """Universal constants and conversion factors."""

    permittivity_0: float = 8.8541878128e-12
    light_speed_0: float = 299792458

    # Conversion factors
    int_factor: float = 1
    radi_factor: float = 1e6
    dist_factor: float = 100
    time_factor: float = 1e15
    area_factor: float = 1e-4
    vol_factor: float = 1e-6


class DomainParameters:
    """Handles computational domain setup and calculations."""

    def __init__(self, constants: UniversalConstants, data: Dict[str, Any]):
        self.constants = constants
        self.data = data
        self.setup_domain_limits()
        self.compute_nodes()
        self.setup_arrays()

    def setup_domain_limits(self):
        """Set up the computational domain limits"""
        self.radi_limits = (self.data["ini_radi_coor"], self.data["fin_radi_coor"] / 20)
        self.dist_limits = (self.data["ini_dist_coor"], self.data["fin_dist_coor"])
        self.time_limits = (-100e-15, 100e-15)

    def compute_nodes(self):
        """Calculate node positions"""
        # Calculate dimensions
        self.n_radi = self.data["e_dist"].shape[0]
        self.n_dist = self.data["e_axis"].shape[0]
        self.n_time = self.data["e_axis"].shape[1]

        # Calculate node positions
        self.nodes = {}
        for dim, (start, end, n_nodes, ini, fin) in {
            "radi": (
                *self.radi_limits,
                self.n_radi,
                self.data["ini_radi_coor"],
                self.data["fin_radi_coor"],
            ),
            "dist": (
                *self.dist_limits,
                self.n_dist,
                self.data["ini_dist_coor"],
                self.data["fin_dist_coor"],
            ),
            "time": (
                *self.time_limits,
                self.n_time,
                self.data["ini_time_coor"],
                self.data["fin_time_coor"],
            ),
        }.items():
            start_val = (start - ini) * (n_nodes - 1) / (fin - ini)
            end_val = (end - ini) * (n_nodes - 1) / (fin - ini)
            self.nodes[dim] = (int(start_val), int(end_val) + 1)

        self.axis_node = self.data["axis_node"]
        self.peak_node = self.data["peak_node"]

    def compute_z_coor(self, indices):
        """Convert k-indices to their corresponding z-coordinates."""

        # Calculate the z coordinate position
        ini_dist_coor = self.data["ini_dist_coor"] * self.constants.dist_factor
        fin_dist_coor = self.data["fin_dist_coor"] * self.constants.dist_factor
        z_coor = ini_dist_coor + (
            indices * (fin_dist_coor - ini_dist_coor) / (self.n_dist - 1)
        )
        return z_coor

    def setup_arrays(self):
        """Set up computational arrays"""
        # Create slice objects
        self.slices = {
            "r": slice(*self.nodes["radi"]),
            "z": slice(*self.nodes["dist"]),
            "t": slice(*self.nodes["time"]),
        }

        # Create sliced arrays
        self.arrays = {
            "radi": np.linspace(
                self.data["ini_radi_coor"], self.data["fin_radi_coor"], self.n_radi
            )[self.slices["r"]],
            "dist": np.linspace(
                self.data["ini_dist_coor"], self.data["fin_dist_coor"], self.n_dist
            )[self.slices["z"]],
            "time": np.linspace(
                self.data["ini_time_coor"], self.data["fin_time_coor"], self.n_time
            )[self.slices["t"]],
        }

        # Create 2D meshgrids
        self.arrays["dist_2d_1"], self.arrays["time_2d_1"] = np.meshgrid(
            self.arrays["dist"], self.arrays["time"], indexing="ij"
        )
        self.arrays["radi_2d_2"], self.arrays["time_2d_2"] = np.meshgrid(
            self.arrays["radi"], self.arrays["time"], indexing="ij"
        )
        self.arrays["radi_2d_3"], self.arrays["dist_2d_3"] = np.meshgrid(
            self.arrays["radi"], self.arrays["dist"], indexing="ij"
        )


class Visualization:
    """Handles all plotting functionality."""

    def __init__(
        self,
        constants: UniversalConstants,
        domain: DomainParameters,
        config: VisualizationConfig,
    ):
        self.constants = constants
        self.domain = domain
        self.config = config
        plt.style.use(self.config.style)
        self.setup_scaled_arrays()

    def calculate_intensities(self, envelope_dist, envelope_axis, envelope_peak):
        """Calculate intensities for plotting."""
        return (
            self.constants.area_factor
            * self.constants.int_factor
            * np.abs(envelope_dist) ** 2,
            self.constants.area_factor
            * self.constants.int_factor
            * np.abs(envelope_axis) ** 2,
            self.constants.area_factor
            * self.constants.int_factor
            * np.abs(envelope_peak) ** 2,
        )

    def calculate_densities(self, density_dist, density_axis, density_peak):
        """Calculate densities for plotting."""
        return (
            self.constants.vol_factor * density_dist,
            self.constants.vol_factor * density_axis,
            self.constants.vol_factor * density_peak,
        )

    def setup_scaled_arrays(self):
        """Set up scaled arrays for plotting."""
        self.scaled_arrays = {
            "radi": self.constants.radi_factor * self.domain.arrays["radi"],
            "dist": self.constants.dist_factor * self.domain.arrays["dist"],
            "time": self.constants.time_factor * self.domain.arrays["time"],
            "dist_2d_1": self.constants.dist_factor * self.domain.arrays["dist_2d_1"],
            "time_2d_1": self.constants.time_factor * self.domain.arrays["time_2d_1"],
            "radi_2d_2": self.constants.radi_factor * self.domain.arrays["radi_2d_2"],
            "time_2d_2": self.constants.time_factor * self.domain.arrays["time_2d_2"],
            "radi_2d_3": self.constants.radi_factor * self.domain.arrays["radi_2d_3"],
            "dist_2d_3": self.constants.dist_factor * self.domain.arrays["dist_2d_3"],
        }

    def plot_1d_solutions(self, data_axis, data_peak, plot_type="intensity"):
        """
        Create 1D solution plots for intensity or density.

        Arguments:
            data: Array containing the data to plot.
            plot_type: "intensity" or "density".
        """
        plot_config = {
            "intensity": {
                "colors": {"init": "green", "final": "blue", "peak": "yellow"},
                "labels": {
                    "ylabel_t": r"$I(t)$ ($\mathrm{W/{cm}^2}$)",
                    "ylabel_z": r"$I(z)$ ($\mathrm{W/{cm}^2}$)",
                },
            },
            "density": {
                "colors": {"init": "green", "final": "blue", "peak": "magenta"},
                "labels": {
                    "ylabel_t": r"$\rho(t)$ ($\mathrm{{cm}^{-3}}$)",
                    "ylabel_z": r"$\rho(z)$ ($\mathrm{{cm}^{-3}}$)",
                },
            },
        }

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.config.figsize)

        # First subplot - temporal evolution
        ax1.plot(
            self.scaled_arrays["time"],
            data_axis[0, :],
            color=self.config.colors[plot_config[plot_type]["colors"]["init"]],
            linestyle="--",
            label=r"On-axis solution at beginning $z$ step",
        )
        ax1.plot(
            self.scaled_arrays["time"],
            data_axis[-1, :],
            color=self.config.colors[plot_config[plot_type]["colors"]["final"]],
            linestyle="-",
            label=r"On-axis solution at final $z$ step",
        )
        ax1.set(
            xlabel=r"$t$ ($\mathrm{fs}$)",
            ylabel=plot_config[plot_type]["labels"]["ylabel_t"],
        )
        ax1.legend(facecolor="black", edgecolor="white")

        # Second subplot - spatial on_axis evolution
        ax2.plot(
            self.scaled_arrays["dist"],
            data_peak[self.domain.axis_node, :],
            color=self.config.colors[plot_config[plot_type]["colors"]["peak"]],
            linestyle="-",
            label="On-axis peak time solution",
        )
        ax2.set(
            xlabel=r"$z$ ($\mathrm{cm}$)",
            ylabel=plot_config[plot_type]["labels"]["ylabel_z"],
        )
        ax2.legend(facecolor="black", edgecolor="white")

        fig.tight_layout()
        plt.show()

    def plot_2d_solutions(self, data, k_array=None, z_coor=None, plot_type="intensity"):
        """
        Create 2D solution plots for different coordinate systems.

        Arguments:
            data: Dictionary containing the datasets for different coordinates.
            k_array: List of z indices to plot (for rt plots).
            plot_type: "intensity" or "density".
        """
        # Configuration for different plot types
        plot_config = {
            "intensity": {
                "cmap": self.config.cmaps["intensity"],
                "zlabel": r"$I$ ($\mathrm{W/{cm}^2}$)",
                "titles": {
                    "zt": r"On-axis intensity $I(z,t)$",
                    "rt": r"Intensity $I(r,t)$ at $z = {:.2f}\,cm$",
                    "rz": r"Peak time intensity $I(r,z)$",
                },
            },
            "density": {
                "cmap": self.config.cmaps["density"],
                "zlabel": r"$\rho$ ($\mathrm{cm}^{-3}$)",
                "titles": {
                    "zt": r"On-axis density $\rho(z,t)$",
                    "rt": r"Density $\rho(r,t)$ at $z = {:.2f}\,cm$",
                    "rz": r"Peak time density $\rho(r,z)$",
                },
            },
        }

        # Plot each coordinate system in a separate figure
        for coord_sys, data1 in data.items():
            if coord_sys == "rt" and k_array is not None:
                # Plot for each z node
                for idx in range(len(k_array)):
                    fig, ax = plt.subplots(figsize=self.config.figsize)
                    x, y = (
                        self.scaled_arrays["radi_2d_2"],
                        self.scaled_arrays["time_2d_2"],
                    )
                    xlabel, ylabel = r"$r$ ($\mathrm{\mu m}$)", r"$t$ ($\mathrm{fs}$)"

                    mesh = ax.pcolormesh(
                        x,
                        y,
                        data1[:, idx, :],
                        cmap=plot_config[plot_type]["cmap"],
                    )
                    fig.colorbar(mesh, ax=ax, label=plot_config[plot_type]["zlabel"])
                    ax.set(xlabel=xlabel, ylabel=ylabel)
                    # Get actual z-position in cm
                    z_pos = z_coor[idx]
                    ax.set_title(
                        plot_config[plot_type]["titles"][coord_sys].format(z_pos)
                    )
                    fig.tight_layout()
                    plt.show()
            else:
                # Plots for zt and rz
                fig, ax = plt.subplots(figsize=self.config.figsize)

                if coord_sys == "zt":
                    x, y = (
                        self.scaled_arrays["dist_2d_1"],
                        self.scaled_arrays["time_2d_1"],
                    )
                    xlabel = r"$z$ ($\mathrm{cm}$)"
                    ylabel = r"$t$ ($\mathrm{fs}$)"
                else:
                    x, y = (
                        self.scaled_arrays["radi_2d_3"],
                        self.scaled_arrays["dist_2d_3"],
                    )
                    xlabel = r"$r$ ($\mathrm{\mu m}$)"
                    ylabel = r"$z$ ($\mathrm{cm}$)"

                mesh = ax.pcolormesh(x, y, data1, cmap=plot_config[plot_type]["cmap"])
                fig.colorbar(mesh, ax=ax, label=plot_config[plot_type]["zlabel"])
                ax.set(xlabel=xlabel, ylabel=ylabel)
                ax.set_title(plot_config[plot_type]["titles"][coord_sys])

                fig.tight_layout()
                plt.show()

    def plot_3d_solutions(self, data, k_array, z_coor=None, plot_type="intensity"):
        """
        Create 3D solution plots for different coordinate systems.

        Arguments:
            data: Dictionary containing the datasets for different coordinates.
            k_array: List of z indices to plot (for rt plots).
            plot_type: "intensity" or "density".
        """
        plot_config = {
            "intensity": {
                "cmap": self.config.cmaps["intensity"],
                "zlabel": r"$I$ ($\mathrm{W/{cm}^2}$)",
                "titles": {
                    "zt": r"On-axis intensity $I(z,t)$",
                    "rt": r"Intensity $I(r,t)$ at $z = {:.2f}\,cm$",
                    "rz": r"Peak time intensity $I(r,z)$",
                },
            },
            "density": {
                "cmap": self.config.cmaps["density"],
                "zlabel": r"$\rho$ ($\mathrm{cm}^{-3}$)",
                "titles": {
                    "zt": r"On-axis density $\rho(z,t)$",
                    "rt": r"Density $\rho(r,t)$ at $z = {:.2f}\,cm$",
                    "rz": r"Peak time density $\rho(r,z)$",
                },
            },
        }

        for coord_sys, data1 in data.items():
            if coord_sys == "rt" and k_array is not None:
                # Plot for each z node
                for idx in range(len(k_array)):
                    fig = plt.figure(figsize=self.config.figsize)
                    ax = fig.add_subplot(projection="3d")
                    x, y = (
                        self.scaled_arrays["radi_2d_2"],
                        self.scaled_arrays["time_2d_2"],
                    )
                    xlabel, ylabel = r"$r$ ($\mathrm{\mu m}$)", r"$t$ ($\mathrm{fs}$)"
                    label = r"Fixed $z$ evolution"

                    surf = ax.plot_surface(
                        x,
                        y,
                        data1[:, idx, :],
                        cmap=plot_config[plot_type]["cmap"],
                        linewidth=0,
                        antialiased=False,
                        label=label,
                    )
                    fig.colorbar(surf, label=plot_config[plot_type]["zlabel"])
                    ax.set(
                        xlabel=xlabel,
                        ylabel=ylabel,
                        zlabel=plot_config[plot_type]["zlabel"],
                    )
                    # Get actual z position in cm
                    z_pos = z_coor[idx]
                    ax.set_title(
                        plot_config[plot_type]["titles"][coord_sys].format(z_pos)
                    )
                    ax.legend(facecolor="black", edgecolor="white", loc="upper right")
                    fig.tight_layout()
                    plt.show()
            else:
                # Plots for zt and rz
                fig = plt.figure(figsize=self.config.figsize)
                ax = fig.add_subplot(projection="3d")

                if coord_sys == "zt":
                    x, y = (
                        self.scaled_arrays["dist_2d_1"],
                        self.scaled_arrays["time_2d_1"],
                    )
                    xlabel = r"$z$ ($\mathrm{cm}$)"
                    ylabel = r"$t$ ($\mathrm{fs}$)"
                    label = "On-axis evolution"
                else:
                    x, y = (
                        self.scaled_arrays["radi_2d_3"],
                        self.scaled_arrays["dist_2d_3"],
                    )
                    xlabel = r"$r$ ($\mathrm{\mu m}$)"
                    ylabel = r"$z$ ($\mathrm{cm}$)"
                    label = "Peak time evolution"

                surf = ax.plot_surface(
                    x,
                    y,
                    data1,
                    cmap=plot_config[plot_type]["cmap"],
                    linewidth=0,
                    antialiased=False,
                    label=label,
                )
                fig.colorbar(surf, label=plot_config[plot_type]["zlabel"])
                ax.set(
                    xlabel=xlabel,
                    ylabel=ylabel,
                    zlabel=plot_config[plot_type]["zlabel"],
                )
                ax.set_title(plot_config[plot_type]["titles"][coord_sys])
                ax.legend(facecolor="black", edgecolor="white", loc="upper right")

                fig.tight_layout()
                plt.show()


def main():
    """Main execution function."""
    # Load data
    data = np.load(
        "/Users/ytoga/projects/phd_thesis/phd_coding/python/storage/ffdmk_fcn_1.npz"
    )

    # Initialize classes
    constants = UniversalConstants()
    config = VisualizationConfig()
    domain = DomainParameters(constants, data)
    plotter = Visualization(constants, domain, config)

    # Define selected propagation indices for plotting
    k_array = data["k_array"]

    # Convert k-indices to their corresponding z-coordinates
    z_coor = [domain.compute_z_coor(k) for k in k_array]

    # Calculate intensities
    plot_int_dist, plot_int_axis, plot_int_peak = plotter.calculate_intensities(
        data["e_dist"][domain.slices["r"], :, domain.slices["t"]],
        data["e_axis"][domain.slices["z"], domain.slices["t"]],
        data["e_peak"][domain.slices["r"], domain.slices["z"]],
    )
    # plot_dens_dist, plot_dens_axis, plot_dens_peak = plotter.calculate_densities(
    #    data["elec_dist"][domain.slices["r"], :, domain.slices["t"]],
    #    data["elec_axis"][domain.slices["z"], domain.slices["t"]],
    #    data["elec_peak"][domain.slices["r"], domain.slices["z"]],
    # )

    # Prepare data dictionaries for different coordinate systems
    intensity_data = {
        "rt": plot_int_dist,
        "zt": plot_int_axis,
        "rz": plot_int_peak,
    }

    # density_data = {
    #    "rt": plot_dens_dist,
    #    "zt": plot_dens_axis,
    #    "rz": plot_dens_peak,
    # }

    # Create 1D plots
    plotter.plot_1d_solutions(plot_int_axis, plot_int_peak, "intensity")
    # plotter.plot_1d_solutions(plot_dens_axis, plot_dens_peak, "density")

    # Create 2D plots
    plotter.plot_2d_solutions(intensity_data, k_array, z_coor, "intensity")
    # plotter.plot_2d_solutions(density_data, k_array, z_coor, "density")

    # Create 3D plots
    plotter.plot_3d_solutions(intensity_data, k_array, z_coor, "intensity")
    # plotter.plot_3d_solutions(density_data, k_array, z_coor, "density")


if __name__ == "__main__":
    main()
