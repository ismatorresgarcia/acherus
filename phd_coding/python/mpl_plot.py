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
    VOL_FACTOR: float = 1e-6


class ComputationalDomain:
    """Handles computational domain setup and calculations"""

    def __init__(self, data: Dict[str, Any], constants: PhysicalConstants):
        self.data = data
        self.constants = constants
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
        self.n_radi = self.data["e_dist"].shape[0]
        self.n_dist = self.data["e_axis"].shape[0]
        self.n_time = self.data["e_axis"].shape[1]

        # Calculate node positions
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

    def calculate_z_coor(self, k_index):
        """Convert saved k-indices to corresponding z-coordinates"""

        # Calculate the z coordinate position
        ini_dist_coor = self.data["INI_DIST_COOR"] * self.constants.DIST_FACTOR
        fin_dist_coor = self.data["FIN_DIST_COOR"] * self.constants.DIST_FACTOR
        z_coor = ini_dist_coor + (
            k_index * (fin_dist_coor - ini_dist_coor) / (self.n_dist - 1)
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
        self.arrays["radi_2d_3"], self.arrays["dist_2d_3"] = np.meshgrid(
            self.arrays["radi"], self.arrays["dist"], indexing="ij"
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

    def calculate_intensities(self, envelope_dist, envelope_axis, envelope_peak):
        """Calculate intensities for plotting"""
        return (
            self.constants.AREA_FACTOR
            * self.constants.INT_FACTOR
            * np.abs(envelope_dist) ** 2,
            self.constants.AREA_FACTOR
            * self.constants.INT_FACTOR
            * np.abs(envelope_axis) ** 2,
            self.constants.AREA_FACTOR
            * self.constants.INT_FACTOR
            * np.abs(envelope_peak) ** 2,
        )

    def calculate_densities(self, density_dist, density_axis, density_peak):
        """Calculate densities for plotting"""
        return (
            self.constants.VOL_FACTOR
            * self.constants.INT_FACTOR
            * np.abs(density_dist) ** 2,
            self.constants.VOL_FACTOR
            * self.constants.INT_FACTOR
            * np.abs(density_axis) ** 2,
            self.constants.VOL_FACTOR
            * self.constants.INT_FACTOR
            * np.abs(density_peak) ** 2,
        )

    def setup_scaled_arrays(self):
        """Set up scaled arrays for plotting"""
        self.scaled_arrays = {
            "radi": self.constants.RADI_FACTOR * self.domain.arrays["radi"],
            "dist": self.constants.DIST_FACTOR * self.domain.arrays["dist"],
            "time": self.constants.TIME_FACTOR * self.domain.arrays["time"],
            "dist_2d_1": self.constants.DIST_FACTOR * self.domain.arrays["dist_2d_1"],
            "time_2d_1": self.constants.TIME_FACTOR * self.domain.arrays["time_2d_1"],
            "radi_2d_2": self.constants.RADI_FACTOR * self.domain.arrays["radi_2d_2"],
            "time_2d_2": self.constants.TIME_FACTOR * self.domain.arrays["time_2d_2"],
            "radi_2d_3": self.constants.RADI_FACTOR * self.domain.arrays["radi_2d_3"],
            "dist_2d_3": self.constants.DIST_FACTOR * self.domain.arrays["dist_2d_3"],
        }

    def plot_1d_solutions(self, data, plot_type="intensity"):
        """Create 1D solution plots for intensity or density

        Args:
            data: Array containing the data to plot
            plot_type: "intensity" or "density"
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
            data[0, :],
            color=self.config.colors[plot_config[plot_type]["colors"]["init"]],
            linestyle="--",
            label=r"On-axis numerical solution at beginning $z$ step",
        )
        ax1.plot(
            self.scaled_arrays["time"],
            data[-1, :],
            color=self.config.colors[plot_config[plot_type]["colors"]["final"]],
            linestyle="-",
            label=r"On-axis numerical solution at final $z$ step",
        )
        ax1.set(
            xlabel=r"$t$ ($\mathrm{fs}$)",
            ylabel=plot_config[plot_type]["labels"]["ylabel_t"],
        )
        ax1.legend(facecolor="black", edgecolor="white")

        # Second subplot - spatial on_axis evolution
        ax2.plot(
            self.scaled_arrays["dist"],
            data[:, self.domain.peak_node],
            color=self.config.colors[plot_config[plot_type]["colors"]["peak"]],
            linestyle="-",
            label="On-axis peak time numerical solution",
        )
        ax2.set(
            xlabel=r"$z$ ($\mathrm{cm}$)",
            ylabel=plot_config[plot_type]["labels"]["ylabel_z"],
        )
        ax2.legend(facecolor="black", edgecolor="white")

        fig.tight_layout()
        plt.show()

    def plot_2d_solutions(self, data, z_nodes=None, z_coor=None, plot_type="intensity"):
        """Create 2D solution plots for different coordinate systems

        Args:
            data: Dictionary containing the datasets for different coordinates
            z_nodes: List of z indices to plot (for rt plots)
            plot_type: "intensity" or "density"
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
            if coord_sys == "rt" and z_nodes is not None:
                # Plot for each z node
                for idx in range(len(z_nodes)):
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

    def plot_3d_solutions(self, data, z_nodes, z_coor=None, plot_type="intensity"):
        """Create 3D solution plots for different coordinate systems

        Args:
            data: Dictionary containing the datasets for different coordinates
            z_nodes: List of z indices to plot (for rt plots)
            plot_type: "intensity" or "density"
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
            if coord_sys == "rt" and z_nodes is not None:
                # Plot for each z node
                for idx in range(len(z_nodes)):
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
    """Main execution function"""
    # Load data
    data = np.load(
        "/Users/ytoga/projects/phd_thesis/phd_coding/python/storage/pruebilla_full_1.npz"
    )

    # Initialize classes
    config = PlotConfig()
    constants = PhysicalConstants()
    domain = ComputationalDomain(data, constants)
    plotter = Plotter(domain, config, constants)

    # Define selected propagation indices for plotting
    z_nodes = data["z_nodes"]

    # Convert k-indices to the corresponding z-coordinates
    z_coor = [domain.calculate_z_coor(k) for k in z_nodes]

    # Calculate intensities
    plot_int_dist, plot_int_axis, plot_int_peak = plotter.calculate_intensities(
        data["e_dist"][domain.slices["r"], :, domain.slices["t"]],
        data["e_axis"][domain.slices["z"], domain.slices["t"]],
        data["e_peak"][domain.slices["r"], domain.slices["z"]],
    )
    plot_dens_dist, plot_dens_axis, plot_dens_peak = plotter.calculate_densities(
        data["elec_dist"][domain.slices["r"], :, domain.slices["t"]],
        data["elec_axis"][domain.slices["z"], domain.slices["t"]],
        data["elec_peak"][domain.slices["r"], domain.slices["z"]],
    )

    # Prepare data dictionaries for different coordinate systems
    intensity_data = {
        "rt": plot_int_dist,
        "zt": plot_int_axis,
        "rz": plot_int_peak,
    }

    density_data = {
        "rt": plot_dens_dist,
        "zt": plot_dens_axis,
        "rz": plot_dens_peak,
    }

    # Create 1D plots
    plotter.plot_1d_solutions(plot_int_axis, "intensity")
    plotter.plot_1d_solutions(plot_dens_axis, "density")

    # Create 2D plots
    plotter.plot_2d_solutions(intensity_data, z_nodes, z_coor, "intensity")
    plotter.plot_2d_solutions(density_data, z_nodes, z_coor, "density")

    # Create 3D plots
    plotter.plot_3d_solutions(intensity_data, z_nodes, z_coor, "intensity")
    plotter.plot_3d_solutions(density_data, z_nodes, z_coor, "density")


if __name__ == "__main__":
    main()
