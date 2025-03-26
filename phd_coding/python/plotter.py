"""
Python script for plotting NumPy arrays saved during the simulations.
The script uses the matplotlib library to plot the results with optimized memory usage.
"""

__version__ = "1.0.0"

import argparse
import os
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class PlotConfiguration:
    """Plot style configuration."""

    style: str = "dark_background"
    font: bool = False
    figsize: Tuple[int, int] = (13, 7)
    colors: Dict[str, str] = field(
        default_factory=lambda: {
            "blue": "#1E90FF",  # Electric Blue
            "green": "#32CD32",  # Lime green
            "magenta": "#FF00FF",  # Magenta
            "yellow": "#FFFF00",  # Pure yellow
        }
    )
    cmaps: Dict[str, Any] = field(
        default_factory=lambda: {
            "density": mpl.colormaps["viridis"],
            "intensity": mpl.colormaps["plasma"],
        }
    )

    def get_plot_config(self, plot_type: str, dimension: str = "all") -> Dict:
        """
        Return configuration for specified plot type and dimension.

        Args:
        - plot_type: Type of data ("intensity" or "density")
        - dimension: Plot dimension ("1d", "2d", "3d", or "all" for complete config)

        Returns:
        - Dictionary with plot configuration settings
        """
        # Base configuration shared across all plot types
        if self.font:
            plt.rcParams.update(
                {
                    "font.family": "serif",
                    "font.serif": ["Times New Roman"],
                    "font.size": 10,
                    "axes.titlesize": 12,
                    "axes.labelsize": 10,
                }
            )
        base_config = {
            "intensity": {
                "cmap": self.cmaps["intensity"],
                "zlabel": r"$I$ ($\mathrm{W/cm^2}$)",
                "colors": {
                    "init": self.colors["green"],
                    "final": self.colors["blue"],
                    "peak": self.colors["yellow"],
                },
                "titles": {
                    "zt": r"On-axis intensity $I(z,t)$",
                    "rt": r"Intensity $I(r,t)$ at $z = {:.2f}$ $\mathrm{cm}$",
                    "rz": r"Maximum intensity $\max\,I(r,z)$",
                },
                "labels": {
                    "ylabel_t": r"$I(t)$ ($\mathrm{W/cm^2}$)",
                    "ylabel_z": r"$\max\,I(z)$ ($\mathrm{W/cm^2}$)",
                    "xlabel_r": r"$r$ ($\mathrm{\mu m}$)",
                    "xlabel_z": r"$z$ ($\mathrm{cm}$)",
                    "xlabel_t": r"$t$ ($\mathrm{fs}$)",
                },
                "legend_labels": {
                    "axis_init": r"On-axis solution at beginning $z$ step",
                    "axis_final": r"On-axis solution at final $z$ step",
                    "axis_max": r"On-axis maximum",
                    "fixed_z": r"Fixed $z$ evolution",
                    "on_axis": r"On-axis evolution",
                    "max_evolution": r"Maximum evolution",
                },
            },
            "density": {
                "cmap": self.cmaps["density"],
                "zlabel": r"$\rho$ ($\mathrm{cm^{-3}}$)",
                "colors": {
                    "init": self.colors["green"],
                    "final": self.colors["blue"],
                    "peak": self.colors["magenta"],
                },
                "titles": {
                    "zt": r"On-axis density $\rho(z,t)$",
                    "rt": r"Density $\rho(r,t)$ at $z = {:.2f}$ $\mathrm{cm}$",
                    "rz": r"Maximum density $\max\,\rho(r,z)$",
                },
                "labels": {
                    "ylabel_t": r"$\rho(t)$ ($\mathrm{cm^{-3}}$)",
                    "ylabel_z": r"$max\,\rho(z)$ ($\mathrm{cm^{-3}}$)",
                    "xlabel_r": r"$r$ ($\mathrm{\mu m}$)",
                    "xlabel_z": r"$z$ ($\mathrm{cm}$)",
                    "xlabel_t": r"$t$ ($\mathrm{fs}$)",
                },
                "legend_labels": {
                    "axis_init": r"On-axis solution at beginning $z$ step",
                    "axis_final": r"On-axis solution at final $z$ step",
                    "axis_max": r"On-axis maximum",
                    "fixed_z": r"Fixed $z$ evolution",
                    "on_axis": r"On-axis evolution",
                    "max_evolution": r"Maximum evolution",
                },
            },
        }

        # Dimension-specific configuration
        dimension_config = {
            "1d": {
                "figsize": self.figsize,
                "legend_settings": {"facecolor": "black", "edgecolor": "white"},
                "dpi": 150,
            },
            "2d": {"figsize": self.figsize, "dpi": 150},
            "3d": {
                "resolutions": {
                    "low": {"stride": (5, 5), "dpi": 100, "antialiased": False},
                    "medium": {"stride": (2, 2), "dpi": 150, "antialiased": True},
                    "high": {"stride": (1, 1), "dpi": 300, "antialiased": True},
                },
                "legend_settings": {
                    "facecolor": "black",
                    "edgecolor": "white",
                    "loc": "upper right",
                },
                "figsize": self.figsize,
            },
        }

        # Return requested configuration
        if dimension == "all":
            return {
                "base": base_config[plot_type],
                **dimension_config,
            }

        return {**base_config[plot_type], **dimension_config.get(dimension, {})}


@dataclass
class Constants:
    "Physical and mathematical constants."

    permittivity: float = 8.8541878128e-12
    light_speed: float = 299792458

    intensity_units: float = 1
    factor_radial: float = 1e6
    factor_distance: float = 100
    factor_time: float = 1e15
    factor_area: float = 1e-4
    factor_volume: float = 1e-6


class PlotGrid:
    """Plotting grid box-sizing."""

    def __init__(self, const: Constants, data: Dict[str, Any]):
        self.const = const
        self.data = data
        self._initialize_boundaries()
        self._initialize_grid_nodes()
        self._initialize_sliced_arrays()

    def _initialize_boundaries(self):
        """Set up the plotting box boundary."""
        self.boundary_radial = (self.data["ini_radi_coor"], self.data["fin_radi_coor"])
        self.boundary_distance = (
            self.data["ini_dist_coor"],
            self.data["fin_dist_coor"],
        )
        self.boundary_time = (self.data["ini_time_coor"], self.data["fin_time_coor"])

    def _initialize_grid_nodes(self):
        """Set up the plotting box boundary nodes."""
        self.nodes_radial = self.data["e_dist"].shape[0]
        self.nodes_distance = self.data["e_axis"].shape[0]
        self.nodes_time = self.data["e_axis"].shape[1]

        self.nodes = {}
        for dim, (start, end, n_nodes, ini, fin) in {
            "radi": (
                *self.boundary_radial,
                self.nodes_radial,
                self.data["ini_radi_coor"],
                self.data["fin_radi_coor"],
            ),
            "dist": (
                *self.boundary_distance,
                self.nodes_distance,
                self.data["ini_dist_coor"],
                self.data["fin_dist_coor"],
            ),
            "time": (
                *self.boundary_time,
                self.nodes_time,
                self.data["ini_time_coor"],
                self.data["fin_time_coor"],
            ),
        }.items():
            start_val = (start - ini) * (n_nodes - 1) / (fin - ini)
            end_val = (end - ini) * (n_nodes - 1) / (fin - ini)
            self.nodes[dim] = (int(start_val), int(end_val) + 1)

        self.axis_node = self.data["axis_node"]
        self.peak_node = self.data["peak_node"]

    def calculate_z_coordinate(self, indices):
        """Convert k-indices to their corresponding z-coordinates with caching."""
        # Calculate the z coordinate position
        ini_dist_coor = self.data["ini_dist_coor"] * self.const.factor_distance
        fin_dist_coor = self.data["fin_dist_coor"] * self.const.factor_distance
        z_coor = ini_dist_coor + (
            indices * (fin_dist_coor - ini_dist_coor) / (self.nodes_distance - 1)
        )
        return z_coor

    def _initialize_sliced_arrays(self):
        """Set up computational arrays"""
        self.slices = {
            "r": slice(*self.nodes["radi"]),
            "z": slice(*self.nodes["dist"]),
            "t": slice(*self.nodes["time"]),
        }

        # Create 1D sliced grids
        self.sliced_grids = {
            "radi": np.linspace(
                self.data["ini_radi_coor"],
                self.data["fin_radi_coor"],
                self.nodes_radial,
            )[self.slices["r"]],
            "dist": np.linspace(
                self.data["ini_dist_coor"],
                self.data["fin_dist_coor"],
                self.nodes_distance,
            )[self.slices["z"]],
            "time": np.linspace(
                self.data["ini_time_coor"], self.data["fin_time_coor"], self.nodes_time
            )[self.slices["t"]],
        }

        # Create 2D sliced grids
        self.sliced_grids["dist_2d_1"], self.sliced_grids["time_2d_1"] = np.meshgrid(
            self.sliced_grids["dist"], self.sliced_grids["time"], indexing="ij"
        )
        self.sliced_grids["radi_2d_2"], self.sliced_grids["time_2d_2"] = np.meshgrid(
            self.sliced_grids["radi"], self.sliced_grids["time"], indexing="ij"
        )
        self.sliced_grids["radi_2d_3"], self.sliced_grids["dist_2d_3"] = np.meshgrid(
            self.sliced_grids["radi"], self.sliced_grids["dist"], indexing="ij"
        )


class ScaledGrid:
    """Plotting grid box with custom units."""

    def __init__(self, const: Constants, box: PlotGrid, config: PlotConfiguration):
        self.const = const
        self.box = box
        self.config = config
        plt.style.use(self.config.style)
        self._scaled_1d_grid = {}
        self._scaled_2d_grid = {}

    def create_scaled_1d_grid(self, name):
        """Get a scaled array, creating it if necessary."""
        if name not in self._scaled_1d_grid:
            if name.startswith("radi"):
                self._scaled_1d_grid[name] = (
                    self.const.factor_radial * self.box.sliced_grids[name]
                )
            elif name.startswith("dist"):
                self._scaled_1d_grid[name] = (
                    self.const.factor_distance * self.box.sliced_grids[name]
                )
            elif name.startswith("time"):
                self._scaled_1d_grid[name] = (
                    self.const.factor_time * self.box.sliced_grids[name]
                )

        return self._scaled_1d_grid[name]

    def create_scaled_2d_grid(self, grid_type):
        """Set up meshgrids only when needed."""
        if grid_type not in self._scaled_2d_grid:
            if grid_type == "dist_time":
                self._scaled_2d_grid[grid_type] = np.meshgrid(
                    self.create_scaled_1d_grid("dist"),
                    self.create_scaled_1d_grid("time"),
                    indexing="ij",
                )
            elif grid_type == "radi_time":
                self._scaled_2d_grid[grid_type] = np.meshgrid(
                    self.create_scaled_1d_grid("radi"),
                    self.create_scaled_1d_grid("time"),
                    indexing="ij",
                )
            elif grid_type == "radi_dist":
                self._scaled_2d_grid[grid_type] = np.meshgrid(
                    self.create_scaled_1d_grid("radi"),
                    self.create_scaled_1d_grid("dist"),
                    indexing="ij",
                )

        return self._scaled_2d_grid[grid_type]


class BasePlotter:
    """Plotting class for all simulation data."""

    def __init__(
        self,
        const: Constants,
        box: PlotGrid,
        config: PlotConfiguration,
        box_scaled: ScaledGrid,
    ):
        self.const = const
        self.box = box
        self.config = config
        self.box_scaled = box_scaled
        plt.style.use(self.config.style)

    def calculate_intensity_values(self, envelope_dist, envelope_axis, envelope_peak):
        """Set up intensities for plotting."""
        return (
            self.const.factor_area
            * self.const.intensity_units
            * np.abs(envelope_dist) ** 2,
            self.const.factor_area
            * self.const.intensity_units
            * np.abs(envelope_axis) ** 2,
            self.const.factor_area
            * self.const.intensity_units
            * np.abs(envelope_peak) ** 2,
        )

    def calculate_density_values(self, density_dist, density_axis, density_peak):
        """Set up densities for plotting."""
        return (
            self.const.factor_volume * density_dist,
            self.const.factor_volume * density_axis,
            self.const.factor_volume * density_peak,
        )

    def save_or_display(self, fig, filename, save_path, dpi=150):
        """Save figure or display it."""
        if save_path:
            save_path = Path(save_path)
            filepath = save_path / filename
            fig.tight_layout()
            fig.savefig(filepath, dpi=dpi)
            plt.close(fig)
        else:
            fig.tight_layout()
            plt.show()

    def get_1d_grid(self, name):
        """Access box-scaled 1d-grids."""
        return self.box_scaled.create_scaled_1d_grid(name)

    def get_2d_grid(self, grid_type):
        """Access box-scaled 2d-grid."""
        return self.box_scaled.create_scaled_2d_grid(grid_type)


class Plotter1D(BasePlotter):
    """Plotting class for 1D solutions."""

    def create_plot(self, data_axis, data_peak, plot_type="intensity", save_path=None):
        """
        Create 1D solution plots for intensity or density.

        Arguments:
            data_axis: Array containing the on-axis data to plot.
            data_peak: Array containing the peak data to plot.
            plot_type: "intensity" or "density".
            save_path: Path to save figures instead of displaying them.
        """
        plot_config = self.config.get_plot_config(plot_type, "1d")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.config.figsize)

        # Get scaled arrays only when needed
        time_array = self.get_1d_grid("time")
        dist_array = self.get_1d_grid("dist")

        # First subplot - temporal evolution
        ax1.plot(
            time_array,
            data_axis[0, :],
            color=plot_config["colors"]["init"],
            linestyle="--",
            label=plot_config["legend_labels"]["axis_init"],
        )
        ax1.plot(
            time_array,
            data_axis[-1, :],
            color=plot_config["colors"]["final"],
            linestyle="-",
            label=plot_config["legend_labels"]["axis_final"],
        )
        ax1.set(
            xlabel=plot_config["labels"]["xlabel_t"],
            ylabel=plot_config["labels"]["ylabel_t"],
        )
        ax1.legend(**plot_config["legend_settings"])

        # Second subplot - spatial on_axis evolution
        ax2.plot(
            dist_array,
            data_peak[self.box.axis_node, :],
            color=plot_config["colors"]["peak"],
            linestyle="-",
            label=plot_config["legend_labels"]["axis_max"],
        )
        ax2.set(
            xlabel=plot_config["labels"]["xlabel_z"],
            ylabel=plot_config["labels"]["ylabel_z"],
        )
        ax2.legend(**plot_config["legend_settings"])

        self.save_or_display(fig, f"1d_{plot_type}.png", save_path, plot_config["dpi"])


class Plotter2D(BasePlotter):
    """Plotting class for 2D solutions."""

    def create_plot(
        self, data, k_array=None, z_coor=None, plot_type="intensity", save_path=None
    ):
        """
        Create 2D solution plots for different coordinate systems.

        Arguments:
            data: Dictionary containing the datasets for different coordinates.
            k_array: List of z indices to plot (for rt plots).
            z_coor: List of z coordinates corresponding to the k indices saved.
            plot_type: "intensity" or "density".
            save_path: Path to save figures instead of displaying them.
        """
        # Configuration for different plot types
        plot_config = self.config.get_plot_config(plot_type, "2d")

        # Plot each coordinate system in a separate figure
        for coord_sys, data1 in data.items():
            if coord_sys == "rt" and k_array is not None:
                # Plot for each z node
                for idx in range(len(k_array)):
                    fig, ax = plt.subplots(figsize=plot_config["figsize"])

                    # Lazily get the meshgrid
                    x, y = self.get_2d_grid("radi_time")
                    xlabel = plot_config["labels"]["xlabel_r"]
                    ylabel = plot_config["labels"]["xlabel_t"]

                    mesh = ax.pcolormesh(
                        x,
                        y,
                        data1[:, idx, :],
                        cmap=plot_config["cmap"],
                    )
                    fig.colorbar(mesh, ax=ax, label=plot_config["zlabel"])
                    ax.set(xlabel=xlabel, ylabel=ylabel)

                    # Get actual z-position in cm
                    z_pos = z_coor[idx]
                    z_pos_format = f"{z_pos:.2f}"
                    title = plot_config["titles"][coord_sys].replace(
                        "{:.2f}", z_pos_format
                    )
                    ax.set_title(title)

                    if save_path:
                        save_path = Path(save_path)
                        filename = f"2d_{plot_type}_{coord_sys}_{z_pos:.2f}.png"
                        filepath = save_path / filename
                        fig.tight_layout()
                        fig.savefig(filepath, dpi=plot_config["dpi"])
                        plt.close(fig)
                    else:
                        fig.tight_layout()
                        plt.show()
            else:
                # Plots for zt and rz
                fig, ax = plt.subplots(figsize=self.config.figsize)

                if coord_sys == "zt":
                    x, y = self.get_2d_grid("dist_time")
                    xlabel = plot_config["labels"]["xlabel_z"]
                    ylabel = plot_config["labels"]["xlabel_t"]
                else:
                    x, y = self.get_2d_grid("radi_dist")
                    xlabel = plot_config["labels"]["xlabel_r"]
                    ylabel = plot_config["labels"]["xlabel_z"]

                mesh = ax.pcolormesh(x, y, data1, cmap=plot_config["cmap"])
                fig.colorbar(mesh, ax=ax, label=plot_config["zlabel"])
                ax.set(xlabel=xlabel, ylabel=ylabel)
                ax.set_title(plot_config["titles"][coord_sys])

                if save_path:
                    save_path = Path(save_path)
                    filename = f"2d_{plot_type}_{coord_sys}.png"
                    filepath = save_path / filename
                    fig.tight_layout()
                    fig.savefig(filepath, dpi=plot_config["dpi"])
                    plt.close(fig)
                else:
                    fig.tight_layout()
                    plt.show()


class Plotter3D(BasePlotter):
    """Plotting class for 3D solutions."""

    def create_plot(
        self,
        data,
        k_array,
        z_coor=None,
        plot_type="intensity",
        resolution="medium",
        save_path=None,
    ):
        """
        Create 3D solution plots for different coordinate systems.

        Arguments:
            data: Dictionary containing the datasets for different coordinates.
            k_array: List of z indices to plot (for rt plots).
            z_coor: List of z coordinates corresponding to the k indices saved.
            plot_type: "intensity" or "density".
            stride: Tuple specifying the stride for mesh plotting (faster rendering).
            resolution: Plot quality (low, medium, high).
            save_path: Path to save figures instead of displaying them.
        """
        # Set configuration
        plot_config = self.config.get_plot_config(plot_type, "3d")
        resolution_config = plot_config["resolutions"].get(
            resolution, plot_config["resolutions"]["medium"]
        )
        stride = resolution_config["stride"]

        # Disable interactive mode if saving
        if save_path and plt.isinteractive():
            plt.ioff()

        for coord_sys, data1 in data.items():
            if coord_sys == "rt" and k_array is not None:
                # Plot for each z node
                for idx in range(len(k_array)):
                    fig = plt.figure(
                        figsize=plot_config["figsize"], dpi=resolution_config["dpi"]
                    )
                    ax = fig.add_subplot(projection="3d")

                    # Get meshgrid lazily
                    x, y = self.get_2d_grid("radi_time")
                    xlabel = plot_config["labels"]["xlabel_r"]
                    ylabel = plot_config["labels"]["xlabel_t"]
                    label = plot_config["legend_labels"]["fixed_z"]

                    surf = ax.plot_surface(
                        x[:: stride[0], :: stride[1]],
                        y[:: stride[0], :: stride[1]],
                        data1[:: stride[0], idx, :: stride[1]],
                        cmap=plot_config["cmap"],
                        linewidth=0,
                        antialiased=resolution_config["antialiased"],
                        label=label,
                    )
                    fig.colorbar(surf, label=plot_config["zlabel"])
                    ax.set(
                        xlabel=xlabel,
                        ylabel=ylabel,
                        zlabel=plot_config["zlabel"],
                    )
                    # Get actual z position in cm
                    z_pos = z_coor[idx]
                    z_pos_format = f"{z_pos:.2f}"
                    title = plot_config["titles"][coord_sys].replace(
                        "{:.2f}", z_pos_format
                    )
                    ax.set_title(title)
                    ax.legend(**plot_config["legend_settings"])

                    if save_path:
                        filename = f"3d_{plot_type}_{coord_sys}_{z_pos:.2f}.png"
                        filepath = os.path.join(save_path, filename)
                        fig.tight_layout()
                        fig.savefig(filepath, dpi=resolution_config["dpi"])
                        plt.close(fig)
                    else:
                        fig.tight_layout()
                        plt.show()

            else:
                # Plots for zt and rz
                fig = plt.figure(
                    figsize=plot_config["figsize"], dpi=resolution_config["dpi"]
                )
                ax = fig.add_subplot(projection="3d")

                if coord_sys == "zt":
                    # Get meshgrid lazily
                    x, y = self.get_2d_grid("dist_time")
                    xlabel = plot_config["labels"]["xlabel_z"]
                    ylabel = plot_config["labels"]["xlabel_t"]
                    label = plot_config["legend_labels"]["on_axis"]
                else:
                    # Get meshgrid lazily
                    x, y = self.get_2d_grid("radi_dist")
                    xlabel = plot_config["labels"]["xlabel_r"]
                    ylabel = plot_config["labels"]["xlabel_z"]
                    label = plot_config["legend_labels"]["max_evolution"]

                # Apply stride for better performance
                surf = ax.plot_surface(
                    x[:: stride[0], :: stride[1]],
                    y[:: stride[0], :: stride[1]],
                    data1[:: stride[0], :: stride[1]],
                    cmap=plot_config["cmap"],
                    linewidth=0,
                    antialiased=resolution_config["antialiased"],
                    label=label,
                )
                fig.colorbar(surf, label=plot_config["zlabel"])
                ax.set(
                    xlabel=xlabel,
                    ylabel=ylabel,
                    zlabel=plot_config["zlabel"],
                )
                ax.set_title(plot_config["titles"][coord_sys])
                ax.legend(**plot_config["legend_settings"])

                if save_path:
                    filename = f"3d_{plot_type}_{coord_sys}.png"
                    filepath = os.path.join(save_path, filename)
                    fig.tight_layout()
                    fig.savefig(filepath, dpi=resolution_config["dpi"])
                    plt.close(fig)
                else:
                    fig.tight_layout()
                    plt.show()


class VisualManager:
    """Manages all plotting classes."""

    def __init__(self, const, box, config, box_scaled):
        self.const = const
        self.box = box
        self.config = config
        self.box_scaled = box_scaled

        # Initialize specialized plotters
        self.base_plotter = BasePlotter(const, box, config, box_scaled)
        self.plot_1d = Plotter1D(const, box, config, box_scaled)
        self.plot_2d = Plotter2D(const, box, config, box_scaled)
        self.plot_3d = Plotter3D(const, box, config, box_scaled)

    def calculate_intensities(self, envelope_dist, envelope_axis, envelope_peak):
        """Calculate intensity data."""
        return self.base_plotter.calculate_intensity_values(
            envelope_dist, envelope_axis, envelope_peak
        )

    def calculate_densities(self, density_dist, density_axis, density_peak):
        """Calculate density data."""
        return self.base_plotter.calculate_density_values(
            density_dist, density_axis, density_peak
        )

    def create_1d_plot(
        self, data_axis, data_peak, plot_type="intensity", save_path=None
    ):
        """Create 1D solution plots."""
        self.plot_1d.create_plot(data_axis, data_peak, plot_type, save_path)

    def create_2d_plot(
        self, data, k_array=None, z_coor=None, plot_type="intensity", save_path=None
    ):
        """Create 2D solution plots."""
        self.plot_2d.create_plot(data, k_array, z_coor, plot_type, save_path)

    def create_3d_plot(
        self,
        data,
        k_array,
        z_coor=None,
        plot_type="intensity",
        resolution="medium",
        save_path=None,
    ):
        """Create 3D solution plots."""
        self.plot_3d.create_plot(
            data, k_array, z_coor, plot_type, resolution, save_path
        )


def parse_cli_options():
    """Parse and validate CLI options."""
    parser = argparse.ArgumentParser(
        description="Plot simulation data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase output verbosity (can be used multiple times)",
    )
    parser.add_argument(
        "--file",
        default="./storage/data.npz",
        help="Path to data file (.npz format)",
    )
    parser.add_argument(
        "--save-path",
        default="./storage/figures_/",
        help="Directory to save plots instead of displaying",
    )
    parser.add_argument(
        "--data-types",
        default="intensity,density",
        help="Data types to plot: intensity,density (comma-separated)",
    )
    parser.add_argument(
        "--plot-types",
        default="1d,2d,3d",
        help="Plot types to generate: 1d,2d,3d (comma-separated)",
    )
    parser.add_argument(
        "--resolution",
        default="medium",
        help="Plot quality for 3D plots: low, medium, high",
    )

    args = parser.parse_args()

    # Convert comma-separated strings to dictionaries for easier access
    args.plot_types = {ptype: True for ptype in args.plot_types.split(",")}
    args.data_types = {dtype: True for dtype in args.data_types.split(",")}

    return args


def setup_output_directory(args):
    """Setup environment for plotting based on arguments."""
    # Ensure we have a save path if needed
    if args.save_path:
        save_path = Path(args.save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        print(f"Saving plots to: {args.save_path}")


def load_simulation_data(file_path, args):
    """Load data with memory optimization based on file size."""
    print(f"Loading data from {file_path}...")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    try:
        file_size = os.path.getsize(file_path) / (1024**2)

        # Process data based on file size
        if file_size > 500:  # If file is larger than 500 MB
            print("Using memory-mapped file loading for large file")
            with np.load(file_path, mmap_mode="r") as npz:
                data = {
                    key: np.array(npz[key])
                    for key in npz.files
                    if key
                    in [
                        "ini_radi_coor",
                        "fin_radi_coor",
                        "ini_dist_coor",
                        "fin_dist_coor",
                        "ini_time_coor",
                        "fin_time_coor",
                        "axis_node",
                        "peak_node",
                        "k_array",
                    ]
                }
                # Reference large arrays without loading fully
                for key in [
                    "e_dist",
                    "e_axis",
                    "e_peak",
                    "elec_dist",
                    "elec_axis",
                    "elec_peak",
                ]:
                    if key in npz.files:
                        data[key] = npz[key]  # Memory-mapped reference
        else:
            print("Loading full data file")
            data = dict(np.load(file_path))

        return data

    except FileNotFoundError as e:
        print(f"Error: Data file not found: {e}")
        raise
    except PermissionError as e:
        print(f"Error: No permission to read file: {e}")
        raise
    except ValueError as e:
        print(f"Error: Invalid NPZ file format: {e}")
        print("The data file may be corrupted or have an incorrect format")
        raise
    except MemoryError:
        print("Error: Not enough memory to load data file")
        print("Try using a machine with more RAM or reduce the file size")
        raise
    except OSError as e:
        print(f"I/O error when accessing file: {e}")
        raise
    except BaseException as e:
        print(f"Unexpected error loading data: {type(e).__name__}: {e}")
        if args.verbose > 0:
            traceback.print_exc()
        else:
            print("Run with -v for more information")
        raise


def create_plot(data_type, plot_data, plot_types, plot, k_array, z_coor, args):
    """Generate requested plot types for the specified data."""

    if plot_types.get("1d", False):
        print(f"Generating 1D {data_type} plots...")
        plot.create_1d_plot(
            plot_data["zt"],
            plot_data["rz"],
            plot_type=data_type,
            save_path=args.save_path,
        )

    if plot_types.get("2d", False):
        print(f"Generating 2D {data_type} plots...")
        plot.create_2d_plot(plot_data, k_array, z_coor, data_type, args.save_path)

    if plot_types.get("3d", False):
        print(f"Generating 3D {data_type} plots...")
        try:
            plot.create_3d_plot(
                plot_data,
                k_array,
                z_coor,
                data_type,
                resolution=args.resolution,
                save_path=args.save_path,
            )
        except MemoryError:
            print(
                "Warning: Not enough memory for 3D plots. Try using a lower resolution"
            )
            raise
        except ValueError as e:
            print(f"Warning: Value error in 3D plot for: {e}")
            print(
                "This may indicate incompatible array dimensions or invalid parameters"
            )
            raise
        except RuntimeError as e:
            print(f"Warning: Runtime error in 3D plot for: {e}")
            print("This may be caused by a Matplotlib rendering issue or invalid data")
            raise
        except KeyError as e:
            print(f"Missing key in 3D plot configuration for: {e}")
            print("Check that your plot configuration contains all required settings")
            raise
        except (TypeError, AttributeError) as e:
            print(f"Python error in 3D plots: {type(e).__name__}: {e}")
            print("This may indicate incompatible data types or invalid method calls")
            raise
        except OSError as e:
            print(f"I/O error in 3D plots: {e}")
            print("Check disk space and write permissions in the save directory")
            raise
        except mpl.MatplotlibError as e:
            print(f"Matplotlib error: {e}")
            print("This may indicate an issue with the plotting configuration")
            raise
        except BaseException as e:
            print(f"Unexpected error in 3D plot: {type(e).__name__}: {e}")
            if args.verbose > 0:
                traceback.print_exc()
            else:
                print("Run with -v for more information")
            raise


def process_simulation_data(data_type, data, plot, box, plot_types, args):
    """Process a specific data type (intensity or density) and generate plots."""
    print(f"Processing {data_type} data...")

    # Confirm required arrays exist
    required_arrays = []
    if data_type == "intensity":
        required_arrays = ["e_dist", "e_axis", "e_peak"]
    else:
        required_arrays = ["elec_dist", "elec_axis", "elec_peak"]

    for key in required_arrays:
        if key in data:
            expected_shape = None
            if key.endswith("_dist"):
                expected_shape = (
                    box.nodes_radial,
                    len(data["k_array"]),
                    box.nodes_time,
                )
            elif key.endswith("_axis"):
                expected_shape = (box.nodes_distance, box.nodes_time)
            elif key.endswith("_peak"):
                expected_shape = (box.nodes_radial, box.nodes_distance)

            if expected_shape and data[key].shape != expected_shape:
                print(
                    f"Warning: {key} has unexpected shape: {data[key].shape}, expected {expected_shape}"
                )

    missing = [arr for arr in required_arrays if arr not in data]
    if missing:
        print(f"Warning: Missing required data for {data_type}: {', '.join(missing)}")
        print(f"Omitting {data_type} plots")
        return

    # Get propagation indices
    k_array = np.array(data["k_array"])
    z_coor = np.array([box.calculate_z_coordinate(k) for k in k_array])

    # For simplicity
    slices = box.slices

    # Calculate data based on type
    if data_type == "intensity":
        plot_data_dist, plot_data_axis, plot_data_peak = plot.calculate_intensities(
            data["e_dist"][slices["r"], :, slices["t"]],
            data["e_axis"][slices["z"], slices["t"]],
            data["e_peak"][slices["r"], slices["z"]],
        )
    elif data_type == "density":
        plot_data_dist, plot_data_axis, plot_data_peak = plot.calculate_densities(
            data["elec_dist"][slices["r"], :, slices["t"]],
            data["elec_axis"][slices["z"], slices["t"]],
            data["elec_peak"][slices["r"], slices["z"]],
        )
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

    # Prepare data dictionary for plotting functions
    plot_data = {
        "rt": plot_data_dist,
        "zt": plot_data_axis,
        "rz": plot_data_peak,
    }

    # Generate requested plot types
    create_plot(data_type, plot_data, plot_types, plot, k_array, z_coor, args)


def main():
    """Main execution function."""
    print(f"Simulation plotter v{__version__}")

    # Parse CLI arguments
    args = parse_cli_options()

    # Setup environment (paths, etc.)
    setup_output_directory(args)

    # Load data
    data = load_simulation_data(args.file, args)

    if args.verbose > 0:
        print("Data file contents:")
        print("   Data available:", ", ".join(data.keys()))
        print(
            f"  Data dimensions: {data["e_axis"].shape if "e_axis" in data else "unknown"}"
        )

    # Initialize classes
    const = Constants()
    config = PlotConfiguration()
    box = PlotGrid(const, data)
    box_scaled = ScaledGrid(const, box, config)
    plot = VisualManager(const, box, config, box_scaled)

    # Process each requested data type
    for data_type, enabled in args.data_types.items():
        if enabled and data_type in ["intensity", "density"]:
            process_simulation_data(data_type, data, plot, box, args.plot_types, args)

    print("Plotting complete!")


if __name__ == "__main__":
    main()
