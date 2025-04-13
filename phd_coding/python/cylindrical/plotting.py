"""
Python script for plotting NumPy arrays saved during the simulations.
The script uses the matplotlib library to plot the results with optimized memory usage.
"""

__version__ = "0.1.2"

import argparse
import os
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

DEFAULT_DATA_FILE_PATH = "./storage/data.npz"
DEFAULT_SAVE_PATH = "./storage/figures_/"


@dataclass
class PlotConfiguration:
    """Plot style configuration."""

    line_colors: Dict[str, str] = field(
        default_factory=lambda: {
            "blue_dark": "#1E90FF",  # Electric Blue
            "green_dark": "#32CD32",  # Lime green
            "magenta_dark": "#FF00FF",  # Strong magenta
            "yellow_dark": "#FFFF00",  # Pure yellow
            "blue_white": "#0066CC",  # Dark Blue
            "green_white": "#007F00",  # Olive green
            "magenta_white": "#CC00CC",  # Shiny magenta
            "yellow_white": "#CC9900",  # Brownish yellow
        }
    )
    colormaps: Dict[str, Any] = field(
        default_factory=lambda: {
            "density": mpl.colormaps["viridis"],
            "intensity": mpl.colormaps["plasma"],
            "intensity_log": mpl.colormaps["turbo"],
        }
    )

    def __post_init__(self):
        """Customize figure styling."""
        plt.style.use("default")

        plt.rcParams.update(
            {
                # Figure size
                "figure.figsize": (13, 7),
                # Grid options
                "axes.grid": False,
                # Axis options
                "axes.linewidth": 0.8,
                # Line options
                "lines.linewidth": 1.5,
                # Font configuration
                "text.usetex": True,
                "font.family": "serif",
                # Font options
                "font.size": 10,
                "axes.labelsize": 11,
                "axes.titlesize": 12,
                # Legend options
                "legend.framealpha": 0.8,
                "legend.loc": "upper right",
            }
        )

    def get_plot_config(self, plot_type: str, dimension: str = "all") -> Dict:
        """
        Return configuration for specified plot type and dimension.

        Args:
        - plot_type: Type of data ("intensity" or "density")
        - dimension: Plot dimension ("line", "map", "3d", or "all" for complete config)

        Returns:
        - Dictionary with plot configuration settings
        """
        base_config = {
            "intensity": {
                "cmap": self.colormaps["intensity"],
                "colorbar_label": r"Intensity $\left[\mathrm{W/cm^2}\right]$",
                "titles": {
                    "z": "On-axis peak intensity over time",
                    "t": r"On-axis intensity at $z = {:.2f}$ $\mathrm{m}$",
                    "zt": "On-axis intensity",
                    "rt": r"Intensity at $z = {:.2f}$ $\mathrm{m}$",
                    "rz": "Peak intensity over time",
                },
                "labels": {
                    "x_r": r"$r$ $[\mathrm{mm}]$",
                    "x_z": r"$z$ $[\mathrm{m}]$",
                    "x_t": r"$t$ $[\mathrm{fs}]$",
                    "y_t": r"$I(r=0,t)$ $\left[\mathrm{W/cm^2}\right]$",
                    "y_z": r"$\max_t$ $I(r=0,z,t)$ $\left[\mathrm{W/cm^2}\right]$",
                },
            },
            "density": {
                "cmap": self.colormaps["density"],
                "colorbar_label": r"Electron density $\left[\mathrm{cm^{-3}}\right]$",
                "titles": {
                    "z": "On-axis peak electron density over time",
                    "t": r"On-axis electron density at $z = {:.2f}$ $\mathrm{m}$",
                    "zt": "On-axis electron density",
                    "rt": r"Electron density at $z = {:.2f}$ $\mathrm{m}$",
                    "rz": "Peak electron density over time",
                },
                "labels": {
                    "x_r": r"$r$ $[\mathrm{mm}]$",
                    "x_z": r"$z$ $[\mathrm{m}]$",
                    "x_t": r"$t$ $[\mathrm{fs}]$",
                    "y_t": r"$\rho(r=0,t)$ $\left[\mathrm{cm^{-3}}\right]$",
                    "y_z": r"$\max_t$ $\rho(r=0,z,t)$ $\left[\mathrm{cm^{-3}}\right]$",
                },
            },
        }

        # Dimension-specific configuration
        dimension_config = {
            "line": {"dpi": 150},
            "map": {"dpi": 150},
            "3d": {
                "resolutions": {
                    "low": {"stride": (5, 5), "dpi": 100, "antialiased": False},
                    "medium": {"stride": (2, 2), "dpi": 150, "antialiased": True},
                    "high": {"stride": (1, 1), "dpi": 300, "antialiased": True},
                },
            },
        }

        # Return requested configuration
        if dimension == "all":
            return {
                "base": base_config[plot_type],
                **dimension_config,
            }

        return {**base_config[plot_type], **dimension_config.get(dimension, {})}


class Units:
    "Unit constants for converting between SI multiples and subdivisions."

    def __init__(
        self, factor_r=1e3, factor_z=1, factor_t=1e15, factor_m2=1e-4, factor_m3=1e-6
    ):

        self.factor_r = factor_r
        self.factor_z = factor_z
        self.factor_t = factor_t
        self.factor_m2 = factor_m2
        self.factor_m3 = factor_m3


class SimulationBox:
    """Plotting grid box-sizing."""

    def __init__(
        self,
        units: Units,
        data: Dict[str, Any],
        symmetry: bool = False,
        radial_limit: float = None,
        time_limit: float = None,
    ):
        self.units = units
        self.data = data
        self.symmetry = symmetry
        self.radial_limit = radial_limit
        self.time_limit = time_limit
        self._initialize_boundaries()
        self._initialize_grid_nodes()
        self._initialize_sliced_arrays()

    def _initialize_boundaries(self):
        """Set up the plotting box boundary."""
        self.r_min_ori = self.data["ini_radi_coor"]
        self.r_max_ori = self.data["fin_radi_coor"]
        self.z_min_ori = self.data["ini_dist_coor"]
        self.z_max_ori = self.data["fin_dist_coor"]
        self.t_min_ori = self.data["ini_time_coor"]
        self.t_max_ori = self.data["fin_time_coor"]

        r_min = self.data["ini_radi_coor"]
        r_max = self.data["fin_radi_coor"]
        z_min = self.data["ini_dist_coor"]
        z_max = self.data["fin_dist_coor"]
        t_min = self.data["ini_time_coor"]
        t_max = self.data["fin_time_coor"]

        if self.radial_limit is not None and self.radial_limit < r_max:
            r_max = self.radial_limit
            r_max_units = r_max * self.units.factor_r
            r_max_ori_units = self.r_max_ori * self.units.factor_r
            print(
                f"Radial grid maximum set from {r_max_ori_units} mm to {r_max_units:.2f} mm"
            )

        if self.time_limit is not None and self.time_limit < t_max:
            t_min = -self.time_limit
            t_max = self.time_limit
            t_max_units = t_max * self.units.factor_t
            t_max_ori_units = self.t_max_ori * self.units.factor_t
            print(
                f"Time grid maximum set from {t_max_ori_units} fs to {t_max_units} fs"
            )

        if self.symmetry:
            r_min = -r_max

        self.boundary_r = (r_min, r_max)
        self.boundary_z = (z_min, z_max)
        self.boundary_t = (t_min, t_max)

    def _initialize_grid_nodes(self):
        """Set up the plotting box boundary nodes."""
        self.nodes_r = self.data["e_dist"].shape[0]
        self.nodes_z = self.data["e_axis"].shape[0]
        self.nodes_t = self.data["e_axis"].shape[1]

        if self.symmetry:
            self.nodes_r_sym = 2 * self.nodes_r - 1

        self.nodes = {}
        for dim, (min_b, max_b, n_nodes, mini, maxi) in {
            "r_data": (
                *self.boundary_r,
                self.nodes_r if not self.symmetry else self.nodes_r_sym,
                (self.r_min_ori if not self.symmetry else -self.r_max_ori),
                self.r_max_ori,
            ),
            "z_data": (
                *self.boundary_z,
                self.nodes_z,
                self.z_min_ori,
                self.z_max_ori,
            ),
            "t_data": (
                *self.boundary_t,
                self.nodes_t,
                self.t_min_ori,
                self.t_max_ori,
            ),
        }.items():
            node_min = (min_b - mini) * (n_nodes - 1) / (maxi - mini)
            node_max = (max_b - mini) * (n_nodes - 1) / (maxi - mini)
            self.nodes[dim] = (int(node_min), int(node_max) + 1)

        if self.symmetry:
            self.node_r0 = self.nodes_r - 1
        else:
            self.node_r0 = 0

    def _initialize_sliced_arrays(self):
        """Set up computational arrays"""
        self.sliced_data = {}
        self.sliced_obj = {  # Get elements from n_min to n_max
            "r": slice(*self.nodes["r_data"]),
            "z": slice(*self.nodes["z_data"]),
            "t": slice(*self.nodes["t_data"]),
        }

        if self.symmetry:
            radi_positive = np.linspace(0, self.r_max_ori, self.nodes_r)
            radi_negative = -np.flip(radi_positive[:-1])
            radi = np.concatenate((radi_negative, radi_positive))
            radi_slice = radi[self.sliced_obj["r"]]
        else:
            radi_slice = np.linspace(self.r_min_ori, self.r_max_ori, self.nodes_r)[
                self.sliced_obj["r"]
            ]

        # Create sliced grids
        self.sliced_grids = {
            "r": radi_slice,
            "z": np.linspace(self.z_min_ori, self.z_max_ori, self.nodes_z)[
                self.sliced_obj["z"]
            ],
            "t": np.linspace(self.t_min_ori, self.t_max_ori, self.nodes_t)[
                self.sliced_obj["t"]
            ],
        }

        # Slice electric field data if present
        if "e_dist" in self.data:
            if self.symmetry:
                self.sliced_data["e_dist"] = self.flip_radial_data(
                    self.data["e_dist"], axis_r=0
                )[self.sliced_obj["r"], :, self.sliced_obj["t"]]
            else:
                self.sliced_data["e_dist"] = self.data["e_dist"][
                    self.sliced_obj["r"], :, self.sliced_obj["t"]
                ]
        if "e_axis" in self.data:
            self.sliced_data["e_axis"] = self.data["e_axis"][
                self.sliced_obj["z"], self.sliced_obj["t"]
            ]
        if "e_peak" in self.data:
            if self.symmetry:
                self.sliced_data["e_peak"] = self.flip_radial_data(
                    self.data["e_peak"], axis_r=0
                )[self.sliced_obj["r"], self.sliced_obj["z"]]
            else:
                self.sliced_data["e_peak"] = self.data["e_peak"][
                    self.sliced_obj["r"], self.sliced_obj["z"]
                ]

        # Slice electron density data if present
        if "elec_dist" in self.data:
            if self.symmetry:
                self.sliced_data["elec_dist"] = self.flip_radial_data(
                    self.data["elec_dist"], axis_r=0
                )[self.sliced_obj["r"], :, self.sliced_obj["t"]]
            else:
                self.sliced_data["elec_dist"] = self.data["elec_dist"][
                    self.sliced_obj["r"], :, self.sliced_obj["t"]
                ]
        if "elec_axis" in self.data:
            self.sliced_data["elec_axis"] = self.data["elec_axis"][
                self.sliced_obj["z"], self.sliced_obj["t"]
            ]
        if "elec_peak" in self.data:
            if self.symmetry:
                self.sliced_data["elec_peak"] = self.flip_radial_data(
                    self.data["elec_peak"], axis_r=0
                )[self.sliced_obj["r"], self.sliced_obj["z"]]
            else:
                self.sliced_data["elec_peak"] = self.data["elec_peak"][
                    self.sliced_obj["r"], self.sliced_obj["z"]
                ]

    def set_snapshot_points(self, indices):
        """Convert k-indices to their corresponding z-coordinates."""
        indices = np.array(indices)  # make sure it's a numpy array
        z_min = self.data["ini_dist_coor"] * self.units.factor_z
        z_max = self.data["fin_dist_coor"] * self.units.factor_z
        z_snap_coor = z_min + (indices * (z_max - z_min) / (self.nodes_z - 1))
        return z_snap_coor

    def flip_radial_data(self, data, axis_r=0):
        """Mirror radial data for symmetry."""
        if not self.symmetry:
            return data

        data_flip = np.flip(data, axis=axis_r)
        if data.shape[axis_r] > 1:
            return np.concatenate((data_flip[:-1], data), axis=axis_r)

        return np.concatenate((data, data), axis=axis_r)


class SimulationBoxUnits:
    """Plotting grid box with custom units."""

    def __init__(self, units: Units, box: SimulationBox, config: PlotConfiguration):
        self.units = units
        self.box = box
        self.config = config
        self._scaled_1d_grid = {}
        self._scaled_2d_grid = {}

    def create_unit_scaled_1d_grid(self, grid_type):
        """Get a scaled array, creating it if necessary."""
        if grid_type not in self._scaled_1d_grid:
            if grid_type.startswith("r"):
                self._scaled_1d_grid[grid_type] = (
                    self.units.factor_r * self.box.sliced_grids[grid_type]
                )
            elif grid_type.startswith("z"):
                self._scaled_1d_grid[grid_type] = (
                    self.units.factor_z * self.box.sliced_grids[grid_type]
                )
            elif grid_type.startswith("t"):
                self._scaled_1d_grid[grid_type] = (
                    self.units.factor_t * self.box.sliced_grids[grid_type]
                )

        return self._scaled_1d_grid[grid_type]

    def create_unit_scaled_2d_grid(self, grid_type):
        """Set up meshgrids only when needed."""
        if grid_type not in self._scaled_2d_grid:
            if grid_type == "zt":
                self._scaled_2d_grid[grid_type] = np.meshgrid(
                    self.create_unit_scaled_1d_grid("z"),
                    self.create_unit_scaled_1d_grid("t"),
                    indexing="ij",
                )
            elif grid_type == "rt":
                self._scaled_2d_grid[grid_type] = np.meshgrid(
                    self.create_unit_scaled_1d_grid("r"),
                    self.create_unit_scaled_1d_grid("t"),
                    indexing="ij",
                )
            elif grid_type == "rz":
                self._scaled_2d_grid[grid_type] = np.meshgrid(
                    self.create_unit_scaled_1d_grid("r"),
                    self.create_unit_scaled_1d_grid("z"),
                    indexing="ij",
                )

        return self._scaled_2d_grid[grid_type]


class BasePlotter:
    """Plotting class for all simulation data."""

    def __init__(
        self,
        units: Units,
        box: SimulationBox,
        config: PlotConfiguration,
        box_units: SimulationBoxUnits,
    ):
        self.units = units
        self.box = box
        self.config = config
        self.box_units = box_units

    def calculate_intensity(self, envelope_dist, envelope_axis, envelope_peak):
        """Set up intensities for plotting."""
        return (
            self.units.factor_m2 * np.abs(envelope_dist) ** 2,
            self.units.factor_m2 * np.abs(envelope_axis) ** 2,
            self.units.factor_m2 * np.abs(envelope_peak) ** 2,
        )

    def calculate_density(self, density_dist, density_axis, density_peak):
        """Set up densities for plotting."""
        return (
            self.units.factor_m3 * density_dist,
            self.units.factor_m3 * density_axis,
            self.units.factor_m3 * density_peak,
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

    def get_1d_grid(self, grid_type):
        """Access box-scaled 1d-grids."""
        return self.box_units.create_unit_scaled_1d_grid(grid_type)

    def get_2d_grid(self, grid_type):
        """Access box-scaled 2d-grid."""
        return self.box_units.create_unit_scaled_2d_grid(grid_type)


class PlotterLine(BasePlotter):
    """Plotting class for line plots."""

    def render_line_data(
        self,
        data,
        k_array=None,
        z_coor=None,
        plot_type="intensity",
        save_path=None,
    ):
        """
        Create line plots for intensity or density.

        Arguments:
            data: Dictionary containing the datasets for different coordinates:
            plot_type: "intensity" or "density".
            save_path: Path to save figures instead of displaying them.
        """
        plot_config = self.config.get_plot_config(plot_type, "line")

        # Plot each coordinate system in a separate figure
        for coord_sys, plot_data in data.items():
            if coord_sys == "rt" and k_array is not None:
                # Plot intensity or density for each z-position with respect to time
                time_array = self.get_1d_grid("t")
                for idx in range(len(k_array)):
                    fig, ax = plt.subplots()
                    ax.plot(
                        time_array,
                        plot_data[self.box.node_r0, idx, :],
                    )

                    # Get actual z-position
                    z_pos = z_coor[idx]
                    z_pos_format = f"{z_pos:.2f}"
                    title = plot_config["titles"]["t"].replace("{:.2f}", z_pos_format)
                    ax.set_title(title)

                    ax.set(
                        xlabel=plot_config["labels"]["x_t"],
                        ylabel=plot_config["labels"]["y_t"],
                    )

                    filename = (
                        f"line_{plot_type}_t_{z_pos:.2f}".replace(".", "-") + ".png"
                    )
                    self.save_or_display(fig, filename, save_path, plot_config["dpi"])

            elif coord_sys == "rz":
                # Plot intensity or density peak value on-axis with respect to distance
                dist_array = self.get_1d_grid("z")
                fig, ax = plt.subplots()

                ax.plot(
                    dist_array,
                    plot_data[self.box.node_r0, :],
                )
                ax.set(
                    xlabel=plot_config["labels"]["x_z"],
                    ylabel=plot_config["labels"]["y_z"],
                )
                ax.set_title(plot_config["titles"]["z"])

                filename = f"line_{plot_type}_z.png"
                self.save_or_display(fig, filename, save_path, plot_config["dpi"])


class PlotterMap(BasePlotter):
    """Plotting class for map plots."""

    def render_map_data(
        self,
        data,
        k_array=None,
        z_coor=None,
        plot_type="intensity",
        save_path=None,
        log_scale=False,
    ):
        """
        Create map plots for different coordinate systems.

        Arguments:
            data: Dictionary containing the datasets for different coordinates.
            k_array: List of z indices to plot (for rt plots).
            z_coor: List of z coordinates corresponding to the k indices saved.
            plot_type: "intensity" or "density".
            save_path: Path to save figures instead of displaying them.
        """
        # Configuration for different plot types
        plot_config = self.config.get_plot_config(plot_type, "map")

        # Plot each coordinate system in a separate figure
        for coord_sys, plot_data in data.items():
            if coord_sys == "rt" and k_array is not None:
                # Plot intensity or density for each z position
                for idx in range(len(k_array)):
                    fig, ax = plt.subplots()

                    # Get the meshgrid
                    x, y = self.get_2d_grid("rt")
                    xlabel = plot_config["labels"]["x_r"]
                    ylabel = plot_config["labels"]["x_t"]

                    # Plot in logarithmic scale if requested
                    if log_scale and plot_type == "intensity":
                        mesh = ax.pcolormesh(
                            x,
                            y,
                            plot_data[:, idx, :],
                            cmap=self.config.colormaps["intensity_log"],
                            norm=LogNorm(
                                vmin=plot_data[:, idx, :].min(),
                                vmax=plot_data[:, idx, :].max(),
                            ),
                        )
                        colorbar_label = plot_config["colorbar_label"] + " (log scale)"
                    else:
                        mesh = ax.pcolormesh(
                            x,
                            y,
                            plot_data[:, idx, :],
                            cmap=plot_config["cmap"],
                        )
                        colorbar_label = plot_config["colorbar_label"]

                    fig.colorbar(mesh, ax=ax, label=colorbar_label)
                    ax.set(xlabel=xlabel, ylabel=ylabel)

                    # Get actual z-position
                    z_pos = z_coor[idx]
                    z_pos_format = f"{z_pos:.2f}"
                    title = plot_config["titles"][coord_sys].replace(
                        "{:.2f}", z_pos_format
                    )
                    ax.set_title(title)

                    filename = (
                        f"map_{plot_type}_{coord_sys}_{z_pos:.2f}".replace(".", "-")
                        + ".png"
                    )
                    self.save_or_display(fig, filename, save_path, plot_config["dpi"])

            else:
                fig, ax = plt.subplots()

                if coord_sys == "zt":
                    # Plot intensity or density on-axis
                    x, y = self.get_2d_grid("zt")
                    xlabel = plot_config["labels"]["x_z"]
                    ylabel = plot_config["labels"]["x_t"]
                else:
                    # Plot intensity or density peak value
                    x, y = self.get_2d_grid("rz")
                    xlabel = plot_config["labels"]["x_r"]
                    ylabel = plot_config["labels"]["x_z"]

                mesh = ax.pcolormesh(x, y, plot_data, cmap=plot_config["cmap"])
                fig.colorbar(mesh, ax=ax, label=plot_config["colorbar_label"])
                ax.set(xlabel=xlabel, ylabel=ylabel)
                ax.set_title(plot_config["titles"][coord_sys])

                filename = f"map_{plot_type}_{coord_sys}.png"
                self.save_or_display(fig, filename, save_path, plot_config["dpi"])


class Plotter3D(BasePlotter):
    """Plotting class for 3D plots."""

    def render_3d_data(
        self,
        data,
        k_array,
        z_coor=None,
        plot_type="intensity",
        resolution="medium",
        save_path=None,
    ):
        """
        Create 3D plots for different coordinate systems.

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

        for coord_sys, plot_data in data.items():
            if coord_sys == "rt" and k_array is not None:
                # Plot intensity or density for each z position
                for idx in range(len(k_array)):
                    fig = plt.figure(dpi=resolution_config["dpi"])
                    ax = fig.add_subplot(projection="3d")

                    # Get the meshgrid
                    x, y = self.get_2d_grid("rt")
                    xlabel = plot_config["labels"]["x_r"]
                    ylabel = plot_config["labels"]["x_t"]

                    ax.plot_surface(
                        x[:: stride[0], :: stride[1]],
                        y[:: stride[0], :: stride[1]],
                        plot_data[:: stride[0], idx, :: stride[1]],
                        cmap=plot_config["cmap"],
                        linewidth=0,
                        antialiased=resolution_config["antialiased"],
                    )
                    # fig.colorbar(surf, label=plot_config["colorbar_label"])
                    ax.set(
                        xlabel=xlabel,
                        ylabel=ylabel,
                        zlabel=plot_config["colorbar_label"],
                    )
                    # Get actual z-position
                    z_pos = z_coor[idx]
                    z_pos_format = f"{z_pos:.2f}"
                    title = plot_config["titles"][coord_sys].replace(
                        "{:.2f}", z_pos_format
                    )
                    ax.set_title(title)

                    filename = (
                        f"3d_{plot_type}_{coord_sys}_{z_pos:.2f}".replace(".", "-")
                        + ".png"
                    )
                    self.save_or_display(
                        fig, filename, save_path, resolution_config["dpi"]
                    )

            else:
                fig = plt.figure(dpi=resolution_config["dpi"])
                ax = fig.add_subplot(projection="3d")

                if coord_sys == "zt":
                    # Plot intensity or density on-axis
                    x, y = self.get_2d_grid("zt")
                    xlabel = plot_config["labels"]["x_z"]
                    ylabel = plot_config["labels"]["x_t"]
                else:
                    # Plot intensity or density peak value
                    x, y = self.get_2d_grid("rz")
                    xlabel = plot_config["labels"]["x_r"]
                    ylabel = plot_config["labels"]["x_z"]

                # Apply stride for better performance
                ax.plot_surface(
                    x[:: stride[0], :: stride[1]],
                    y[:: stride[0], :: stride[1]],
                    plot_data[:: stride[0], :: stride[1]],
                    cmap=plot_config["cmap"],
                    linewidth=0,
                    antialiased=resolution_config["antialiased"],
                )
                # fig.colorbar(surf, label=plot_config["colorbar_label"])
                ax.set(
                    xlabel=xlabel,
                    ylabel=ylabel,
                    zlabel=plot_config["colorbar_label"],
                )
                ax.set_title(plot_config["titles"][coord_sys])

                filename = f"3d_{plot_type}_{coord_sys}.png"
                self.save_or_display(fig, filename, save_path, resolution_config["dpi"])


class VisualManager:
    """Manages all plotting classes."""

    def __init__(self, units, box, config, box_units):
        self.units = units
        self.box = box
        self.config = config
        self.box_units = box_units

        # Initialize specialized plotters
        self.base_plotter = BasePlotter(units, box, config, box_units)
        self.plot_line = PlotterLine(units, box, config, box_units)
        self.plot_map = PlotterMap(units, box, config, box_units)
        self.plot_3d = Plotter3D(units, box, config, box_units)

    def get_intensity_data(self, envelope_dist, envelope_axis, envelope_peak):
        """Calculate intensity data."""
        return self.base_plotter.calculate_intensity(
            envelope_dist, envelope_axis, envelope_peak
        )

    def get_density_data(self, density_dist, density_axis, density_peak):
        """Calculate density data."""
        return self.base_plotter.calculate_density(
            density_dist, density_axis, density_peak
        )

    def create_line_plot(
        self, data, k_array=None, z_coor=None, plot_type="intensity", save_path=None
    ):
        """Create line plots."""
        self.plot_line.render_line_data(data, k_array, z_coor, plot_type, save_path)

    def create_map_plot(
        self,
        data,
        k_array=None,
        z_coor=None,
        plot_type="intensity",
        save_path=None,
        log_scale=False,
    ):
        """Create map plots."""
        self.plot_map.render_map_data(
            data, k_array, z_coor, plot_type, save_path, log_scale
        )

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
        self.plot_3d.render_3d_data(
            data, k_array, z_coor, plot_type, resolution, save_path
        )


def parse_cli_options():
    """Parse and validate CLI options."""
    parser = argparse.ArgumentParser(
        description="Plot simulation data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase output verbosity (can be used multiple times).",
    )
    parser.add_argument(
        "--file",
        default=DEFAULT_DATA_FILE_PATH,
        help="Path to data file (.npz format).",
    )
    parser.add_argument(
        "--save-path",
        default=DEFAULT_SAVE_PATH,
        help="Directory to save plots instead of displaying.",
    )
    parser.add_argument(
        "--data-types",
        default="intensity,density",
        help="Data types to plot: intensity,density (comma-separated).",
    )
    parser.add_argument(
        "--plot-types",
        default="line,map,3d",
        help="Plot types to generate: line,map,3d (comma-separated).",
    )
    parser.add_argument(
        "--resolution",
        default="medium",
        help="Plot quality for 3D plots: low, medium, high.",
    )
    parser.add_argument(
        "--radial-limit",
        type=float,
        default=None,
        help="Maximum value for radial grid (in meters).",
    )
    parser.add_argument(
        "--symmetric",
        action="store_true",
        default=False,
        help="Plot radial data symmetrically.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="Maximum value for time grid (in seconds).",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        default=False,
        help="Use logarithmic scale for intensity colormaps.",
    )

    args = parser.parse_args()

    # Convert comma-separated strings to dictionaries for easier access
    args.data_types = {dtype: True for dtype in args.data_types.split(",")}
    args.plot_types = {ptype: True for ptype in args.plot_types.split(",")}

    return args


def setup_output_directory(args):
    """Setup environment for plotting based on arguments."""
    # Ask if we have a save path when required
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
        if file_size > 500:  # larger than 500 MB
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


def process_plot_request(
    data_type, plot_data, plot_types, plot, z_snap_idx, z_snap_coor, args
):
    """Generate requested plot types for the specified data."""

    if plot_types.get("line", False):
        print(f"Generating line plots for {data_type} ...")
        plot.create_line_plot(
            plot_data,
            z_snap_idx,
            z_snap_coor,
            data_type,
            args.save_path,
        )

    if plot_types.get("map", False):
        print(f"Generating map plots for {data_type}")
        plot.create_map_plot(
            plot_data,
            z_snap_idx,
            z_snap_coor,
            data_type,
            args.save_path,
            args.log_scale,
        )

    if plot_types.get("3d", False):
        print(f"Generating 3D plots for {data_type}")
        try:
            plot.create_3d_plot(
                plot_data,
                z_snap_idx,
                z_snap_coor,
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

    # Ask if required arrays exist
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
                    box.nodes_r,
                    len(data["k_array"]),
                    box.nodes_t,
                )
            elif key.endswith("_axis"):
                expected_shape = (box.nodes_z, box.nodes_t)
            elif key.endswith("_peak"):
                expected_shape = (box.nodes_r, box.nodes_z)

            if expected_shape and data[key].shape != expected_shape:
                print(
                    f"Warning: {key} has unexpected shape: {data[key].shape}, expected {expected_shape}"
                )

    missing = [arr for arr in required_arrays if arr not in data]
    if missing:
        print(f"Warning: Missing required data for {data_type}: {', '.join(missing)}")
        print(f"Omitting {data_type} plots")
        return

    # Load snapshot data
    z_snap_idx = data["k_array"]
    z_snap_coor = box.set_snapshot_points(z_snap_idx)

    # Calculate data based on type
    if data_type == "intensity":
        plot_data_dist, plot_data_axis, plot_data_peak = plot.get_intensity_data(
            box.sliced_data["e_dist"],
            box.sliced_data["e_axis"],
            box.sliced_data["e_peak"],
        )
    elif data_type == "density":
        plot_data_dist, plot_data_axis, plot_data_peak = plot.get_density_data(
            box.sliced_data["elec_dist"],
            box.sliced_data["elec_axis"],
            box.sliced_data["elec_peak"],
        )
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

    # Set up data dictionary for plotting functions
    plot_data = {
        "rt": plot_data_dist,
        "zt": plot_data_axis,
        "rz": plot_data_peak,
    }

    # Generate requested plot types
    process_plot_request(
        data_type, plot_data, plot_types, plot, z_snap_idx, z_snap_coor, args
    )


def main():
    """Main execution function."""
    print(f"Simulation plotter {__version__}")

    # Initialize CLI arguments parsing
    args = parse_cli_options()

    # Initialize output directory
    setup_output_directory(args)

    # Load data
    data = load_simulation_data(args.file, args)

    if args.verbose > 0:
        print("Data file contents:")
        print("   Data available:", ", ".join(data.keys()))
        print(
            f"   Data dimensions: {data["e_axis"].shape if "e_axis" in data else "unknown"}"
        )
        print(f"   Radial symmetry: {'enabled' if args.symmetric else 'disabled'}")

    # Initialize classes
    units = Units()
    config = PlotConfiguration()
    box = SimulationBox(
        units,
        data,
        symmetry=args.symmetric,
        radial_limit=args.radial_limit,
        time_limit=args.time_limit,
    )
    box_units = SimulationBoxUnits(units, box, config)
    plot = VisualManager(units, box, config, box_units)

    # Process each requested data type
    for data_type, enabled in args.data_types.items():
        if enabled and data_type in ["intensity", "density"]:
            process_simulation_data(data_type, data, plot, box, args.plot_types, args)

    print("Plotting complete!")


if __name__ == "__main__":
    main()
