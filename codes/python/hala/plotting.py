"""
Python tool for plotting NumPy arrays saved after
the simulations have finished execution.
"""

import argparse
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from ._version import __version__

base_dir = Path("./path_to_base_directory")

sim_dir = base_dir / "data" / "sim_folder_name"
fig_dir = base_dir / "figures" / "sim_folder_name"


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
            "fluence": mpl.colormaps["turbo"],
            "intensity_log": mpl.colormaps["inferno"],
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
        - dimension: Plot dimension ("1d", "2d", "3d", or "all" for complete config)

        Returns:
        - Dictionary with plot configuration settings
        """
        base_config = {
            "intensity": {
                "cmap": self.colormaps["intensity"],
                "colorbar_label": r"Intensity $\Bigl[\mathrm{W/cm^2}\Bigr]$",
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
                    "y_t": r"$I(r=0,t)$ $\Bigl[\mathrm{W/cm^2}\Bigr]$",
                    "y_z": r"$\max_t$ $I(r=0,z,t)$ $\Bigl[\mathrm{W/cm^2}\Bigr]$",
                },
            },
            "density": {
                "cmap": self.colormaps["density"],
                "colorbar_label": r"Electron density $\Bigl[\mathrm{cm^{-3}}\Bigr]$",
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
                    "y_t": r"$\rho(r=0,t)$ $\Bigl[\mathrm{cm^{-3}}\Bigr]$",
                    "y_z": r"$\max_t$ $\rho(r=0,z,t)$ $\Bigl[\mathrm{cm^{-3}}\Bigr]$",
                },
            },
            "fluence": {
                "cmap": self.colormaps["fluence"],
                "colorbar_label": r"Fluence $\Bigl[\mathrm{mJ/cm^2}\Bigr]$",
                "titles": {
                    "z": "On-axis fluence distribution",
                    "rz": "Fluence distribution",
                },
                "labels": {
                    "x_r": r"$r$ $[\mathrm{mm}]$",
                    "x_z": r"$z$ $[\mathrm{m}]$",
                    "y_z": r"$F(r=0,z)$ $\Bigl[\mathrm{mJ/cm^2}\Bigr]$",
                },
            },
            "radius": {
                "cmap": self.colormaps["fluence"],
                "titles": {
                    "z": "Beam radius",
                },
                "labels": {
                    "x_z": r"$z$ $[\mathrm{m}]$",
                    "y_z": r"$R(z)$ $[\mathrm{mm}]$",
                },
            },
        }

        # Dimension-specific configuration
        dimension_config = {
            "1d": {"dpi": 150},
            "2d": {
                "resolutions": {
                    "low": {"stride": (5, 5), "dpi": 100, "antialiased": False},
                    "medium": {"stride": (2, 2), "dpi": 150, "antialiased": True},
                    "high": {"stride": (1, 1), "dpi": 300, "antialiased": True},
                },
            },
            "3d": {
                "resolutions": {
                    "low": {"stride": (5, 5), "dpi": 100, "antialiased": False},
                    "medium": {"stride": (2, 2), "dpi": 150, "antialiased": True},
                    "high": {"stride": (1, 1), "dpi": 300, "antialiased": True},
                },
                "camera_angles": {
                    "rt": {
                        "elevation": 15,
                        "azimuth": 200,
                        "rotation": 0,
                    },
                    "zt": {
                        "elevation": 10,
                        "azimuth": 235,
                        "rotation": 0,
                    },
                    "rz": {
                        "elevation": 10,
                        "azimuth": 325,
                        "rotation": 0,
                    },
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
        self,
        factor_r=1e3,
        factor_z=1,
        factor_t=1e15,
        factor_m2=1e-4,
        factor_m3=1e-6,
        factor_j=1e3,
    ):

        self.factor_r = factor_r
        self.factor_z = factor_z
        self.factor_t = factor_t
        self.factor_m2 = factor_m2
        self.factor_m3 = factor_m3
        self.factor_j = factor_j


class SimulationBox:
    """Plotting grid box-sizing."""

    def __init__(
        self,
        units: Units,
        data: Dict[str, Any],
        radial_symmetry: bool = False,
        radial_limit: float = None,
        axial_range: tuple = None,
        time_range: tuple = None,
    ):
        self.units = units
        self.data = data
        self.radial_symmetry = radial_symmetry
        self.radial_limit = radial_limit
        self.axial_range = axial_range
        self.time_range = time_range
        self._init_boundaries()
        self._init_grid_nodes()
        self._init_sliced_arrays()

    def _init_boundaries(self):
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

        if self.axial_range is not None:
            z_min_new, z_max_new = self.axial_range
            if z_min_new > z_min:
                z_min = z_min_new
                z_min_units = z_min * self.units.factor_z
                z_min_ori_units = self.z_min_ori * self.units.factor_z
                print(
                    f"Axial minimum set from {z_min_ori_units:.2f} m to {z_min_units:.2f} m"
                )
            if z_max_new < z_max:
                z_max = z_max_new
                z_max_units = z_max * self.units.factor_z
                z_max_ori_units = self.z_max_ori * self.units.factor_z
                print(
                    f"Axial maximum set from {z_max_ori_units:.2f} m to {z_max_units:.2f} m"
                )

        if self.time_range is not None:
            t_min_new, t_max_new = self.time_range
            if t_min_new > t_min:
                t_min = t_min_new
                t_min_units = t_min * self.units.factor_t
                t_min_ori_units = self.t_min_ori * self.units.factor_t
                print(
                    f"Time minimum set from {t_min_ori_units:.2f} fs to {t_min_units:.2f} fs"
                )
            if t_max_new < t_max:
                t_max = t_max_new
                t_max_units = t_max * self.units.factor_t
                t_max_ori_units = self.t_max_ori * self.units.factor_t
                print(
                    f"Time maximum set from {t_max_ori_units:.2f} fs to {t_max_units:.2f} fs"
                )

        if self.radial_symmetry:
            r_min = -r_max

        self.boundary_r = (r_min, r_max)
        self.boundary_z = (z_min, z_max)
        self.boundary_t = (t_min, t_max)

    def _init_grid_nodes(self):
        """Set up the plotting box boundary nodes."""
        self.nodes_r = self.data["e_dist"].shape[0]
        self.nodes_z = self.data["e_axis"].shape[0]
        self.nodes_t = self.data["e_axis"].shape[1]

        if self.radial_symmetry:
            self.nodes_r_sym = 2 * self.nodes_r - 1

        self.nodes = {}
        for dim, (min_b, max_b, n_nodes, mini, maxi) in {
            "r_data": (
                *self.boundary_r,
                self.nodes_r if not self.radial_symmetry else self.nodes_r_sym,
                (self.r_min_ori if not self.radial_symmetry else -self.r_max_ori),
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

        if self.radial_symmetry:
            self.node_r0 = self.nodes_r - 1
        else:
            self.node_r0 = 0

    def _init_sliced_arrays(self):
        """Set up computational arrays"""
        self.sliced_data = {}
        self.sliced_obj = {  # Get elements from n_min to n_max
            "r": slice(*self.nodes["r_data"]),
            "z": slice(*self.nodes["z_data"]),
            "t": slice(*self.nodes["t_data"]),
        }

        if self.radial_symmetry:
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
            if self.radial_symmetry:
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
            if self.radial_symmetry:
                self.sliced_data["e_peak"] = self.flip_radial_data(
                    self.data["e_peak"], axis_r=0
                )[self.sliced_obj["r"], self.sliced_obj["z"]]
            else:
                self.sliced_data["e_peak"] = self.data["e_peak"][
                    self.sliced_obj["r"], self.sliced_obj["z"]
                ]

        # Slice electron density data if present
        if "elec_dist" in self.data:
            if self.radial_symmetry:
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
            if self.radial_symmetry:
                self.sliced_data["elec_peak"] = self.flip_radial_data(
                    self.data["elec_peak"], axis_r=0
                )[self.sliced_obj["r"], self.sliced_obj["z"]]
            else:
                self.sliced_data["elec_peak"] = self.data["elec_peak"][
                    self.sliced_obj["r"], self.sliced_obj["z"]
                ]

        # Slice beam fluence distribution data if present
        if "b_fluence" in self.data:
            if self.radial_symmetry:
                self.sliced_data["b_fluence"] = self.flip_radial_data(
                    self.data["b_fluence"], axis_r=0
                )[self.sliced_obj["r"], self.sliced_obj["z"]]
            else:
                self.sliced_data["b_fluence"] = self.data["b_fluence"][
                    self.sliced_obj["r"], self.sliced_obj["z"]
                ]

        # Slice beam radius data if present
        if "b_radius" in self.data:
            self.sliced_data["b_radius"] = self.data["b_radius"][self.sliced_obj["z"]]

    def set_snapshot_points(self, indices):
        """Convert k-indices to their corresponding z-coordinates."""
        z_min = self.data["ini_dist_coor"] * self.units.factor_z
        z_max = self.data["fin_dist_coor"] * self.units.factor_z
        z_snap_coor = z_min + (indices * (z_max - z_min) / (self.nodes_z - 1))
        return z_snap_coor

    def flip_radial_data(self, data, axis_r=0):
        """Mirror radial data for symmetry."""
        if not self.radial_symmetry:
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


class BasePlot:
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

    def calculate_fluence(self, b_fluence):
        """Set up fluence distribution for plotting."""
        return self.units.factor_m2 * self.units.factor_j * b_fluence

    def calculate_radius(self, b_radius):
        """Set up beam radius for plotting."""
        return self.units.factor_r * b_radius

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


class Plot1D(BasePlot):
    """Plotting class for 1D (line) plots."""

    def render_1d_data(
        self,
        data,
        k_array=None,
        z_coor=None,
        plot_type="intensity",
        save_path=None,
    ):
        """
        Create 1D (line) plots for different coordinate systems.

        Arguments:
            data: Dictionary containing the datasets for different coordinates:
            plot_type: "intensity", "density" or "radius".
            save_path: Path to save figures instead of displaying them.
        """
        plot_config = self.config.get_plot_config(plot_type, "1d")

        # Plot each coordinate system in a separate figure
        for coord_sys, plot_data in data.items():
            if coord_sys == "rt" and k_array is not None:
                # Plot intensity or density for each z-position
                # with respect to time
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
                        f"1d_{plot_type}_t_{z_pos:.2f}".replace(".", "-") + ".png"
                    )
                    self.save_or_display(fig, filename, save_path, plot_config["dpi"])

            elif coord_sys == "rz":
                # Plot intensity or density peak value on-axis
                # with respect to distance
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

                filename = f"1d_{plot_type}_z.png"
                self.save_or_display(fig, filename, save_path, plot_config["dpi"])

            elif coord_sys == "z" and plot_type == "radius":
                # Plot beam radius with respect to distance
                dist_array = self.get_1d_grid("z")
                fig, ax = plt.subplots()

                if self.box.radial_symmetry:
                    y_max = np.max(plot_data)
                    num_bands = 50
                    cmap = plot_config["cmap"]

                    for i in range(num_bands):
                        fraction = i / num_bands
                        scale = 1.0 - fraction
                        y_pos = plot_data * scale
                        y_neg = -plot_data * scale
                        gradient = cmap(fraction)
                        ax.fill_between(
                            dist_array,
                            y_pos,
                            y_neg,
                            color=gradient,
                        )

                    ax.plot(dist_array, plot_data)
                    ax.plot(dist_array, -plot_data)

                    ax.set_ylim(-1.1 * y_max, 1.1 * y_max)
                    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
                else:
                    ax.plot(dist_array, plot_data)

                ax.set(
                    xlabel=plot_config["labels"]["x_z"],
                    ylabel=plot_config["labels"]["y_z"],
                )
                ax.set_title(plot_config["titles"]["z"])

                filename = f"1d_{plot_type}_z.png"
                self.save_or_display(fig, filename, save_path, plot_config["dpi"])


class Plot2D(BasePlot):
    """Plotting class for 2D (colormap) plots."""

    def render_2d_data(
        self,
        data,
        k_array=None,
        z_coor=None,
        plot_type="intensity",
        resolution="medium",
        save_path=None,
        stride_pair=None,
        log_scale=False,
    ):
        """
        Create 2D (colormap) plots for different coordinate systems.

        Arguments:
            data: Dictionary containing the datasets for different coordinates.
            k_array: List of z indices to plot (for rt plots).
            z_coor: List of z coordinates corresponding to the k indices saved.
            plot_type: "intensity" or "density".
            resolution: Plot quality (low, medium, high).
            stride_pair: Tuple specifying the stride for mesh plotting (faster rendering).
            save_path: Path to save figures instead of displaying them.
            log_scale: Whether to use logarithmic scale.
        """
        # Configuration for different plot types
        plot_config = self.config.get_plot_config(plot_type, "2d")

        dimension_config = self.config.get_plot_config(plot_type, "all")
        resolution_config = dimension_config.get("2d", {}).get("resolutions", {})
        resolution_opt = resolution_config.get(
            resolution, resolution_config.get("medium", {})
        )
        stride = stride_pair or resolution_opt.get("stride", (1, 1))

        # Plot each coordinate system in a separate figure
        for coord_sys, plot_data in data.items():
            if coord_sys == "rt" and k_array is not None:
                # Plot intensity or density for each z position
                for idx in range(len(k_array)):
                    fig, ax = plt.subplots()

                    # Get the meshgrid
                    x, y = self.get_2d_grid("rt")
                    x_strided = x[:: stride[0], :: stride[1]]
                    y_strided = y[:: stride[0], :: stride[1]]
                    data_strided = plot_data[:, idx, :][:: stride[0], :: stride[1]]
                    xlabel = plot_config["labels"]["x_r"]
                    ylabel = plot_config["labels"]["x_t"]

                    # Plot in logarithmic scale if requested
                    if log_scale and plot_type == "intensity":
                        mesh = ax.pcolormesh(
                            x_strided,
                            y_strided,
                            data_strided,
                            cmap=plot_config["cmap"],
                            norm=LogNorm(
                                vmin=data_strided.min(),
                                vmax=data_strided.max(),
                            ),
                        )
                        colorbar_label = plot_config["colorbar_label"] + " (log scale)"
                    else:
                        # Apply stride to mesh plotting
                        mesh = ax.pcolormesh(
                            x_strided,
                            y_strided,
                            data_strided,
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
                        f"2d_{plot_type}_{coord_sys}_{z_pos:.2f}".replace(".", "-")
                        + ".png"
                    )
                    self.save_or_display(
                        fig, filename, save_path, resolution_opt["dpi"]
                    )

            else:
                fig, ax = plt.subplots()

                if coord_sys == "zt":
                    # Plot intensity or density on-axis
                    x, y = self.get_2d_grid("zt")
                    x_strided = x[:: stride[0], :: stride[1]]
                    y_strided = y[:: stride[0], :: stride[1]]
                    data_strided = plot_data[:: stride[0], :: stride[1]]
                    xlabel = plot_config["labels"]["x_z"]
                    ylabel = plot_config["labels"]["x_t"]
                elif coord_sys == "rz":
                    # Plot intensity or density peak values
                    # or fluence distribution
                    x, y = self.get_2d_grid("rz")
                    x_strided = x[:: stride[0], :: stride[1]]
                    y_strided = y[:: stride[0], :: stride[1]]
                    data_strided = plot_data[:: stride[0], :: stride[1]]
                    xlabel = plot_config["labels"]["x_r"]
                    ylabel = plot_config["labels"]["x_z"]

                mesh = ax.pcolormesh(
                    x_strided, y_strided, data_strided, cmap=plot_config["cmap"]
                )
                fig.colorbar(mesh, ax=ax, label=plot_config["colorbar_label"])
                ax.set(xlabel=xlabel, ylabel=ylabel)
                ax.set_title(plot_config["titles"][coord_sys])

                filename = f"2d_{plot_type}_{coord_sys}.png"
                self.save_or_display(fig, filename, save_path, resolution_opt["dpi"])


class Plot3D(BasePlot):
    """Plotting class for 3D (surface) plots."""

    def render_3d_data(
        self,
        data,
        k_array,
        z_coor=None,
        plot_type="intensity",
        resolution="medium",
        save_path=None,
        stride_pair=None,
    ):
        """
        Create 3D (surface) plots for different coordinate systems.

        Arguments:
            data: Dictionary containing the datasets for different coordinates.
            k_array: List of z indices to plot (for rt plots).
            z_coor: List of z coordinates corresponding to the k indices saved.
            plot_type: "intensity" or "density".
            resolution: Plot quality (low, medium, high).
            stride: Tuple specifying the stride for mesh plotting (faster rendering).
            save_path: Path to save figures instead of displaying them.
        """
        # Set configuration
        plot_config = self.config.get_plot_config(plot_type, "3d")
        dimension_config = self.config.get_plot_config(plot_type, "all")
        resolution_config = dimension_config.get("3d", {}).get("resolutions", {})
        camera_angles = plot_config.get("camera_angles", {})
        resolution_opt = resolution_config.get(
            resolution, resolution_config.get("medium", {})
        )
        stride = stride_pair or resolution_opt.get("stride", (1, 1))

        # Disable interactive mode if saving
        if save_path and plt.isinteractive():
            plt.ioff()

        for coord_sys, plot_data in data.items():
            camera_curr = camera_angles[coord_sys]
            if coord_sys == "rt" and k_array is not None:
                # Plot intensity or density for each z position
                for idx in range(len(k_array)):
                    fig = plt.figure(dpi=resolution_opt["dpi"])
                    ax = fig.add_subplot(projection="3d")

                    # Get the meshgrid
                    x, y = self.get_2d_grid("rt")
                    x_strided = x[:: stride[0], :: stride[1]]
                    y_strided = y[:: stride[0], :: stride[1]]
                    data_strided = plot_data[:, idx, :][:: stride[0], :: stride[1]]
                    xlabel = plot_config["labels"]["x_r"]
                    ylabel = plot_config["labels"]["x_t"]

                    ax.plot_surface(
                        x_strided,
                        y_strided,
                        data_strided,
                        cmap=plot_config["cmap"],
                        linewidth=0,
                        antialiased=resolution_opt["antialiased"],
                    )
                    ax.view_init(
                        elev=camera_curr["elevation"], azim=camera_curr["azimuth"]
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
                        fig, filename, save_path, resolution_opt["dpi"]
                    )

            else:
                fig = plt.figure(dpi=resolution_opt["dpi"])
                ax = fig.add_subplot(projection="3d")

                if coord_sys == "zt":
                    # Plot intensity or density on-axis
                    x, y = self.get_2d_grid("zt")
                    x_strided = x[:: stride[0], :: stride[1]]
                    y_strided = y[:: stride[0], :: stride[1]]
                    data_strided = plot_data[:: stride[0], :: stride[1]]
                    xlabel = plot_config["labels"]["x_z"]
                    ylabel = plot_config["labels"]["x_t"]
                else:
                    # Plot intensity or density peak value
                    x, y = self.get_2d_grid("rz")
                    x_strided = x[:: stride[0], :: stride[1]]
                    y_strided = y[:: stride[0], :: stride[1]]
                    data_strided = plot_data[:: stride[0], :: stride[1]]
                    xlabel = plot_config["labels"]["x_r"]
                    ylabel = plot_config["labels"]["x_z"]

                # Apply stride for better performance
                ax.plot_surface(
                    x_strided,
                    y_strided,
                    data_strided,
                    cmap=plot_config["cmap"],
                    linewidth=0,
                    antialiased=resolution_opt["antialiased"],
                )
                ax.view_init(elev=camera_curr["elevation"], azim=camera_curr["azimuth"])
                # fig.colorbar(surf, label=plot_config["colorbar_label"])
                ax.set(
                    xlabel=xlabel,
                    ylabel=ylabel,
                    zlabel=plot_config["colorbar_label"],
                )
                ax.set_title(plot_config["titles"][coord_sys])

                filename = f"3d_{plot_type}_{coord_sys}.png"
                self.save_or_display(fig, filename, save_path, resolution_opt["dpi"])


class VisualManager:
    """Manages all plotting classes."""

    def __init__(self, units, box, config, box_units):
        self.units = units
        self.box = box
        self.config = config
        self.box_units = box_units

        # Initialize specialized plotters
        self.base_plot = BasePlot(units, box, config, box_units)
        self.plot_1d = Plot1D(units, box, config, box_units)
        self.plot_2d = Plot2D(units, box, config, box_units)
        self.plot_3d = Plot3D(units, box, config, box_units)

    def get_intensity_data(self, envelope_dist, envelope_axis, envelope_peak):
        """Calculate intensity data."""
        return self.base_plot.calculate_intensity(
            envelope_dist, envelope_axis, envelope_peak
        )

    def get_density_data(self, density_dist, density_axis, density_peak):
        """Calculate density data."""
        return self.base_plot.calculate_density(
            density_dist, density_axis, density_peak
        )

    def get_fluence_data(self, b_fluence):
        """Calculate fluence data."""
        return self.base_plot.calculate_fluence(b_fluence)

    def get_radius_data(self, b_radius):
        """Calculate beam radius data."""
        return self.base_plot.calculate_radius(b_radius)

    def create_1d_plot(
        self, data, k_array=None, z_coor=None, plot_type="intensity", save_path=None
    ):
        """Create line plots."""
        self.plot_1d.render_1d_data(data, k_array, z_coor, plot_type, save_path)

    def create_2d_plot(
        self,
        data,
        k_array=None,
        z_coor=None,
        plot_type="intensity",
        resolution="medium",
        save_path=None,
        stride=None,
        log_scale=False,
    ):
        """Create colormap plots."""
        self.plot_2d.render_2d_data(
            data, k_array, z_coor, plot_type, resolution, save_path, stride, log_scale
        )

    def create_3d_plot(
        self,
        data,
        k_array,
        z_coor=None,
        plot_type="intensity",
        resolution="medium",
        save_path=None,
        stride=None,
    ):
        """Create 3D solution plots."""
        self.plot_3d.render_3d_data(
            data, k_array, z_coor, plot_type, resolution, save_path, stride
        )


def parse_cli_options():
    """Parse and validate CLI options."""
    parser = argparse.ArgumentParser(
        description="Plot simulation data from HDF5 or NPZ files.",
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
        default=sim_dir,
        help="Path to data file (.npz format).",
    )
    parser.add_argument(
        "--save-path",
        default=fig_dir,
        help="Directory to save plots instead of displaying.",
    )
    parser.add_argument(
        "--data-types",
        default="intensity,density,fluence,radius",
        help="Data types to plot: intensity,density,fluence,radius (comma-separated).",
    )
    parser.add_argument(
        "--plot-types",
        default="1d,2d,3d",
        help="Plot types to generate: 1d,2d,3d (comma-separated).",
    )
    parser.add_argument(
        "--resolution",
        default="medium",
        help="Plot quality for 3D plots: low, medium, high.",
    )
    parser.add_argument(
        "--stride",
        default="1,1",
        help="Data stride (x,y format) for plotting 2D and 3D plots.",
    )
    parser.add_argument(
        "--radial-limit",
        type=float,
        default=None,
        help="Maximum value for radial grid (in meters).",
    )
    parser.add_argument(
        "--axial-range",
        default=None,
        help="Axial grid min,max values (in meters).",
    )
    parser.add_argument(
        "--time-range",
        default=None,
        help="Time grid min,max values (in seconds).",
    )
    parser.add_argument(
        "--radial-symmetry",
        action="store_true",
        default=False,
        help="Plot every radial axis symmetrically.",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        default=False,
        help="Use logarithmic scale for intensity colormaps.",
    )

    args = parser.parse_args()

    if args.axial_range:
        try:
            z_min, z_max = map(float, args.axial_range.split(","))
            args.axial_range = (z_min, z_max)
        except ValueError:
            print(
                "Error: Axial range format must be in format 'min,max'. Using full range."
            )
            args.axial_range = None

    if args.time_range:
        try:
            t_min, t_max = map(float, args.time_range.split(","))
            args.time_range = (t_min, t_max)
        except ValueError:
            print(
                "Error: Time range format must be in format 'min,max'. Using full range."
            )
            args.time_range = None

    # Convert comma-separated strings to dictionaries for easier access
    stride_pair = [int(s) for s in args.stride.split(",")]
    args.data_types = {dtype: True for dtype in args.data_types.split(",")}
    args.plot_types = {ptype: True for ptype in args.plot_types.split(",")}
    args.stride = (stride_pair[0], stride_pair[1])

    return args


def setup_output_directory(args):
    """Setup environment for plotting based on arguments."""
    # Ask if we have a save path when required
    if args.save_path:
        save_path = args.save_path
        save_path.mkdir(parents=True, exist_ok=True)
        print(f"Saving plots to: {save_path.relative_to(base_dir)}")
    else:
        print("Displaying plots interactively.")


def load_simulation_data(base_file_path, args):
    """Load data from HDF5 or NPZ files."""
    snapshots_path = base_file_path / "snapshots.h5"
    diagnostic_path = base_file_path / "final_diagnostic.h5"
    npz_path = base_file_path / "data.npz"

    data = {}

    has_snapshots = snapshots_path.exists()
    has_final_diag = diagnostic_path.exists()

    if has_snapshots and has_final_diag:
        print(
            f"Loading data from HDF5 files: {snapshots_path.relative_to(base_dir)}"
            f"and {diagnostic_path.relative_to(base_dir)} ..."
        )

        try:
            print(f"Using snapshots file: {snapshots_path.name}")

            # Load the data from snapshot file
            with h5py.File(snapshots_path, "r") as f:
                data["k_array"] = np.array(f["snap_z_idx"])
                if "envelope_snapshot_rzt" in f:
                    data["e_dist"] = np.array(f["envelope_snapshot_rzt"])
                if "density_snapshot_rzt" in f:
                    data["elec_dist"] = np.array(f["density_snapshot_rzt"])

            print(f"Using diagnostic file: {diagnostic_path.name}")

            # Load the data from diagnostic file
            with h5py.File(diagnostic_path, "r") as f:
                coords = f["coordinates"]
                data["ini_radi_coor"] = coords["r_min"][()]
                data["fin_radi_coor"] = coords["r_max"][()]
                data["ini_dist_coor"] = coords["z_min"][()]
                data["fin_dist_coor"] = coords["z_max"][()]
                data["ini_time_coor"] = coords["t_min"][()]
                data["fin_time_coor"] = coords["t_max"][()]

                if "envelope" in f:
                    envelope = f["envelope"]
                    if "axis_zt" in envelope:
                        data["e_axis"] = np.array(envelope["axis_zt"])
                    if "peak_rz" in envelope:
                        data["e_peak"] = np.array(envelope["peak_rz"])

                if "density" in f:
                    density = f["density"]
                    if "axis_zt" in density:
                        data["elec_axis"] = np.array(density["axis_zt"])
                    if "peak_rz" in density:
                        data["elec_peak"] = np.array(density["peak_rz"])

                if "pulse" in f:
                    pulse = f["pulse"]
                    if "fluence_rz" in pulse:
                        data["b_fluence"] = np.array(pulse["fluence_rz"])
                    if "radius_z" in pulse:
                        data["b_radius"] = np.array(pulse["radius_z"])

            return data

        except Exception as e:
            print(f"Error loading HDF5 data: {type(e).__name__}: {e}")
            if args.verbose > 0:
                traceback.print_exc()
            else:
                print("Run with -v for more information")
            raise

    elif npz_path.exists():
        print(f"Loading data from NPZ file: {npz_path.relative_to(base_dir)} ...")
        try:
            npz_size = npz_path.stat().st_size / (1024**2)  # in MB

            # Load the data from file using memory mapping
            if npz_size > 500:
                print("Using memory-mapped file loading")
                with np.load(npz_path, mmap_mode="r") as npz:
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
                        "b_fluence",
                        "b_radius",
                    ]:
                        if key in npz.files:
                            data[key] = npz[key]  # Memory-mapped reference
            else:
                print("Loading full data file")
                data = dict(np.load(npz_path))

            return data

        except Exception as e:
            print(f"Error loading NPZ file: {type(e).__name__}: {e}")
            if args.verbose > 0:
                traceback.print_exc()
            else:
                print("Run with -v for more information")
            raise

    else:
        raise FileNotFoundError(f"No HDF5 or NPZ files found at {base_file_path}")


def process_plot_request(
    data_type, plot_data, plot_types, plot, z_snap_idx, z_snap_coor, args
):
    """Generate requested plot types for the specified data."""

    if plot_types.get("1d", False):
        print(f"Generating 1D (line) plots for {data_type} ...")
        plot.create_1d_plot(
            plot_data,
            z_snap_idx,
            z_snap_coor,
            data_type,
            args.save_path,
        )

    if plot_types.get("2d", False) and data_type != "radius":
        print(f"Generating 2D (colormap) plots for {data_type} ...")
        plot.create_2d_plot(
            plot_data,
            z_snap_idx,
            z_snap_coor,
            data_type,
            args.resolution,
            args.save_path,
            args.stride,
            args.log_scale,
        )

    if plot_types.get("3d", False) and data_type != "radius":
        print(f"Generating 3D (surface) plots for {data_type} ...")
        try:
            plot.create_3d_plot(
                plot_data,
                z_snap_idx,
                z_snap_coor,
                data_type,
                resolution=args.resolution,
                save_path=args.save_path,
                stride=args.stride,
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

    # Ask if the data arrays exist
    required_arrays = []
    if data_type == "intensity":
        required_arrays = ["e_dist", "e_axis", "e_peak"]
    elif data_type == "density":
        required_arrays = ["elec_dist", "elec_axis", "elec_peak"]
    elif data_type == "fluence":
        required_arrays = ["b_fluence"]
    elif data_type == "radius":
        required_arrays = ["b_radius"]

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
            elif key.endswith("_peak") or key.endswith("_fluence"):
                expected_shape = (box.nodes_r, box.nodes_z)
            elif key.endswith("_radius"):
                expected_shape = (box.nodes_z,)

            if expected_shape and data[key].shape != expected_shape:
                print(
                    f"Warning: {key} has unexpected shape: {data[key].shape}. "
                    f"expected {expected_shape}. "
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
        plot_data = {
            "rt": plot_data_dist,
            "zt": plot_data_axis,
            "rz": plot_data_peak,
        }
    elif data_type == "density":
        plot_data_dist, plot_data_axis, plot_data_peak = plot.get_density_data(
            box.sliced_data["elec_dist"],
            box.sliced_data["elec_axis"],
            box.sliced_data["elec_peak"],
        )
        plot_data = {
            "rt": plot_data_dist,
            "zt": plot_data_axis,
            "rz": plot_data_peak,
        }
    elif data_type == "fluence":
        plot_data_fluence = plot.get_fluence_data(box.sliced_data["b_fluence"])
        plot_data = {
            "rz": plot_data_fluence,
        }
    elif data_type == "radius":
        plot_data_radius = plot.get_radius_data(box.sliced_data["b_radius"])
        plot_data = {
            "z": plot_data_radius,
        }
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

    # Generate requested plot types
    process_plot_request(
        data_type, plot_data, plot_types, plot, z_snap_idx, z_snap_coor, args
    )


def main():
    """Main execution function."""
    print(f"Running HALA v{__version__} plotter")

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
            f"   Data dimensions: {data['e_axis'].shape if 'e_axis' in data else 'unknown'}"
        )
        print(
            f"   Duplicate radius: {'enabled' if args.radial_symmetry else 'disabled'}"
        )

    # Initialize classes
    units = Units()
    config = PlotConfiguration()
    box = SimulationBox(
        units,
        data,
        radial_symmetry=args.radial_symmetry,
        radial_limit=args.radial_limit,
        axial_range=args.axial_range,
        time_range=args.time_range,
    )
    box_units = SimulationBoxUnits(units, box, config)
    plot = VisualManager(units, box, config, box_units)

    # Process each requested data type
    for data_type, enabled in args.data_types.items():
        if enabled and data_type in ["intensity", "density", "fluence", "radius"]:
            process_simulation_data(data_type, data, plot, box, args.plot_types, args)

    print("Plotting complete!")


if __name__ == "__main__":
    main()
