"""
Python tool for plotting NumPy arrays saved after
the simulations have finished execution.
"""

import argparse
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from h5py import File
from matplotlib.colors import LogNorm

from .data.paths import get_base_dir, get_user_paths

AVAILABLE_1D_COLORS = {
    "blue_dark": "#1E90FF",
    "green_dark": "#32CD32",
    "magenta_dark": "#FF00FF",
    "yellow_dark": "#FFFF00",
    "blue_white": "#0066CC",
    "green_white": "#007F00",
    "magenta_white": "#CC00CC",
    "yellow_white": "#CC9900",
}

AVAILABLE_2D_COLORS = {
    "viridis": mpl.colormaps["viridis"],
    "plasma": mpl.colormaps["plasma"],
    "inferno": mpl.colormaps["inferno"],
    "magma": mpl.colormaps["magma"],
    "cividis": mpl.colormaps["cividis"],
    "jet": mpl.colormaps["jet"],
    "rdbu": mpl.colormaps["RdBu"],
    "piyg": mpl.colormaps["PiYG"],
    "spectral": mpl.colormaps["Spectral"],
}

SUPPORTED_VARIABLES = ("intensity", "density", "fluence", "radius")


@dataclass
class PlotConfiguration:
    """Plot style configuration."""

    default_1d_color: str = "#1E90FF"
    default_2d_color: str = "viridis"

    colors1d: Dict[str, str] = field(default_factory=AVAILABLE_1D_COLORS.copy)
    colors2d: Dict[str, Any] = field(default_factory=AVAILABLE_2D_COLORS.copy)

    def get_plot_config(self, plot_type: str, dimension: str = "all") -> Dict:
        """
        Return configuration for specified plot type and dimension.

        Args:
        - plot_type: Physical magnitude ("intensity", "density", "fluence", or "radius")
        - dimension: Plot dimension ("1d", "2d", "3d", or "all" for all dimensions)

        Returns:
        - Dictionary with predefined configuration settings for plotting.
        """
        # Physical magnitude configuration options
        base_configuration = {
            "intensity": {
                "cmap": self.colors2d[self.default_2d_color],
                "line_color": self.default_1d_color,
                "bar_label": r"Intensity $[W/cm^2]$",
                "title": {
                    "z": "On-axis peak intensity over time",
                    "t": r"On-axis intensity at $z = {:.2f}$ m",
                    "zt": "On-axis intensity",
                    "rt": r"Intensity at $z = {:.2f}$ m",
                    "rz": "Peak intensity over time",
                },
                "label": {
                    "xr": r"$r$ [m]",
                    "xz": r"$z$ [m]",
                    "xt": r"$t$ [s]",
                    "yt": r"$I(r=0,t)$ $[W/cm^2]$",
                    "yz": r"$\max_t\, I(r=0,z,t)$ $[W/cm^2]$",
                },
            },
            "density": {
                "cmap": self.colors2d[self.default_2d_color],
                "line_color": self.default_1d_color,
                "bar_label": r"Electron density $[cm^{-3}]$",
                "title": {
                    "z": "On-axis peak electron density over time",
                    "t": r"On-axis electron density at $z = {:.2f}$ m",
                    "zt": "On-axis electron density",
                    "rt": r"Electron density at $z = {:.2f}$ m",
                    "rz": "Peak electron density over time",
                },
                "label": {
                    "xr": r"$r$ [m]",
                    "xz": r"$z$ [m]",
                    "xt": r"$t$ [s]",
                    "yt": r"$\rho_e(r=0,t)$ $[cm^{-3}]$",
                    "yz": r"$\max_t\, \rho_e(r=0,z,t)$ $[cm^{-3}]$",
                },
            },
            "fluence": {
                "cmap": self.colors2d[self.default_2d_color],
                "line_color": self.default_1d_color,
                "bar_label": r"Fluence $[J/cm^2]$",
                "title": {
                    "z": "On-axis fluence distribution",
                    "rz": "Fluence distribution",
                },
                "label": {
                    "xr": r"$r$ [m]",
                    "xz": r"$z$ [m]",
                    "yz": r"$F(r=0,z)$ $[J/cm^2]$",
                },
            },
            "radius": {
                "cmap": self.colors2d[self.default_2d_color],
                "line_color": self.default_1d_color,
                "title": {"z": "Beam radius"},
                "label": {
                    "xz": r"$z$ [m]",
                    "yz": r"$R(z)$ [m]",
                },
            },
        }

        # Dimension configuration options
        dimension_configuration = {
            "1d": {"dpi": 150},
            "2d": {
                "resolution": {
                    "low": {"stride": (5, 5), "dpi": 100},
                    "medium": {"stride": (2, 2), "dpi": 150},
                    "high": {"stride": (1, 1), "dpi": 300},
                },
            },
            "3d": {
                "resolution": {
                    "low": {"stride": (5, 5), "dpi": 100},
                    "medium": {"stride": (2, 2), "dpi": 150},
                    "high": {"stride": (1, 1), "dpi": 300},
                },
                "camera_angle": {
                    "rt": {"elevation": 15, "azimuth": 200},
                    "zt": {"elevation": 10, "azimuth": 235},
                    "rz": {"elevation": 10, "azimuth": 325},
                },
            },
        }

        if dimension == "all":
            return {
                "base": base_configuration[plot_type],
                **dimension_configuration,
            }

        return {
            **base_configuration[plot_type],
            **dimension_configuration.get(dimension, {}),
        }


class Units:
    """Unit factors for converting coordinate and magnitude arrays."""

    def __init__(
        self,
        factor_r=1,
        factor_z=1,
        factor_t=1,
        factor_m2=1e-4,
        factor_m3=1e-6,
        factor_j=1,
    ):

        self.fr = factor_r
        self.fz = factor_z
        self.ft = factor_t
        self.fa = factor_m2
        self.fv = factor_m3
        self.fj = factor_j


class SimulationBox:
    """Simulation plotting domain with optional slicing constraints."""

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
        self.r_sym = radial_symmetry
        self.r_limit = radial_limit
        self.z_range = axial_range
        self.t_range = time_range
        self.init_boundaries()
        self.init_grid_nodes()
        self.init_sliced_arrays()

    def init_boundaries(self):
        """Set up the plotting box boundary."""
        self.r_min_ori = self.data["ini_radi_coor"]
        self.r_max_ori = self.data["fin_radi_coor"]
        self.z_min_ori = self.data["ini_dist_coor"]
        self.z_max_ori = self.data["fin_dist_coor"]
        self.t_min_ori = self.data["ini_time_coor"]
        self.t_max_ori = self.data["fin_time_coor"]

        r_min = self.r_min_ori
        r_max = self.r_max_ori
        z_min = self.z_min_ori
        z_max = self.z_max_ori
        t_min = self.t_min_ori
        t_max = self.t_max_ori

        if self.r_limit is not None:
            r_max = min(r_max, self.r_limit)

        if self.z_range is not None:
            z_min_new, z_max_new = self.z_range
            z_min = max(z_min, z_min_new)
            z_max = min(z_max, z_max_new)

        if self.t_range is not None:
            t_min_new, t_max_new = self.t_range
            t_min = max(t_min, t_min_new)
            t_max = min(t_max, t_max_new)

        if self.r_sym:
            r_min = -r_max

        self.b_r = (r_min, r_max)
        self.b_z = (z_min, z_max)
        self.b_t = (t_min, t_max)

    def init_grid_nodes(self):
        """Set up the plotting box boundary nodes."""
        self.nr = self.data["e_dist"].shape[0]
        self.nz = self.data["e_axis"].shape[0]
        self.nt = self.data["e_axis"].shape[1]

        if self.r_sym:
            dr_ori = self.r_max_ori - self.r_min_ori
            self.nr_sym = 2 * self.nr - 1
            self.nr_0 = int(-self.b_r[0] * (self.nr - 1) / dr_ori)
        else:
            self.nr_0 = 0

        self.nodes = {}
        for dim, (min_b, max_b, n_nodes, min_o, max_o) in {
            "r_data": (
                *self.b_r,
                self.nr if not self.r_sym else self.nr_sym,
                (self.r_min_ori if not self.r_sym else -self.r_max_ori),
                self.r_max_ori,
            ),
            "z_data": (
                *self.b_z,
                self.nz,
                self.z_min_ori,
                self.z_max_ori,
            ),
            "t_data": (
                *self.b_t,
                self.nt,
                self.t_min_ori,
                self.t_max_ori,
            ),
        }.items():
            n_min = (min_b - min_o) * (n_nodes - 1) / (max_o - min_o)
            n_max = (max_b - min_o) * (n_nodes - 1) / (max_o - min_o)
            self.nodes[dim] = (int(n_min), int(n_max + 1))

    def init_sliced_arrays(self):
        """Create sliced grids and sliced data arrays used for plotting."""
        self.sliced_data = {}
        self.sliced_coor = {  # Get elements from n_min to n_max
            "r": slice(*self.nodes["r_data"]),
            "z": slice(*self.nodes["z_data"]),
            "t": slice(*self.nodes["t_data"]),
        }

        if self.r_sym:
            radius_p = np.linspace(0, self.r_max_ori, self.nr)
            radius_n = -np.flip(radius_p[1:])
            radius = np.concatenate((radius_n, radius_p))
            radius_slice = radius[self.sliced_coor["r"]]
        else:
            radius_slice = np.linspace(self.r_min_ori, self.r_max_ori, self.nr)[
                self.sliced_coor["r"]
            ]

        # Create sliced grids
        self.sliced_grids = {
            "r": radius_slice,
            "z": np.linspace(self.z_min_ori, self.z_max_ori, self.nz)[
                self.sliced_coor["z"]
            ],
            "t": np.linspace(self.t_min_ori, self.t_max_ori, self.nt)[
                self.sliced_coor["t"]
            ],
        }

        # Slice electric field data if present
        if "e_dist" in self.data:
            if self.r_sym:
                self.sliced_data["e_dist"] = self.flip_radial_data(
                    self.data["e_dist"], axis_r=0
                )[self.sliced_coor["r"], :, self.sliced_coor["t"]]
            else:
                self.sliced_data["e_dist"] = self.data["e_dist"][
                    self.sliced_coor["r"], :, self.sliced_coor["t"]
                ]
        if "e_axis" in self.data:
            self.sliced_data["e_axis"] = self.data["e_axis"][
                self.sliced_coor["z"], self.sliced_coor["t"]
            ]
        if "e_peak" in self.data:
            if self.r_sym:
                self.sliced_data["e_peak"] = self.flip_radial_data(
                    self.data["e_peak"], axis_r=0
                )[self.sliced_coor["r"], self.sliced_coor["z"]]
            else:
                self.sliced_data["e_peak"] = self.data["e_peak"][
                    self.sliced_coor["r"], self.sliced_coor["z"]
                ]

        # Slice electron density data if present
        if "elec_dist" in self.data:
            if self.r_sym:
                self.sliced_data["elec_dist"] = self.flip_radial_data(
                    self.data["elec_dist"], axis_r=0
                )[self.sliced_coor["r"], :, self.sliced_coor["t"]]
            else:
                self.sliced_data["elec_dist"] = self.data["elec_dist"][
                    self.sliced_coor["r"], :, self.sliced_coor["t"]
                ]
        if "elec_axis" in self.data:
            self.sliced_data["elec_axis"] = self.data["elec_axis"][
                self.sliced_coor["z"], self.sliced_coor["t"]
            ]
        if "elec_peak" in self.data:
            if self.r_sym:
                self.sliced_data["elec_peak"] = self.flip_radial_data(
                    self.data["elec_peak"], axis_r=0
                )[self.sliced_coor["r"], self.sliced_coor["z"]]
            else:
                self.sliced_data["elec_peak"] = self.data["elec_peak"][
                    self.sliced_coor["r"], self.sliced_coor["z"]
                ]

        # Slice beam fluence distribution data if present
        if "b_fluence" in self.data:
            if self.r_sym:
                self.sliced_data["b_fluence"] = self.flip_radial_data(
                    self.data["b_fluence"], axis_r=0
                )[self.sliced_coor["r"], self.sliced_coor["z"]]
            else:
                self.sliced_data["b_fluence"] = self.data["b_fluence"][
                    self.sliced_coor["r"], self.sliced_coor["z"]
                ]

    def set_snapshot_points(self, indices):
        """Convert k-indices to their corresponding z-coordinates."""
        z_min = self.data["ini_dist_coor"] * self.units.fz
        z_max = self.data["fin_dist_coor"] * self.units.fz
        z_snap_coor = z_min + (indices * (z_max - z_min) / (self.nz - 1))
        return z_snap_coor

    def flip_radial_data(self, data, axis_r=0):
        """Mirror radial data for symmetry."""
        if not self.r_sym:
            return data

        flipped_data = np.flip(data[1:], axis=axis_r)
        return np.concatenate((flipped_data, data), axis=axis_r)


class SimulationBoxUnits:
    """Cached unit-scaled plotting grids."""

    def __init__(self, units: Units, box: SimulationBox, config: PlotConfiguration):
        self.units = units
        self.box = box
        self.config = config
        self.scaled_1d_grid = {}
        self.scaled_2d_grid = {}

    def create_unit_scaled_1d_grid(self, grid_type):
        """Get a scaled array, creating it if necessary."""
        if grid_type not in self.scaled_1d_grid:
            if grid_type.startswith("r"):
                self.scaled_1d_grid[grid_type] = (
                    self.units.fr * self.box.sliced_grids[grid_type]
                )
            elif grid_type.startswith("z"):
                self.scaled_1d_grid[grid_type] = (
                    self.units.fz * self.box.sliced_grids[grid_type]
                )
            elif grid_type.startswith("t"):
                self.scaled_1d_grid[grid_type] = (
                    self.units.ft * self.box.sliced_grids[grid_type]
                )

        return self.scaled_1d_grid[grid_type]

    def create_unit_scaled_2d_grid(self, grid_type):
        """Set up meshgrids only when needed."""
        if grid_type not in self.scaled_2d_grid:
            if grid_type == "zt":
                self.scaled_2d_grid[grid_type] = np.meshgrid(
                    self.create_unit_scaled_1d_grid("z"),
                    self.create_unit_scaled_1d_grid("t"),
                    indexing="ij",
                )
            elif grid_type == "rt":
                self.scaled_2d_grid[grid_type] = np.meshgrid(
                    self.create_unit_scaled_1d_grid("r"),
                    self.create_unit_scaled_1d_grid("t"),
                    indexing="ij",
                )
            elif grid_type == "rz":
                self.scaled_2d_grid[grid_type] = np.meshgrid(
                    self.create_unit_scaled_1d_grid("r"),
                    self.create_unit_scaled_1d_grid("z"),
                    indexing="ij",
                )

        return self.scaled_2d_grid[grid_type]


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

    def compute_intensity(self, envelope_dist, envelope_axis, envelope_peak):
        """Set up intensities for plotting."""
        return (
            self.units.fa * np.abs(envelope_dist) ** 2,
            self.units.fa * np.abs(envelope_axis) ** 2,
            self.units.fa * np.abs(envelope_peak) ** 2,
        )

    def compute_density(self, density_dist, density_axis, density_peak):
        """Set up densities for plotting."""
        return (
            self.units.fv * density_dist,
            self.units.fv * density_axis,
            self.units.fv * density_peak,
        )

    def compute_fluence(self, b_fluence):
        """Set up fluence distribution for plotting."""
        return self.units.fa * self.units.fj * b_fluence

    def compute_radius(self, b_fluence, radial_grid):
        """Set up beam radius for plotting."""
        fluence = b_fluence
        r_grid = radial_grid

        if self.box.r_sym:
            radius_p = r_grid >= 0
            fluence = fluence[radius_p, :]
            r_grid = r_grid[radius_p]

        nz = fluence.shape[1]
        b_radius = np.full(nz, np.nan, dtype=np.float64)

        for z_idx in range(nz):
            fluence_z = fluence[:, z_idx]
            peak_idx = np.argmax(fluence_z)
            half_max = 0.5 * fluence_z[peak_idx]

            if half_max <= 0 or not np.isfinite(half_max):
                continue

            f_slice = fluence_z[peak_idx:]
            r_slice = r_grid[peak_idx:]

            indices = (f_slice[:-1] >= half_max) & (f_slice[1:] < half_max)
            if not np.any(indices):
                b_radius[z_idx] = r_slice[-1] if f_slice[-1] >= half_max else np.nan
                continue

            idx = np.argmax(indices)
            f0, f1 = f_slice[idx], f_slice[idx + 1]
            r0, r1 = r_slice[idx], r_slice[idx + 1]
            slope = (half_max - f0) / (f1 - f0)
            radius = r0 + slope * (r1 - r0)

            b_radius[z_idx] = radius

        return self.units.fr * b_radius

    def save_or_display(self, fig, filename, fig_path, dpi=150):
        """Save figure or display it."""
        fig.tight_layout()
        if fig_path:
            fig_path = Path(fig_path)
            filepath = fig_path / filename
            fig.savefig(filepath, dpi=dpi)
            plt.close(fig)
        else:
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
        z_idx=None,
        z_coor=None,
        magnitude="intensity",
        fig_path=None,
        scale="both",
        log_y_range=None,
    ):
        """
        Create 1D (line) plots for different coordinate systems.

        Arguments:
            data: Dictionary containing the datasets for different coordinates:
            z_idx: List of z indices to plot (for rt plots).
            z_coor: List of z coordinates corresponding to z_idx.
            magnitude: "intensity", "density", "fluence" or "radius".
            fig_path: Path to save figures instead of displaying them.
        """
        plot_config = self.config.get_plot_config(magnitude, "1d")

        # Plot each coordinate system in a separate figure
        for coord_key, dataset in data.items():
            if coord_key == "rt" and z_idx is not None:
                # Plot intensity or density for each z-position
                # with respect to time
                x = self.get_1d_grid("t")
                x_label = plot_config["label"]["xt"]
                y_label = plot_config["label"]["yt"]
                for idx in range(len(z_idx)):
                    y = dataset[self.box.nr_0, idx, :]

                    fig, ax = plt.subplots()
                    ax.plot(x, y, color=plot_config["line_color"])
                    ax.set(xlabel=x_label, ylabel=y_label)

                    z_pos = z_coor[idx]
                    z_pos_format = f"{z_pos:.2f}"
                    title = plot_config["title"]["t"].replace("{:.2f}", z_pos_format)
                    ax.set_title(title)

                    filename = (
                        f"1d_{magnitude}_t_{z_pos:.2f}".replace(".", "-") + ".png"
                    )
                    self.save_or_display(fig, filename, fig_path, plot_config["dpi"])

            elif coord_key == "rz":
                # Plot intensity or density peak value on-axis
                # with respect to distance
                x = self.get_1d_grid("z")
                y = dataset[self.box.nr_0, :]
                x_label = plot_config["label"]["xz"]
                y_label = plot_config["label"]["yz"]

                if scale in ("both", "linear"):
                    fig, ax = plt.subplots()
                    ax.plot(x, y, color=plot_config["line_color"])
                    ax.set(xlabel=x_label, ylabel=y_label)
                    ax.set_title(plot_config["title"]["z"])

                    filename = f"1d_{magnitude}_z.png"
                    self.save_or_display(fig, filename, fig_path, plot_config["dpi"])

                # Save logarithmic y-axis version too
                if scale in ("both", "log"):
                    if log_y_range is None:
                        raise ValueError(
                            "Log 1D plots require --log-y-range 'min,max'."
                        )

                    y_min, y_max = log_y_range
                    y_log = np.where((y > 0) & (y >= y_min) & (y <= y_max), y, np.nan)
                    if not np.any(np.isfinite(y_log)):
                        continue

                    fig, ax = plt.subplots()
                    ax.plot(x, y_log, color=plot_config["line_color"])
                    ax.set(xlabel=x_label, ylabel=y_label)
                    ax.set_yscale("log")
                    ax.set_ylim(bottom=y_min, top=y_max)
                    ax.set_title(plot_config["title"]["z"] + " (log-scale)")

                    filename = f"1d_{magnitude}_z_log.png"
                    self.save_or_display(fig, filename, fig_path, plot_config["dpi"])

            elif coord_key == "z" and magnitude == "radius":
                # Plot beam radius with respect to distance
                x = self.get_1d_grid("z")
                y = dataset
                x_label = plot_config["label"]["xz"]
                y_label = plot_config["label"]["yz"]

                fig, ax = plt.subplots()
                if self.box.r_sym:
                    ax.plot(x, y, color=plot_config["line_color"])
                    ax.plot(x, -y, color=plot_config["line_color"])
                else:
                    ax.plot(x, y, color=plot_config["line_color"])

                ax.set(xlabel=x_label, ylabel=y_label)
                ax.set_title(plot_config["title"]["z"])

                filename = f"1d_{magnitude}_z.png"
                self.save_or_display(fig, filename, fig_path, plot_config["dpi"])


class Plot2D(BasePlot):
    """Plotting class for 2D (colormap) plots."""

    @staticmethod
    def _render_2d_plot(
        x,
        y,
        z,
        cmap,
        x_label,
        y_label,
        title,
        c_label,
        norm=None,
    ):
        """Create a pcolormesh figure and return figure/axis."""
        fig, ax = plt.subplots()
        mesh = ax.pcolormesh(x, y, z, cmap=cmap, norm=norm)
        fig.colorbar(mesh, ax=ax, label=c_label)
        ax.set(xlabel=x_label, ylabel=y_label)
        ax.set_title(title)
        return fig

    def render_2d_data(
        self,
        data,
        z_idx=None,
        z_coor=None,
        magnitude="intensity",
        quality="medium",
        fig_path=None,
        stride=None,
        scale="both",
        log_y_range=None,
        log_rt_levels=None,
    ):
        """
        Create 2D (colormap) plots for different coordinate systems.

        Arguments:
            data: Dictionary containing the datasets for different coordinates.
            z_idx: List of z indices to plot (for rt plots).
            z_coor: List of z coordinates corresponding to z_idx.
            magnitude: "intensity", "density", "fluence" or "radius".
            quality: Plot quality (low, medium, high).
            fig_path: Path to save figures instead of displaying them.
            stride: Tuple specifying the stride for mesh plotting (faster rendering).
        """
        plot_config = self.config.get_plot_config(magnitude, "2d")
        resolution_map = plot_config.get("resolution", {})
        render_settings = resolution_map.get(quality, resolution_map.get("medium", {}))
        stride_pair = stride or render_settings.get("stride", (1, 1))

        # Plot each coordinate system in a separate figure
        for coord_key, dataset in data.items():
            if coord_key == "rt" and z_idx is not None:
                # Plot intensity or density for each z position
                y, x = self.get_2d_grid("rt")
                x_st = x[:: stride_pair[0], :: stride_pair[1]]
                y_st = y[:: stride_pair[0], :: stride_pair[1]]
                x_label = plot_config["label"]["xt"]
                y_label = plot_config["label"]["xr"]
                colorbar_label = plot_config["bar_label"]
                for idx in range(len(z_idx)):
                    z_st = dataset[:, idx, :][:: stride_pair[0], :: stride_pair[1]]
                    z_pos_coord = z_coor[idx]
                    z_pos_format = f"{z_pos_coord:.2f}"
                    title = plot_config["title"][coord_key].replace(
                        "{:.2f}", z_pos_format
                    )

                    if scale in ("both", "linear"):
                        fig = self._render_2d_plot(
                            x_st,
                            y_st,
                            z_st,
                            plot_config["cmap"],
                            x_label,
                            y_label,
                            title,
                            colorbar_label,
                        )

                        filename = (
                            f"2d_{magnitude}_{coord_key}_{z_pos_coord:.2f}".replace(
                                ".", "-"
                            )
                            + ".png"
                        )
                        self.save_or_display(
                            fig, filename, fig_path, render_settings["dpi"]
                        )

                    # Logarithmic version
                    if scale in ("both", "log"):
                        if log_rt_levels is None:
                            raise ValueError(
                                "2D rt log plots require --log-rt-levels "
                                "when --scale is 'log' or 'both'."
                            )

                        z_max = np.max(z_st)
                        if not np.isfinite(z_max) or z_max <= 0:
                            continue

                        z_norm = np.ma.masked_less_equal(z_st / z_max, 0)
                        vmin = 10.0 ** (-log_rt_levels)
                        levels_per_decade = 10
                        levels = np.logspace(
                            -log_rt_levels,
                            0,
                            log_rt_levels * levels_per_decade + 1,
                        )
                        rt_cmap = plot_config["cmap"].copy()
                        rt_cmap.set_under("white")
                        rt_cmap.set_bad("white")

                        fig, ax = plt.subplots()
                        mesh = ax.contourf(
                            x_st,
                            y_st,
                            z_norm,
                            levels=levels,
                            cmap=rt_cmap,
                            norm=LogNorm(vmin=vmin, vmax=1.0),
                            extend="min",
                        )
                        fig.colorbar(
                            mesh, ax=ax, label=colorbar_label + " (normalized)"
                        )
                        ax.set(xlabel=x_label, ylabel=y_label)
                        ax.set_title(title + " (log-scale)")

                        filename = (
                            f"2d_{magnitude}_{coord_key}_{z_pos_coord:.2f}_log".replace(
                                ".", "-"
                            )
                            + ".png"
                        )
                        self.save_or_display(
                            fig, filename, fig_path, render_settings["dpi"]
                        )

            elif coord_key == "zt":
                # Plot intensity or density on-axis
                x, y = self.get_2d_grid("zt")
                x_st = x[:: stride_pair[0], :: stride_pair[1]]
                y_st = y[:: stride_pair[0], :: stride_pair[1]]
                z_st = dataset[:: stride_pair[0], :: stride_pair[1]]
                x_label = plot_config["label"]["xz"]
                y_label = plot_config["label"]["xt"]
                colorbar_label = plot_config["bar_label"]
                title = plot_config["title"][coord_key]

                fig = self._render_2d_plot(
                    x_st,
                    y_st,
                    z_st,
                    plot_config["cmap"],
                    x_label,
                    y_label,
                    title,
                    colorbar_label,
                )

                filename = f"2d_{magnitude}_{coord_key}.png"
                self.save_or_display(fig, filename, fig_path, render_settings["dpi"])

            elif coord_key == "rz":
                # Plot intensity or density peak values
                # or fluence distribution
                y, x = self.get_2d_grid("rz")
                x_st = x[:: stride_pair[0], :: stride_pair[1]]
                y_st = y[:: stride_pair[0], :: stride_pair[1]]
                z_st = dataset[:: stride_pair[0], :: stride_pair[1]]
                x_label = plot_config["label"]["xz"]
                y_label = plot_config["label"]["xr"]
                colorbar_label = plot_config["bar_label"]
                title = plot_config["title"][coord_key]

                if scale in ("both", "linear"):
                    fig = self._render_2d_plot(
                        x_st,
                        y_st,
                        z_st,
                        plot_config["cmap"],
                        x_label,
                        y_label,
                        title,
                        colorbar_label,
                    )

                    filename = f"2d_{magnitude}_{coord_key}.png"
                    self.save_or_display(
                        fig, filename, fig_path, render_settings["dpi"]
                    )

                # Logarithmic version
                if scale in ("both", "log"):
                    if log_y_range is None:
                        raise ValueError(
                            "2D rz log plots require --log-y-range 'min,max'."
                        )

                    z_min, z_max = log_y_range
                    z_log = np.ma.masked_less_equal(z_st, 0)
                    if np.ma.count(z_log) > 0:
                        z_log = np.ma.masked_where(
                            (z_log < z_min) | (z_log > z_max), z_log
                        )
                        if np.ma.count(z_log) == 0 or z_max <= z_min:
                            continue

                        fig = self._render_2d_plot(
                            x_st,
                            y_st,
                            z_log,
                            plot_config["cmap"],
                            x_label,
                            y_label,
                            title + " (log-scale)",
                            colorbar_label,
                            norm=LogNorm(vmin=z_min, vmax=z_max),
                        )

                        filename = f"2d_{magnitude}_{coord_key}_log.png"
                        self.save_or_display(
                            fig, filename, fig_path, render_settings["dpi"]
                        )


class Plot3D(BasePlot):
    """Plotting class for 3D (surface) plots."""

    @staticmethod
    def _render_surface_plot(x, y, z, cmap, dpi):
        """Create a 3D surface plot and return figure/axis."""
        fig = plt.figure(dpi=dpi)
        ax = fig.add_subplot(projection="3d")
        ax.plot_surface(x, y, z, cmap=cmap)
        return fig, ax

    def render_3d_data(
        self,
        data,
        z_idx,
        z_coor=None,
        magnitude="intensity",
        quality="medium",
        fig_path=None,
        stride=None,
    ):
        """
        Create 3D (surface) plots for different coordinate systems.

        Arguments:
            data: Dictionary containing the datasets for different coordinates.
            z_idx: List of z indices to plot (for rt plots).
            z_coor: List of z coordinates corresponding to the z indices saved.
            magnitude: "intensity", "density", "fluence" or "radius".
            quality: Plot quality (low, medium, high).
            stride: Tuple specifying the stride for mesh plotting (faster rendering).
            fig_path: Path to save figures instead of displaying them.
        """
        plot_config = self.config.get_plot_config(magnitude, "3d")
        resolution_map = plot_config.get("resolution", {})
        camera_angles = plot_config.get("camera_angle", {})
        render_settings = resolution_map.get(quality, resolution_map.get("medium", {}))
        stride_pair = stride or render_settings.get("stride", (1, 1))

        for coord_key, dataset in data.items():
            view_angle = camera_angles[coord_key]
            if coord_key == "rt" and z_idx is not None:
                # Plot intensity or density for each z position
                x, y = self.get_2d_grid("rt")
                x_st = x[:: stride_pair[0], :: stride_pair[1]]
                y_st = y[:: stride_pair[0], :: stride_pair[1]]
                x_label = plot_config["label"]["xr"]
                y_label = plot_config["label"]["xt"]
                colorbar_label = plot_config["bar_label"]
                for idx in range(len(z_idx)):
                    z_st = dataset[:, idx, :][:: stride_pair[0], :: stride_pair[1]]

                    fig, ax = self._render_surface_plot(
                        x_st,
                        y_st,
                        z_st,
                        plot_config["cmap"],
                        render_settings["dpi"],
                    )
                    ax.view_init(
                        elev=view_angle["elevation"], azim=view_angle["azimuth"]
                    )
                    ax.set(
                        xlabel=x_label,
                        ylabel=y_label,
                        zlabel=colorbar_label,
                    )

                    z_pos = z_coor[idx]
                    z_pos_format = f"{z_pos:.2f}"
                    title = plot_config["title"][coord_key].replace(
                        "{:.2f}", z_pos_format
                    )
                    ax.set_title(title)

                    filename = (
                        f"3d_{magnitude}_{coord_key}_{z_pos:.2f}".replace(".", "-")
                        + ".png"
                    )
                    self.save_or_display(
                        fig, filename, fig_path, render_settings["dpi"]
                    )

            elif coord_key == "zt":
                # Plot intensity or density on-axis
                x, y = self.get_2d_grid("zt")
                x_st = x[:: stride_pair[0], :: stride_pair[1]]
                y_st = y[:: stride_pair[0], :: stride_pair[1]]
                z_st = dataset[:: stride_pair[0], :: stride_pair[1]]
                x_label = plot_config["label"]["xz"]
                y_label = plot_config["label"]["xt"]
                colorbar_label = plot_config["bar_label"]

                fig, ax = self._render_surface_plot(
                    x_st,
                    y_st,
                    z_st,
                    plot_config["cmap"],
                    render_settings["dpi"],
                )
                ax.view_init(elev=view_angle["elevation"], azim=view_angle["azimuth"])
                ax.set(
                    xlabel=x_label,
                    ylabel=y_label,
                    zlabel=colorbar_label,
                )
                ax.set_title(plot_config["title"][coord_key])

                filename = f"3d_{magnitude}_{coord_key}.png"
                self.save_or_display(fig, filename, fig_path, render_settings["dpi"])

            elif coord_key == "rz":
                # Plot intensity or density peak value
                x, y = self.get_2d_grid("rz")
                x_st = x[:: stride_pair[0], :: stride_pair[1]]
                y_st = y[:: stride_pair[0], :: stride_pair[1]]
                z_st = dataset[:: stride_pair[0], :: stride_pair[1]]
                x_label = plot_config["label"]["xr"]
                y_label = plot_config["label"]["xz"]
                colorbar_label = plot_config["bar_label"]

                fig, ax = self._render_surface_plot(
                    x_st,
                    y_st,
                    z_st,
                    plot_config["cmap"],
                    render_settings["dpi"],
                )
                ax.view_init(elev=view_angle["elevation"], azim=view_angle["azimuth"])
                ax.set(
                    xlabel=x_label,
                    ylabel=y_label,
                    zlabel=colorbar_label,
                )
                ax.set_title(plot_config["title"][coord_key])

                filename = f"3d_{magnitude}_{coord_key}.png"
                self.save_or_display(fig, filename, fig_path, render_settings["dpi"])


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
        """Compute intensity data."""
        return self.base_plot.compute_intensity(
            envelope_dist, envelope_axis, envelope_peak
        )

    def get_density_data(self, density_dist, density_axis, density_peak):
        """Compute density data."""
        return self.base_plot.compute_density(density_dist, density_axis, density_peak)

    def get_fluence_data(self, b_fluence):
        """Compute fluence data."""
        return self.base_plot.compute_fluence(b_fluence)

    def get_radius_data(self, b_fluence, radial_grid):
        """Compute beam radius data using fluence."""
        return self.base_plot.compute_radius(b_fluence, radial_grid)

    def create_1d_plot(
        self,
        data,
        z_idx=None,
        z_coor=None,
        magnitude="intensity",
        fig_path=None,
        scale="both",
        log_y_range=None,
    ):
        """Create line plots."""
        self.plot_1d.render_1d_data(
            data, z_idx, z_coor, magnitude, fig_path, scale, log_y_range
        )

    def create_2d_plot(
        self,
        data,
        z_idx=None,
        z_coor=None,
        magnitude="intensity",
        quality="medium",
        fig_path=None,
        stride=None,
        scale="both",
        log_y_range=None,
        log_rt_levels=None,
    ):
        """Create colormap plots."""
        self.plot_2d.render_2d_data(
            data,
            z_idx,
            z_coor,
            magnitude,
            quality,
            fig_path,
            stride,
            scale,
            log_y_range,
            log_rt_levels,
        )

    def create_3d_plot(
        self,
        data,
        z_idx,
        z_coor=None,
        magnitude="intensity",
        quality="medium",
        fig_path=None,
        stride=None,
    ):
        """Create 3D solution plots."""
        self.plot_3d.render_3d_data(
            data, z_idx, z_coor, magnitude, quality, fig_path, stride
        )


def parse_cli_options():
    """Parse and validate CLI options."""
    user_paths = get_user_paths(create=False)

    def parse_range_arg(raw_value, argument_name, fail_hard=False):
        """Parse a comma-separated numeric range."""
        if raw_value is None:
            return None

        try:
            lower, upper = map(float, raw_value.split(","))
            return (lower, upper)
        except ValueError as exc:
            if fail_hard:
                raise ValueError(f"{argument_name} format must be 'min,max'.") from exc
            print(
                f"Error: {argument_name} format must be 'min,max'. " "Using full range."
            )
            return None

    def parse_csv_flags(raw_value):
        """Parse comma-separated values to enabled-flag dictionary."""
        return {item: True for item in raw_value.split(",")}

    parser = argparse.ArgumentParser(
        description="Plot simulation data from HDF5 files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="TOML configuration file used for sim/figure path defaults.",
    )
    parser.add_argument(
        "--sim-path",
        type=Path,
        default=None,
        help="Path to simulation data file.",
    )
    parser.add_argument(
        "--fig-path",
        type=Path,
        default=None,
        help="Path to figures file.",
    )
    parser.add_argument(
        "--variables",
        default=",".join(SUPPORTED_VARIABLES),
        help="Variables to plot: intensity,density,fluence,radius (comma-separated).",
    )
    parser.add_argument(
        "--dimensions",
        default="1d,2d,3d",
        help="Dimensions to generate: 1d,2d,3d (comma-separated).",
    )
    parser.add_argument(
        "--resolution",
        default="medium",
        help="Plot quality for 3D plots: low, medium, high.",
    )
    parser.add_argument(
        "--scale",
        default="both",
        choices=["linear", "log", "both"],
        help="Axis scale for 1D and 2D plots.",
    )
    parser.add_argument(
        "--colors-1d",
        default=None,
        choices=sorted(AVAILABLE_1D_COLORS.keys()),
        help="Line color key for 1D plots. Default: #1E90FF.",
    )
    parser.add_argument(
        "--colors-2d",
        default=None,
        choices=tuple(AVAILABLE_2D_COLORS.keys()),
        help="Colormap key for 2D/3D plots. Default: viridis.",
    )
    parser.add_argument(
        "--log-y-range",
        default=None,
        help="Log y-range as 'min,max' (used for 1D/2D rz log plots).",
    )
    parser.add_argument(
        "--log-rt-levels",
        type=int,
        default=None,
        help=(
            "Number of contour levels for 2D rt log plots "
            "(required for --scale log|both when plotting "
            "2D intensity/density)."
        ),
    )
    parser.add_argument(
        "--stride",
        default="1,1",
        help="Pixel stride (x,y format) for plotting 2D and 3D plots.",
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
        default=False,
        help="Plot all radial axis symmetrically.",
    )

    args = parser.parse_args()

    sim_default = user_paths["sim_dir"]
    fig_default = user_paths["fig_dir"]

    if args.config is not None:
        with open(args.config, "rb") as f:
            cfg = tomllib.load(f)

        cfg_output_path = cfg.get("data_output_path")
        cfg_figure_output_path = cfg.get("figure_output_path")

        if cfg_output_path:
            sim_default = Path(cfg_output_path)
            fig_default = sim_default / "figures"

        if cfg_figure_output_path:
            fig_default = Path(cfg_figure_output_path)

    args.sim_path = args.sim_path or sim_default
    args.fig_path = args.fig_path or fig_default

    args.axial_range = parse_range_arg(args.axial_range, "Axial range")
    args.time_range = parse_range_arg(args.time_range, "Time range")
    args.log_y_range = parse_range_arg(
        args.log_y_range, "--log-y-range", fail_hard=True
    )

    if args.log_rt_levels is not None and args.log_rt_levels <= 0:
        raise ValueError("--log-rt-levels must be a positive integer.")

    # Convert comma-separated strings to dictionaries for easier access
    stride_pair = [int(s) for s in args.stride.split(",")]
    args.variables = parse_csv_flags(args.variables)
    args.dimensions = parse_csv_flags(args.dimensions)
    args.stride = (stride_pair[0], stride_pair[1])

    has_1d = args.dimensions.get("1d", False)
    has_2d = args.dimensions.get("2d", False)
    is_log_scale = args.scale in ("log", "both")

    if is_log_scale and (has_1d or has_2d) and args.log_y_range is None:
        raise ValueError(
            "--log-y-range is required for log plotting when '1d' or '2d' dimensions are requested."
        )

    if is_log_scale and has_2d and args.log_rt_levels is None:
        raise ValueError(
            "--log-rt-levels is required for log plotting when '2d' dimension is requested."
        )

    return args


def _display_path(path: Path, root: Path) -> Path:
    """Return path relative to root when possible."""
    try:
        return path.relative_to(root)
    except ValueError:
        return path


def setup_output_directory(args):
    """Prepare output directory for file-based plotting, if requested."""
    base_dir = get_base_dir()

    if args.fig_path:
        fig_path = args.fig_path
        fig_path.mkdir(parents=True, exist_ok=True)
        print(f"Saving figures to file: {_display_path(fig_path, base_dir)}")
    else:
        print("Displaying figures interactively.")


def load_simulation_data(directory):
    """Load simulation data from HDF5 files."""
    base_dir = get_base_dir()
    snapshots_path = directory / "acherus_snapshots.h5"
    diagnostics_path = directory / "acherus_diagnostics.h5"

    data = {}

    has_snapshots = snapshots_path.exists()
    has_diagnostics = diagnostics_path.exists()

    if has_snapshots:
        print(f"Loading data from file: {_display_path(snapshots_path, base_dir)}")

        with File(snapshots_path, "r") as f:
            data["z_idx"] = np.array(f["snap_z_idx"])
            if "envelope_snapshot_rzt" in f:
                data["e_dist"] = np.array(f["envelope_snapshot_rzt"])
            if "density_snapshot_rzt" in f:
                data["elec_dist"] = np.array(f["density_snapshot_rzt"])

    if has_diagnostics:
        print(f"Loading data from file: {_display_path(diagnostics_path, base_dir)}")

        with File(diagnostics_path, "r") as f:
            coor = f["coordinates"]
            data["ini_radi_coor"] = coor["r_min"][()]
            data["fin_radi_coor"] = coor["r_max"][()]
            data["ini_dist_coor"] = coor["z_min"][()]
            data["fin_dist_coor"] = coor["z_max"][()]
            data["ini_time_coor"] = coor["t_min"][()]
            data["fin_time_coor"] = coor["t_max"][()]

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

            if "fluence" in f:
                fluence = f["fluence"]
                if "fluence_rz" in fluence:
                    data["b_fluence"] = np.array(fluence["fluence_rz"])

    if not has_diagnostics and not has_snapshots:
        raise FileNotFoundError(
            f"No files were found in {_display_path(directory, base_dir)}"
        )

    return data


def plot_request(data_type, plot_data, plot_types, plot, z_snap_idx, z_snap_coor, args):
    """Generate requested physical magnitudes for the specified data."""

    if plot_types.get("1d", False):
        print(f"Generating 1D (line) plots for {data_type} ...")
        plot.create_1d_plot(
            plot_data,
            z_snap_idx,
            z_snap_coor,
            data_type,
            args.fig_path,
            args.scale,
            args.log_y_range,
        )

    if plot_types.get("2d", False) and data_type != "radius":
        print(f"Generating 2D (colormap) plots for {data_type} ...")
        plot.create_2d_plot(
            plot_data,
            z_snap_idx,
            z_snap_coor,
            data_type,
            args.resolution,
            args.fig_path,
            args.stride,
            args.scale,
            args.log_y_range,
            args.log_rt_levels,
        )

    if plot_types.get("3d", False) and data_type != "radius":
        print(f"Generating 3D (surface) plots for {data_type} ...")
        plot.create_3d_plot(
            plot_data,
            z_snap_idx,
            z_snap_coor,
            data_type,
            args.resolution,
            args.fig_path,
            args.stride,
        )


def process_simulation_data(data_type, data, plot, box, plot_types, args):
    """Process a specific physical variable and generate the plots."""
    print(f"Processing {data_type} data...")

    z_snap_idx = data["z_idx"]
    z_snap_coor = box.set_snapshot_points(z_snap_idx)

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
        plot_data = {"rz": plot_data_fluence}
    elif data_type == "radius":
        plot_data_radius = plot.get_radius_data(
            box.sliced_data["b_fluence"],
            box.sliced_grids["r"],
        )
        plot_data = {"z": plot_data_radius}
    else:
        raise ValueError(f"Unsupported physical variable: {data_type}")

    plot_request(data_type, plot_data, plot_types, plot, z_snap_idx, z_snap_coor, args)


def main():
    """Main execution function."""

    args = parse_cli_options()
    data = load_simulation_data(args.sim_path)
    setup_output_directory(args)

    units = Units()
    color_1d = AVAILABLE_1D_COLORS.get(args.colors_1d, "#1E90FF")
    color_2d = args.colors_2d or "viridis"
    config = PlotConfiguration(default_1d_color=color_1d, default_2d_color=color_2d)
    box = SimulationBox(
        units,
        data,
        args.radial_symmetry,
        args.radial_limit,
        args.axial_range,
        args.time_range,
    )
    box_units = SimulationBoxUnits(units, box, config)
    plot = VisualManager(units, box, config, box_units)

    # Process each requested data type
    for variable, enabled in args.variables.items():
        if enabled and variable in SUPPORTED_VARIABLES:
            process_simulation_data(variable, data, plot, box, args.dimensions, args)

    print("Plotting complete!")


if __name__ == "__main__":
    main()
