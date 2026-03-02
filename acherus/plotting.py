"""
Python tool for plotting NumPy arrays saved after
the simulations have finished execution.
"""

import argparse
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
from h5py import File
from matplotlib.colors import LogNorm

from .data.paths import get_base_dir, get_user_paths

SUPPORTED_VARIABLES = ("intensity", "density", "fluence", "radius")


def _format_z_snapshots(value, decimals=6):
    """Format z-snapshots coordinate values for figure names."""
    text = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    if text in ("", "-0"):
        return "0"
    return text


@dataclass
class PlotConfiguration:
    """Plot style configuration."""

    default_1d_color: tuple = (0.121569, 0.466667, 0.705882)
    default_2d_color: str = "viridis"

    def get_plot_config(self, plot_type: str) -> Dict:
        """
        Return configuration for specified plot type.

        Args:
        - plot_type: Physical magnitude ("intensity", "density", "fluence", or "radius")

        Returns:
        - Dictionary with predefined configuration settings for plotting.
        """
        # Physical magnitude configuration options
        base_configuration = {
            "intensity": {
                "cmap": plt.get_cmap(self.default_2d_color),
                "line_color": self.default_1d_color,
                "bar_label": r"Intensity $[W/cm^2]$",
                "label": {
                    "xr": r"$r$ [m]",
                    "xz": r"$z$ [m]",
                    "xt": r"$t$ [s]",
                    "yt": r"$I(r=0,t)$ $[W/cm^2]$",
                    "yz": r"$\max_t\, I(r=0,z,t)$ $[W/cm^2]$",
                },
            },
            "density": {
                "cmap": plt.get_cmap(self.default_2d_color),
                "line_color": self.default_1d_color,
                "bar_label": r"Electron density $[cm^{-3}]$",
                "label": {
                    "xr": r"$r$ [m]",
                    "xz": r"$z$ [m]",
                    "xt": r"$t$ [s]",
                    "yt": r"$\rho_e(r=0,t)$ $[cm^{-3}]$",
                    "yz": r"$\max_t\, \rho_e(r=0,z,t)$ $[cm^{-3}]$",
                },
            },
            "fluence": {
                "cmap": plt.get_cmap(self.default_2d_color),
                "line_color": self.default_1d_color,
                "bar_label": r"Fluence $[J/cm^2]$",
                "label": {
                    "xr": r"$r$ [m]",
                    "xz": r"$z$ [m]",
                    "yz": r"$F(r=0,z)$ $[J/cm^2]$",
                },
            },
            "radius": {
                "cmap": plt.get_cmap(self.default_2d_color),
                "line_color": self.default_1d_color,
                "label": {
                    "xz": r"$z$ [m]",
                    "yz": r"$R(z)$ [m]",
                },
            },
        }

        return base_configuration[plot_type]


class SimulationBox:
    """Simulation plotting domain with optional slicing constraints."""

    def __init__(
        self,
        data: Dict[str, Any],
        radial_symmetry: bool = False,
        radial_limit: float = None,
        axial_range: tuple = None,
        time_range: tuple = None,
    ):
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

        r_min, r_max = self.r_min_ori, self.r_max_ori
        z_min, z_max = self.z_min_ori, self.z_max_ori
        t_min, t_max = self.t_min_ori, self.t_max_ori

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
        z_min = self.data["ini_dist_coor"]
        z_max = self.data["fin_dist_coor"]
        z_snap_coords = z_min + (indices * (z_max - z_min) / (self.nz - 1))
        return z_snap_coords

    def flip_radial_data(self, data, axis_r=0):
        """Mirror radial data for symmetry."""
        if not self.r_sym:
            return data

        flipped_data = np.flip(data[1:], axis=axis_r)
        return np.concatenate((flipped_data, data), axis=axis_r)


class SimulationBoxUnits:
    """Cached unit-scaled plotting grids."""

    def __init__(self, box: SimulationBox, config: PlotConfiguration):
        self.box = box
        self.config = config
        self.scaled_1d_grid = {}
        self.scaled_2d_grid = {}

    def create_unit_scaled_1d_grid(self, grid_type):
        """Get a cached 1D grid array."""
        if grid_type not in self.scaled_1d_grid:
            self.scaled_1d_grid[grid_type] = self.box.sliced_grids[grid_type]

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
        box: SimulationBox,
        config: PlotConfiguration,
        box_units: SimulationBoxUnits,
    ):
        self.box = box
        self.config = config
        self.box_units = box_units

    def compute_intensity(self, envelope_dist, envelope_axis, envelope_peak):
        """Set up intensities for plotting."""
        return 1e-4 * (
            np.abs(envelope_dist) ** 2,
            np.abs(envelope_axis) ** 2,
            np.abs(envelope_peak) ** 2,
        )

    def compute_density(self, density_dist, density_axis, density_peak):
        """Set up densities for plotting."""
        return 1e-6 * (
            density_dist,
            density_axis,
            density_peak,
        )

    def compute_fluence(self, b_fluence):
        """Set up fluence distribution for plotting."""
        return 1e-4 * b_fluence

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

        return b_radius

    def save_or_display(self, fig, filename, fig_path, dpi=300):
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
        z_snap_idx=None,
        z_snap_coords=None,
        variable=None,
        dpi=300,
        fig_path=None,
        scale="linear",
        log_y_range=None,
    ):
        """Create 1D plots."""
        if variable is None:
            raise ValueError("variable must be provided for 1D plotting.")

        plot_config = self.config.get_plot_config(variable)

        for coord_key, dataset in data.items():
            if coord_key == "rt" and z_snap_idx is not None:
                x_st = self.get_1d_grid("t")
                x_label = plot_config["label"]["xt"]
                y_label = plot_config["label"]["yt"]
                for idx in range(len(z_snap_idx)):
                    y_st = dataset[self.box.nr_0, idx, :]

                    fig, ax = plt.subplots()
                    ax.plot(x_st, y_st, color=plot_config["line_color"])
                    ax.set(xlabel=x_label, ylabel=y_label)

                    z_pos = z_snap_coords[idx]
                    z_pos_format = _format_z_snapshots(z_pos)
                    z_pos_file = z_pos_format.replace(".", "-")
                    filename = f"1d_{variable}_t_{z_pos_file}.png"
                    self.save_or_display(fig, filename, fig_path, dpi)

            elif coord_key == "rz":
                x_st = self.get_1d_grid("z")
                y_st = dataset[self.box.nr_0, :]
                x_label = plot_config["label"]["xz"]
                y_label = plot_config["label"]["yz"]

                if scale in ("both", "linear"):
                    fig, ax = plt.subplots()
                    ax.plot(x_st, y_st, color=plot_config["line_color"])
                    ax.set(xlabel=x_label, ylabel=y_label)

                    filename = f"1d_{variable}_z.png"
                    self.save_or_display(fig, filename, fig_path, dpi)

                if scale in ("both", "log"):
                    if log_y_range is None:
                        raise ValueError(
                            "Log 1D plots require --log-y-range 'min,max'."
                        )

                    y_min, y_max = log_y_range
                    y_log = np.where(
                        (y_st > 0) & (y_st >= y_min) & (y_st <= y_max), y_st, np.nan
                    )
                    if not np.any(np.isfinite(y_log)):
                        continue

                    fig, ax = plt.subplots()
                    ax.plot(x_st, y_log, color=plot_config["line_color"])
                    ax.set(xlabel=x_label, ylabel=y_label)
                    ax.set_yscale("log")
                    ax.set_ylim(bottom=y_min, top=y_max)

                    filename = f"1d_{variable}_z_log.png"
                    self.save_or_display(fig, filename, fig_path, dpi)

            elif coord_key == "z" and variable == "radius":
                x_st = self.get_1d_grid("z")
                y_st = dataset
                x_label = plot_config["label"]["xz"]
                y_label = plot_config["label"]["yz"]

                fig, ax = plt.subplots()
                if self.box.r_sym:
                    ax.plot(x_st, y_st, color=plot_config["line_color"])
                    ax.plot(x_st, -y_st, color=plot_config["line_color"])
                else:
                    ax.plot(x_st, y_st, color=plot_config["line_color"])

                ax.set(xlabel=x_label, ylabel=y_label)

                filename = f"1d_{variable}_z.png"
                self.save_or_display(fig, filename, fig_path, dpi)


class Plot2D(BasePlot):
    """Plotting class for 2D (colormap) plots."""

    @staticmethod
    def _render_cmap_plot(
        x_coords,
        y_coords,
        z_values,
        color_map,
        x_label,
        y_label,
        colorbar_label,
        norm=None,
    ):
        """Create a pcolormesh figure and return figure/axis."""
        fig, ax = plt.subplots()
        mesh = ax.pcolormesh(x_coords, y_coords, z_values, cmap=color_map, norm=norm)
        fig.colorbar(mesh, ax=ax, label=colorbar_label)
        ax.set(xlabel=x_label, ylabel=y_label)
        return fig

    def render_2d_data(
        self,
        data,
        z_snap_idx=None,
        z_snap_coords=None,
        variable=None,
        dpi=300,
        fig_path=None,
        scale="linear",
        log_y_range=None,
        log_rt_levels=None,
    ):
        """Create 2D plots."""
        if variable is None:
            raise ValueError("variable must be provided for 2D plotting.")

        plot_config = self.config.get_plot_config(variable)

        for coord_key, dataset in data.items():
            if coord_key == "rt" and z_snap_idx is not None:
                y_st, x_st = self.get_2d_grid("rt")
                x_label = plot_config["label"]["xt"]
                y_label = plot_config["label"]["xr"]
                colorbar_label = plot_config["bar_label"]
                for idx in range(len(z_snap_idx)):
                    z_st = dataset[:, idx, :]
                    z_pos_coord = z_snap_coords[idx]
                    z_pos_format = _format_z_snapshots(z_pos_coord)
                    z_pos_file = z_pos_format.replace(".", "-")

                    if scale in ("both", "linear"):
                        fig = self._render_cmap_plot(
                            x_st,
                            y_st,
                            z_st,
                            plot_config["cmap"],
                            x_label,
                            y_label,
                            colorbar_label,
                        )

                        filename = f"2d_{variable}_{coord_key}_{z_pos_file}.png"
                        self.save_or_display(fig, filename, fig_path, dpi)

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
                        z_norm = np.ma.masked_less(z_norm, vmin)
                        levels_per_decade = 10
                        levels = np.logspace(
                            -log_rt_levels,
                            0,
                            log_rt_levels * levels_per_decade + 1,
                        )
                        rt_cmap = plot_config["cmap"].copy()
                        rt_cmap.set_bad("white")

                        fig, ax = plt.subplots()
                        mesh = ax.contourf(
                            x_st,
                            y_st,
                            z_norm,
                            levels=levels,
                            cmap=rt_cmap,
                            norm=LogNorm(vmin=vmin, vmax=1.0),
                            extend="neither",
                        )
                        fig.colorbar(
                            mesh, ax=ax, label=colorbar_label + " (normalized)"
                        )
                        ax.set(xlabel=x_label, ylabel=y_label)

                        filename = f"2d_{variable}_{coord_key}_{z_pos_file}_log.png"
                        self.save_or_display(fig, filename, fig_path, dpi)

            elif coord_key == "zt":
                x_st, y_st = self.get_2d_grid("zt")
                z_st = dataset
                x_label = plot_config["label"]["xz"]
                y_label = plot_config["label"]["xt"]
                colorbar_label = plot_config["bar_label"]

                fig = self._render_cmap_plot(
                    x_st,
                    y_st,
                    z_st,
                    plot_config["cmap"],
                    x_label,
                    y_label,
                    colorbar_label,
                )

                filename = f"2d_{variable}_{coord_key}.png"
                self.save_or_display(fig, filename, fig_path, dpi)

            elif coord_key == "rz":
                y_st, x_st = self.get_2d_grid("rz")
                z_st = dataset
                x_label = plot_config["label"]["xz"]
                y_label = plot_config["label"]["xr"]
                colorbar_label = plot_config["bar_label"]

                if scale in ("both", "linear"):
                    fig = self._render_cmap_plot(
                        x_st,
                        y_st,
                        z_st,
                        plot_config["cmap"],
                        x_label,
                        y_label,
                        colorbar_label,
                    )

                    filename = f"2d_{variable}_{coord_key}.png"
                    self.save_or_display(fig, filename, fig_path, dpi)

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

                        fig = self._render_cmap_plot(
                            x_st,
                            y_st,
                            z_log,
                            plot_config["cmap"],
                            x_label,
                            y_label,
                            colorbar_label,
                            norm=LogNorm(vmin=z_min, vmax=z_max),
                        )

                        filename = f"2d_{variable}_{coord_key}_log.png"
                        self.save_or_display(fig, filename, fig_path, dpi)


class Plot3D(BasePlot):
    """Plotting class for 3D (surface) plots."""

    @staticmethod
    def _render_mesh_plot(x_coords, y_coords, z_values, color_map, dpi):
        """Create a 3D surface plot and return figure/axis."""
        fig = plt.figure(dpi=dpi)
        ax = fig.add_subplot(projection="3d")
        ax.plot_surface(x_coords, y_coords, z_values, cmap=color_map)
        return fig, ax

    def render_3d_data(
        self,
        data,
        z_snap_idx,
        z_snap_coords=None,
        variable=None,
        dpi=300,
        fig_path=None,
        camera_view=(200, 15, 0),
    ):
        """Create 3D plots."""
        if variable is None:
            raise ValueError("variable must be provided for 3D plotting.")

        plot_config = self.config.get_plot_config(variable)
        azimuth, elevation, altitude = camera_view

        def apply_camera(axis):
            try:
                axis.view_init(elev=elevation, azim=azimuth, roll=altitude)
            except TypeError:
                axis.view_init(elev=elevation, azim=azimuth)

        for coord_key, dataset in data.items():
            if coord_key == "rt" and z_snap_idx is not None:
                x_st, y_st = self.get_2d_grid("rt")
                x_label = plot_config["label"]["xr"]
                y_label = plot_config["label"]["xt"]
                colorbar_label = plot_config["bar_label"]
                for idx in range(len(z_snap_idx)):
                    z_st = dataset[:, idx, :]

                    fig, ax = self._render_mesh_plot(
                        x_st,
                        y_st,
                        z_st,
                        plot_config["cmap"],
                        dpi,
                    )
                    apply_camera(ax)
                    ax.set(
                        xlabel=x_label,
                        ylabel=y_label,
                        zlabel=colorbar_label,
                    )

                    z_pos = z_snap_coords[idx]
                    z_pos_format = _format_z_snapshots(z_pos)

                    z_pos_file = z_pos_format.replace(".", "-")
                    filename = f"3d_{variable}_{coord_key}_{z_pos_file}.png"
                    self.save_or_display(fig, filename, fig_path, dpi)

            elif coord_key == "zt":
                x_st, y_st = self.get_2d_grid("zt")
                z_st = dataset
                x_label = plot_config["label"]["xz"]
                y_label = plot_config["label"]["xt"]
                colorbar_label = plot_config["bar_label"]

                fig, ax = self._render_mesh_plot(
                    x_st,
                    y_st,
                    z_st,
                    plot_config["cmap"],
                    dpi,
                )
                apply_camera(ax)
                ax.set(
                    xlabel=x_label,
                    ylabel=y_label,
                    zlabel=colorbar_label,
                )

                filename = f"3d_{variable}_{coord_key}.png"
                self.save_or_display(fig, filename, fig_path, dpi)

            elif coord_key == "rz":
                x_st, y_st = self.get_2d_grid("rz")
                z_st = dataset
                x_label = plot_config["label"]["xr"]
                y_label = plot_config["label"]["xz"]
                colorbar_label = plot_config["bar_label"]

                fig, ax = self._render_mesh_plot(
                    x_st,
                    y_st,
                    z_st,
                    plot_config["cmap"],
                    dpi,
                )
                apply_camera(ax)
                ax.set(
                    xlabel=x_label,
                    ylabel=y_label,
                    zlabel=colorbar_label,
                )

                filename = f"3d_{variable}_{coord_key}.png"
                self.save_or_display(fig, filename, fig_path, dpi)


class VisualManager:
    """Manages all plotting classes."""

    def __init__(self, box, config, box_units):
        self.box = box
        self.config = config
        self.box_units = box_units

        # Initialize specialized plotters
        self.base_plot = BasePlot(box, config, box_units)
        self.plot_1d = Plot1D(box, config, box_units)
        self.plot_2d = Plot2D(box, config, box_units)
        self.plot_3d = Plot3D(box, config, box_units)

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
        z_snap_idx=None,
        z_snap_coords=None,
        variable=None,
        dpi=300,
        fig_path=None,
        scale="linear",
        log_y_range=None,
    ):
        """Create line plots."""
        self.plot_1d.render_1d_data(
            data, z_snap_idx, z_snap_coords, variable, dpi, fig_path, scale, log_y_range
        )

    def create_2d_plot(
        self,
        data,
        z_snap_idx=None,
        z_snap_coords=None,
        variable=None,
        dpi=300,
        fig_path=None,
        scale="linear",
        log_y_range=None,
        log_rt_levels=None,
    ):
        """Create colormap plots."""
        self.plot_2d.render_2d_data(
            data,
            z_snap_idx,
            z_snap_coords,
            variable,
            dpi,
            fig_path,
            scale,
            log_y_range,
            log_rt_levels,
        )

    def create_3d_plot(
        self,
        data,
        z_snap_idx=None,
        z_snap_coords=None,
        variable=None,
        dpi=300,
        fig_path=None,
        camera_view=(200, 15, 0),
    ):
        """Create 3D solution plots."""
        self.plot_3d.render_3d_data(
            data,
            z_snap_idx,
            z_snap_coords,
            variable,
            dpi,
            fig_path,
            camera_view,
        )


def parse_cli_options():
    """Parse and validate CLI options."""
    user_paths = get_user_paths(create=False)

    def parse_range_arg(raw_text, argument_name, fail_hard=False):
        """Parse a comma-separated numeric range."""
        if raw_text is None:
            return None

        try:
            lower, upper = map(float, raw_text.split(","))
            return (lower, upper)
        except ValueError as exc:
            if fail_hard:
                raise ValueError(f"{argument_name} format must be 'min,max'.") from exc
            print(
                f"Error: {argument_name} format must be 'min,max'. " "Using full range."
            )
            return None

    def parse_csv_flags(raw_text):
        """Parse comma-separated values to enabled-flag dictionary."""
        return {item: True for item in raw_text.split(",")}

    def parse_camera_view_arg(raw_text):
        """Parse camera view as 'azimuth,elevation,altitude'."""
        try:
            azimuth, elevation, altitude = map(int, raw_text.split(","))
            return (azimuth, elevation, altitude)
        except ValueError as exc:
            raise ValueError(
                "--camera-view format must be 'azimuth,elevation,altitude'."
            ) from exc

    def parse_1d_color_arg(raw_text):
        """Parse 1D color as 'r,g,b' with values between 0 and 1."""
        try:
            values = tuple(map(float, raw_text.split(",")))
        except ValueError as exc:
            raise ValueError(
                "--colors-1d format must be 'r,g,b' with numbers between 0 and 1."
            ) from exc

        if len(values) != 3:
            raise ValueError("--colors-1d must contain exactly 3 values: 'r,g,b'.")

        if any((value < 0.0 or value > 1.0) for value in values):
            raise ValueError("--colors-1d values must be between 0 and 1.")

        return values

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
        default="1d,2d",
        help="Dimensions to generate: 1d,2d (comma-separated).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure resolution in dots per inch for all plot dimensions.",
    )
    parser.add_argument(
        "--scale",
        default="linear",
        choices=["linear", "log", "both"],
        help="Axis scale for 1D and 2D plots.",
    )
    parser.add_argument(
        "--colors-1d",
        default="0.121569,0.466667,0.705882",
        help="Line color for 1D plots as RGB triplet 'r,g,b' with values in [0,1].",
    )
    parser.add_argument(
        "--colors-2d",
        default="viridis",
        help="Matplotlib colormap name for 2D/3D plots (e.g. viridis).",
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
        "--camera-view",
        default="200,15,0",
        help="3D camera view as 'azimuth,elevation,altitude'.",
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

    if args.dpi <= 0:
        raise ValueError("--dpi must be a positive integer.")

    args.colors_1d = parse_1d_color_arg(args.colors_1d)

    if args.colors_2d not in plt.colormaps():
        raise ValueError("--colors-2d must be a valid matplotlib colormap name.")

    # Convert comma-separated strings to dictionaries for easier access
    args.variables = parse_csv_flags(args.variables)
    args.dimensions = parse_csv_flags(args.dimensions)

    args.camera_view = parse_camera_view_arg(args.camera_view)

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

    def copy_arrays_from_group(file_obj, group_name, mapping):
        if group_name not in file_obj:
            return
        group = file_obj[group_name]
        for source_key, target_key in mapping.items():
            if source_key in group:
                data[target_key] = np.array(group[source_key])

    has_snapshots = snapshots_path.exists()
    has_diagnostics = diagnostics_path.exists()

    if has_snapshots:
        print(f"Loading data from file: {_display_path(snapshots_path, base_dir)}")

        with File(snapshots_path, "r") as f:
            data["z_idx"] = np.array(f["snap_z_idx"])
            for source_key, target_key in {
                "envelope_snapshot_rzt": "e_dist",
                "density_snapshot_rzt": "elec_dist",
            }.items():
                if source_key in f:
                    data[target_key] = np.array(f[source_key])

    if has_diagnostics:
        print(f"Loading data from file: {_display_path(diagnostics_path, base_dir)}")

        with File(diagnostics_path, "r") as f:
            coor = f["coordinates"]
            for source_key, target_key in {
                "r_min": "ini_radi_coor",
                "r_max": "fin_radi_coor",
                "z_min": "ini_dist_coor",
                "z_max": "fin_dist_coor",
                "t_min": "ini_time_coor",
                "t_max": "fin_time_coor",
            }.items():
                data[target_key] = coor[source_key][()]

            copy_arrays_from_group(
                f,
                "envelope",
                {"axis_zt": "e_axis", "peak_rz": "e_peak"},
            )
            copy_arrays_from_group(
                f,
                "density",
                {"axis_zt": "elec_axis", "peak_rz": "elec_peak"},
            )
            copy_arrays_from_group(
                f,
                "fluence",
                {"fluence_rz": "b_fluence"},
            )

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
            args.dpi,
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
            args.dpi,
            args.fig_path,
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
            args.dpi,
            args.fig_path,
            args.camera_view,
        )


def process_simulation_data(data_type, data, plot, box, plot_types, args):
    """Process a specific physical variable and generate the plots."""
    print(f"Processing {data_type} data...")

    z_snap_idx = data["z_idx"]
    z_snap_coor = box.set_snapshot_points(z_snap_idx)

    three_view_fetchers = {
        "intensity": (
            plot.get_intensity_data,
            ("e_dist", "e_axis", "e_peak"),
        ),
        "density": (
            plot.get_density_data,
            ("elec_dist", "elec_axis", "elec_peak"),
        ),
    }

    if data_type in three_view_fetchers:
        fetcher, keys = three_view_fetchers[data_type]
        dist_data, axis_data, peak_data = fetcher(
            *(box.sliced_data[key] for key in keys)
        )
        plot_data = {"rt": dist_data, "zt": axis_data, "rz": peak_data}
    elif data_type == "fluence":
        plot_data = {"rz": plot.get_fluence_data(box.sliced_data["b_fluence"])}
    elif data_type == "radius":
        z_slice = box.sliced_coor["z"]
        fluence = box.data["b_fluence"][:, z_slice]
        radial_grid = np.linspace(box.r_min_ori, box.r_max_ori, box.nr)

        plot_data = {"z": plot.get_radius_data(fluence, radial_grid)}
    else:
        raise ValueError(f"Unsupported physical variable: {data_type}")

    plot_request(data_type, plot_data, plot_types, plot, z_snap_idx, z_snap_coor, args)


def main():
    """Run plotting CLI."""

    args = parse_cli_options()
    data = load_simulation_data(args.sim_path)
    setup_output_directory(args)

    config = PlotConfiguration(
        default_1d_color=args.colors_1d,
        default_2d_color=args.colors_2d,
    )
    box = SimulationBox(
        data,
        args.radial_symmetry,
        args.radial_limit,
        args.axial_range,
        args.time_range,
    )
    box_units = SimulationBoxUnits(box, config)
    plot = VisualManager(box, config, box_units)

    for variable, enabled in args.variables.items():
        if enabled and variable in SUPPORTED_VARIABLES:
            process_simulation_data(variable, data, plot, box, args.dimensions, args)

    print("Plotting complete!")


if __name__ == "__main__":
    main()
