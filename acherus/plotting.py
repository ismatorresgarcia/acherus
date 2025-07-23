"""
Python tool for plotting NumPy arrays saved after
the simulations have finished execution.
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import cmasher as cmr
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from h5py import File
from matplotlib.colors import LogNorm

from .data.paths import base_dir, fig_dir, sim_dir


@dataclass
class PlotConfiguration:
    """Plot style configuration."""

    colors1d: Dict[str, str] = field(
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
    colors2d: Dict[str, Any] = field(
        default_factory=lambda: {
            # Sequential colormaps
            "vir": mpl.colormaps["viridis"],
            "pls": mpl.colormaps["plasma"],
            "inf": mpl.colormaps["inferno"],
            "mag": mpl.colormaps["magma"],
            "civ": mpl.colormaps["cividis"],
            "sb": cmr.sunburst,  # pylint: disable=no-name-in-module
            "tx": cmr.toxic,  # pylint: disable=no-name-in-module
            # Diverging colormaps
            "rdbu": mpl.colormaps["RdBu"],
            "piyg": mpl.colormaps["PiYG"],
            "sp": mpl.colormaps["Spectral"],
            "rs": cmr.redshift,  # pylint: disable=no-name-in-module
            "ib": cmr.iceburn,  # pylint: disable=no-name-in-module
        }
    )

    def __post_init__(self):
        """Customize figure styling."""
        plt.style.use("default")

        plt.rcParams.update(
            {
                "figure.figsize": (13, 7),
                "axes.grid": False,
                "axes.linewidth": 0.8,
                "lines.linewidth": 1.5,
                "text.usetex": True,
                "text.latex.preamble": r"\usepackage{siunitx}",
                "font.family": "serif",
                "font.size": 10,
                "axes.labelsize": 11,
                "axes.titlesize": 12,
                "legend.framealpha": 0.8,
                "legend.loc": "upper right",
            }
        )

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
                "cmap": self.colors2d["pls"],
                "bar_label": r"Intensity $\Bigl[\unit{W/cm^2}\Bigr]$",
                "title": {
                    "z": "On-axis peak intensity over time",
                    "t": r"On-axis intensity at $z = {:.2f}$ $\unit{m}$",
                    "zt": "On-axis intensity",
                    "rt": r"Intensity at $z = {:.2f}$ $\unit{m}$",
                    "rz": "Peak intensity over time",
                },
                "label": {
                    "xr": r"$r$ $[\unit{mm}]$",
                    "xz": r"$z$ $[\unit{m}]$",
                    "xt": r"$t$ $[\unit{fs}]$",
                    "yt": r"$I(r=0,t)$ $\Bigl[\unit{W/cm^2}\Bigr]$",
                    "yz": r"$\max_t$ $I(r=0,z,t)$ $\Bigl[\unit{W/cm^2}\Bigr]$",
                },
            },
            "density": {
                "cmap": self.colors2d["pls"],
                "bar_label": r"Electron density $\Bigl[\unit{cm^{-3}}\Bigr]$",
                "title": {
                    "z": "On-axis peak electron density over time",
                    "t": r"On-axis electron density at $z = {:.2f}$ $\unit{m}$",
                    "zt": "On-axis electron density",
                    "rt": r"Electron density at $z = {:.2f}$ $\unit{m}$",
                    "rz": "Peak electron density over time",
                },
                "label": {
                    "xr": r"$r$ $[\unit{mm}]$",
                    "xz": r"$z$ $[\unit{m}]$",
                    "xt": r"$t$ $[\unit{fs}]$",
                    "yt": r"$\rho_e(r=0,t)$ $\Bigl[\unit{cm^{-3}}\Bigr]$",
                    "yz": r"$\max_t$ $\rho_e(r=0,z,t)$ $\Bigl[\unit{cm^{-3}}\Bigr]$",
                },
            },
            "fluence": {
                "cmap": self.colors2d["pls"],
                "bar_label": r"Fluence $\Bigl[\unit{J/cm^2}\Bigr]$",
                "title": {
                    "z": "On-axis fluence distribution",
                    "rz": "Fluence distribution",
                },
                "label": {
                    "xr": r"$r$ $[\unit{mm}]$",
                    "xz": r"$z$ $[\unit{m}]$",
                    "yz": r"$F(r=0,z)$ $\Bigl[\unit{J/cm^2}\Bigr]$",
                },
            },
            "radius": {
                "cmap": self.colors2d["pls"],
                "title": {"z": "Beam radius"},
                "label": {
                    "xz": r"$z$ $[\unit{m}]$",
                    "yz": r"$R(z)$ $[\unit{mm}]$",
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
    "Unit constants for converting between SI multiples and subdivisions."

    def __init__(
        self,
        factor_r=1e3,
        factor_z=1,
        factor_t=1e15,
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
        self.r_sym = radial_symmetry
        self.r_limit = radial_limit
        self.z_range = axial_range
        self.t_range = time_range
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

    def _init_grid_nodes(self):
        """Set up the plotting box boundary nodes."""
        self.nr = self.data["e_dist"].shape[0]
        self.nz = self.data["e_axis"].shape[0]
        self.nt = self.data["e_axis"].shape[1]

        if self.r_sym:
            dr_ori = self.r_max_ori - self.r_min_ori
            self.nr_sym = 2 * self.nr - 1
            self.nr_0 = -self.b_r[0] * (self.nr - 1) / dr_ori
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
            self.nodes[dim] = (int(n_min), int(n_max) + 1)

    def _init_sliced_arrays(self):
        """Set up computational arrays"""
        self.sliced_data = {}
        self.sliced_coor = {  # Get elements from n_min to n_max
            "r": slice(*self.nodes["r_data"]),
            "z": slice(*self.nodes["z_data"]),
            "t": slice(*self.nodes["t_data"]),
        }

        if self.r_sym:
            radius_p = np.linspace(0, self.r_max_ori, self.nr)
            radius_n = -np.flip(radius_p[:-1])
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

        # Slice beam radius data if present
        if "b_radius" in self.data:
            self.sliced_data["b_radius"] = self.data["b_radius"][self.sliced_coor["z"]]

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

        flipped_data = np.flip(data[:-1], axis=axis_r)
        return np.concatenate((flipped_data, data), axis=axis_r)


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
                    self.units.fr * self.box.sliced_grids[grid_type]
                )
            elif grid_type.startswith("z"):
                self._scaled_1d_grid[grid_type] = (
                    self.units.fz * self.box.sliced_grids[grid_type]
                )
            elif grid_type.startswith("t"):
                self._scaled_1d_grid[grid_type] = (
                    self.units.ft * self.box.sliced_grids[grid_type]
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

    def compute_radius(self, b_radius):
        """Set up beam radius for plotting."""
        return self.units.fr * b_radius

    def save_or_display(self, fig, filename, fig_path, dpi=150):
        """Save figure or display it."""
        if fig_path:
            fig_path = Path(fig_path)
            filepath = fig_path / filename
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
        k_i=None,
        z_c=None,
        magn="intensity",
        fig_path=None,
    ):
        """
        Create 1D (line) plots for different coordinate systems.

        Arguments:
            data: Dictionary containing the datasets for different coordinates:
            k_i: List of z indices to plot (for rt plots).
            z_c: List of z coordinates corresponding to k_i.
            magn: "intensity", "density", "fluence" or "radius".
            fig_path: Path to save figures instead of displaying them.
        """
        c = self.config.get_plot_config(magn, "1d")

        # Plot each coordinate system in a separate figure
        for ii, jj in data.items():
            if ii == "rt" and k_i is not None:
                # Plot intensity or density for each z-position
                # with respect to time
                x = self.get_1d_grid("t")
                x_label = c["label"]["xt"]
                y_label = c["label"]["yt"]
                for idx in range(len(k_i)):
                    y = jj[self.box.nr_0, idx, :]

                    fig, ax = plt.subplots()
                    ax.plot(x, y, color="tab:blue")
                    ax.set(xlabel=x_label, ylabel=y_label)

                    z_pos = z_c[idx]
                    z_pos_format = f"{z_pos:.2f}"
                    title = c["title"]["t"].replace("{:.2f}", z_pos_format)
                    ax.set_title(title)

                    filename = f"1d_{magn}_t_{z_pos:.2f}".replace(".", "-") + ".png"
                    self.save_or_display(fig, filename, fig_path, c["dpi"])

            elif ii == "rz":
                # Plot intensity or density peak value on-axis
                # with respect to distance
                x = self.get_1d_grid("z")
                y = jj[self.box.nr_0, :]
                x_label = c["label"]["xz"]
                y_label = c["label"]["yz"]

                fig, ax = plt.subplots()
                ax.plot(x, y, color="tab:blue")
                ax.set(xlabel=x_label, ylabel=y_label)
                # ax.set_yscale("log")
                # ax.set_ylim(bottom=1e11, top=1e13)
                ax.set_title(c["title"]["z"])

                filename = f"1d_{magn}_z.png"
                self.save_or_display(fig, filename, fig_path, c["dpi"])

            elif ii == "z" and magn == "radius":
                # Plot beam radius with respect to distance
                x = self.get_1d_grid("z")
                y = jj
                x_label = c["label"]["xz"]
                y_label = c["label"]["yz"]

                fig, ax = plt.subplots()
                if self.box.r_sym:
                    ax.plot(x, y, color="tab:blue")
                    ax.plot(x, -y, color="tab:blue")
                else:
                    ax.plot(x, y, color="tab:blue")

                ax.set(xlabel=x_label, ylabel=y_label)
                ax.set_title(c["title"]["z"])

                filename = f"1d_{magn}_z.png"
                self.save_or_display(fig, filename, fig_path, c["dpi"])


class Plot2D(BasePlot):
    """Plotting class for 2D (colormap) plots."""

    def render_2d_data(
        self,
        data,
        k_i=None,
        z_c=None,
        magn="intensity",
        qua="medium",
        fig_path=None,
        stride=None,
    ):
        """
        Create 2D (colormap) plots for different coordinate systems.

        Arguments:
            data: Dictionary containing the datasets for different coordinates.
            k_i: List of z indices to plot (for rt plots).
            z_c: List of z coordinates corresponding to k_i.
            magn: "intensity", "density", "fluence" or "radius".
            qua: Plot quality (low, medium, high).
            fig_path: Path to save figures instead of displaying them.
            str: Tuple specifying the stride for mesh plotting (faster rendering).
        """
        c = self.config.get_plot_config(magn, "2d")
        resolution = c.get("resolution", {})
        res = resolution.get(qua, resolution.get("medium", {}))
        st = stride or res.get("stride", (1, 1))

        # Plot each coordinate system in a separate figure
        for ii, jj in data.items():
            if ii == "rt" and k_i is not None:
                # Plot intensity or density for each z position
                y, x = self.get_2d_grid("rt")
                x_st = x[:: st[0], :: st[1]]
                y_st = y[:: st[0], :: st[1]]
                x_label = c["label"]["xt"]
                y_label = c["label"]["xr"]
                c_label = c["bar_label"]
                for idx in range(len(k_i)):
                    z_st = jj[:, idx, :][:: st[0], :: st[1]]

                    fig, ax = plt.subplots()
                    # Choose between these plotting methods:
                    # 1. Uncomment for linear scaling using `pcolormesh`
                    mesh = ax.pcolormesh(x_st, y_st, z_st, cmap=c["cmap"])
                    # 2. Uncomment for logarithmic scaling using `pcolormesh`
                    # mesh = ax.pcolormesh(
                    #    x_st,
                    #    y_st,
                    #    z_st / np.max(z_st),
                    #    norm=LogNorm(vmin=1e-6, vmax=1e-1)
                    #    cmap=c["cmap"],
                    # )
                    # 3. Uncomment for logarithmic scaling using `contourf`
                    # lvls = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
                    # mesh = ax.contourf(
                    #    x_st,
                    #    y_st,
                    #    z_st / np.max(z_st),
                    #    levels=lvls,
                    #    norm=LogNorm(vmin=1e-6, vmax=1e-1)
                    #    cmap=c["cmap"],
                    # )

                    fig.colorbar(mesh, ax=ax, label=c_label)
                    ax.set(xlabel=x_label, ylabel=y_label)

                    z_pos = z_c[idx]
                    z_pos_format = f"{z_pos:.2f}"
                    title = c["title"][ii].replace("{:.2f}", z_pos_format)
                    ax.set_title(title)

                    filename = f"2d_{magn}_{ii}_{z_pos:.2f}".replace(".", "-") + ".png"
                    self.save_or_display(fig, filename, fig_path, res["dpi"])

            elif ii == "zt":
                # Plot intensity or density on-axis
                x, y = self.get_2d_grid("zt")
                x_st = x[:: st[0], :: st[1]]
                y_st = y[:: st[0], :: st[1]]
                z_st = jj[:: st[0], :: st[1]]
                x_label = c["label"]["xz"]
                y_label = c["label"]["xt"]
                c_label = c["bar_label"]
                title = c["title"][ii]

                fig, ax = plt.subplots()
                mesh = ax.pcolormesh(x_st, y_st, z_st, cmap=c["cmap"])
                fig.colorbar(mesh, ax=ax, label=c_label)
                ax.set(xlabel=x_label, ylabel=y_label)
                ax.set_title(title)

                filename = f"2d_{magn}_{ii}.png"
                self.save_or_display(fig, filename, fig_path, res["dpi"])

            elif ii == "rz":
                # Plot intensity or density peak values
                # or fluence distribution
                y, x = self.get_2d_grid("rz")
                x_st = x[:: st[0], :: st[1]]
                y_st = y[:: st[0], :: st[1]]
                z_st = jj[:: st[0], :: st[1]]
                x_label = c["label"]["xz"]
                y_label = c["label"]["xr"]
                c_label = c["bar_label"]
                title = c["title"][ii]

                fig, ax = plt.subplots()
                mesh = ax.pcolormesh(x_st, y_st, z_st, cmap=c["cmap"])
                fig.colorbar(mesh, ax=ax, label=c_label)
                ax.set(xlabel=x_label, ylabel=y_label)
                ax.set_title(title)

                filename = f"2d_{magn}_{ii}.png"
                self.save_or_display(fig, filename, fig_path, res["dpi"])


class Plot3D(BasePlot):
    """Plotting class for 3D (surface) plots."""

    def render_3d_data(
        self,
        data,
        k_i,
        z_c=None,
        magn="intensity",
        qua="medium",
        fig_path=None,
        stride=None,
    ):
        """
        Create 3D (surface) plots for different coordinate systems.

        Arguments:
            data: Dictionary containing the datasets for different coordinates.
            k_i: List of z indices to plot (for rt plots).
            z_c: List of z coordinates corresponding to the k indices saved.
            magn: "intensity", "density", "fluence" or "radius".
            qua: Plot quality (low, medium, high).
            str: Tuple specifying the stride for mesh plotting (faster rendering).
            fig_path: Path to save figures instead of displaying them.
        """
        c = self.config.get_plot_config(magn, "3d")
        resolution = c.get("resolution", {})
        ang = c.get("camera_angle", {})
        res = resolution.get(qua, resolution.get("medium", {}))
        st = stride or res.get("stride", (1, 1))

        for ii, jj in data.items():
            aa = ang[ii]
            if ii == "rt" and k_i is not None:
                # Plot intensity or density for each z position
                x, y = self.get_2d_grid("rt")
                x_st = x[:: st[0], :: st[1]]
                y_st = y[:: st[0], :: st[1]]
                x_label = c["label"]["xr"]
                y_label = c["label"]["xt"]
                c_label = c["bar_label"]
                for idx in range(len(k_i)):
                    z_st = jj[:, idx, :][:: st[0], :: st[1]]

                    fig = plt.figure(dpi=res["dpi"])
                    ax = fig.add_subplot(projection="3d")
                    ax.plot_surface(
                        x_st,
                        y_st,
                        z_st,
                        cmap=c["cmap"],
                    )
                    ax.view_init(elev=aa["elevation"], azim=aa["azimuth"])
                    ax.set(
                        xlabel=x_label,
                        ylabel=y_label,
                        zlabel=c_label,
                    )

                    z_pos = z_c[idx]
                    z_pos_format = f"{z_pos:.2f}"
                    title = c["title"][ii].replace("{:.2f}", z_pos_format)
                    ax.set_title(title)

                    filename = f"3d_{magn}_{ii}_{z_pos:.2f}".replace(".", "-") + ".png"
                    self.save_or_display(fig, filename, fig_path, res["dpi"])

            elif ii == "zt":
                # Plot intensity or density on-axis
                x, y = self.get_2d_grid("zt")
                x_st = x[:: st[0], :: st[1]]
                y_st = y[:: st[0], :: st[1]]
                z_st = jj[:: st[0], :: st[1]]
                x_label = c["label"]["xz"]
                y_label = c["label"]["xt"]
                c_label = c["bar_label"]

                fig = plt.figure(dpi=res["dpi"])
                ax = fig.add_subplot(projection="3d")

                ax.plot_surface(
                    x_st,
                    y_st,
                    z_st,
                    cmap=c["cmap"],
                )
                ax.view_init(elev=aa["elevation"], azim=aa["azimuth"])
                ax.set(
                    xlabel=x_label,
                    ylabel=y_label,
                    zlabel=c_label,
                )
                ax.set_title(c["title"][ii])

                filename = f"3d_{magn}_{ii}.png"
                self.save_or_display(fig, filename, fig_path, res["dpi"])

            elif ii == "rz":
                # Plot intensity or density peak value
                x, y = self.get_2d_grid("rz")
                x_st = x[:: st[0], :: st[1]]
                y_st = y[:: st[0], :: st[1]]
                z_st = jj[:: st[0], :: st[1]]
                x_label = c["label"]["xr"]
                y_label = c["label"]["xz"]
                c_label = c["bar_label"]

                fig = plt.figure(dpi=res["dpi"])
                ax = fig.add_subplot(projection="3d")

                ax.plot_surface(
                    x_st,
                    y_st,
                    z_st,
                    cmap=c["cmap"],
                )
                ax.view_init(elev=aa["elevation"], azim=aa["azimuth"])
                ax.set(
                    xlabel=x_label,
                    ylabel=y_label,
                    zlabel=c_label,
                )
                ax.set_title(c["title"][ii])

                filename = f"3d_{magn}_{ii}.png"
                self.save_or_display(fig, filename, fig_path, res["dpi"])


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

    def get_radius_data(self, b_radius):
        """Compute beam radius data."""
        return self.base_plot.compute_radius(b_radius)

    def create_1d_plot(self, data, k_i=None, z_c=None, magn="intensity", fig_path=None):
        """Create line plots."""
        self.plot_1d.render_1d_data(data, k_i, z_c, magn, fig_path)

    def create_2d_plot(
        self,
        data,
        k_i=None,
        z_c=None,
        magn="intensity",
        qua="medium",
        fig_path=None,
        stride=None,
    ):
        """Create colormap plots."""
        self.plot_2d.render_2d_data(data, k_i, z_c, magn, qua, fig_path, stride)

    def create_3d_plot(
        self,
        data,
        k_i,
        z_c=None,
        magn="intensity",
        qua="medium",
        fig_path=None,
        stride=None,
    ):
        """Create 3D solution plots."""
        self.plot_3d.render_3d_data(data, k_i, z_c, magn, qua, fig_path, stride)


def parse_cli_options():
    """Parse and validate CLI options."""
    parser = argparse.ArgumentParser(
        description="Plot simulation data from HDF5 files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sim-path",
        default=sim_dir,
        help="Path to simulation data file.",
    )
    parser.add_argument(
        "--fig-path",
        default=fig_dir,
        help="Path to figures file.",
    )
    parser.add_argument(
        "--variables",
        default="intensity,density,fluence,radius",
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
        "--stride",
        default="1,1",
        help="Pixel stride (x,y format) for plotting 2D and 3D plots.",
    )
    parser.add_argument(
        "--radial-limit",
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

    if args.axial_range:
        try:
            z_min, z_max = map(float, args.axial_range.split(","))
            args.axial_range = (z_min, z_max)
        except ValueError:
            print(
                "Error: Axial range format must be 'min,max'. Using full range."
            )
            args.axial_range = None

    if args.time_range:
        try:
            t_min, t_max = map(float, args.time_range.split(","))
            args.time_range = (t_min, t_max)
        except ValueError:
            print(
                "Error: Time range format must be 'min,max'. Using full range."
            )
            args.time_range = None

    # Convert comma-separated strings to dictionaries for easier access
    stride_pair = [int(s) for s in args.stride.split(",")]
    args.variables = {dtype: True for dtype in args.variables.split(",")}
    args.dimensions = {ptype: True for ptype in args.dimensions.split(",")}
    args.stride = (stride_pair[0], stride_pair[1])

    return args


def setup_output_directory(args):
    """Setup environment for plotting based on arguments."""
    if args.fig_path:
        fig_path = args.fig_path
        fig_path.mkdir(parents=True, exist_ok=True)
        print(f"Saving figures to file: {fig_path.relative_to(base_dir)}")
    else:
        print("Displaying figures interactively.")


def load_simulation_data(directory):
    """Load simulation data from HDF5 files."""
    snapshots_path = directory / "acherus_snapshots.h5"
    diagnostics_path = directory / "acherus_diagnostics.h5"

    data = {}

    has_snapshots = snapshots_path.exists()
    has_diagnostics = diagnostics_path.exists()

    if has_snapshots:
        print(f"Loading data from file: {snapshots_path.relative_to(base_dir)}")

        with File(snapshots_path, "r") as f:
            data["k_i"] = np.array(f["snap_z_idx"])
            if "envelope_snapshot_rzt" in f:
                data["e_dist"] = np.array(f["envelope_snapshot_rzt"])
            if "density_snapshot_rzt" in f:
                data["elec_dist"] = np.array(f["density_snapshot_rzt"])

    if has_diagnostics:
        print(f"Loading data from file: {diagnostics_path.relative_to(base_dir)}")

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

            if "pulse" in f:
                pulse = f["pulse"]
                if "fluence_rz" in pulse:
                    data["b_fluence"] = np.array(pulse["fluence_rz"])
                if "radius_z" in pulse:
                    data["b_radius"] = np.array(pulse["radius_z"])

    if not has_diagnostics and not has_snapshots:
        raise FileNotFoundError(f"No files were found in {base_dir.name}")

    return data


def plot_request(
    data_type, plot_data, plot_types, plot, z_snap_idx, z_snap_coor, args
):
    """Generate requested physical magnitudes for the specified data."""

    if plot_types.get("1d", False):
        print(f"Generating 1D (line) plots for {data_type} ...")
        plot.create_1d_plot(
            plot_data,
            z_snap_idx,
            z_snap_coor,
            data_type,
            args.fig_path,
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

    z_snap_idx = data["k_i"]
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
        plot_data_radius = plot.get_radius_data(box.sliced_data["b_radius"])
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
    config = PlotConfiguration()
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
        if enabled and variable in ["intensity", "density", "fluence", "radius"]:
            process_simulation_data(variable, data, plot, box, args.dimensions, args)

    print("Plotting complete!")


if __name__ == "__main__":
    main()
