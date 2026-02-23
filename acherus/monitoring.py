"""
Python tool for monitoring some Acherus
data chosen to be saved while a simulation
is still running and did not finish.
"""

import argparse
import tomllib
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from h5py import File

from .data.paths import get_user_paths


MONITORING_FILENAME = "acherus_monitoring.h5"
MONITORING_FIGURE_SUBDIR = "acherus_monitoring"


def parse_cli_options():
    """Parse monitoring options."""
    user_paths = get_user_paths(create=False)

    parser = argparse.ArgumentParser(
        description="Plot monitoring data from HDF5 file.",
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
        help="Path where monitoring HDF5 data is stored.",
    )
    parser.add_argument(
        "--fig-path",
        type=Path,
        default=None,
        help="Path where monitoring figures will be saved.",
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

    return args


def _display_path(path: Path, root: Path) -> Path:
    """Return path relative to root when possible."""
    try:
        return path.relative_to(root)
    except ValueError:
        return path


def load_monitoring_data(file_path: Path) -> dict[str, Any]:
    """Load data from HDF5 file."""
    data = {}

    with File(file_path, "r") as f:
        if "coordinates" in f:
            coordinates = f["coordinates"]
            data["r_min"] = coordinates["r_min"][()]
            data["r_max"] = coordinates["r_max"][()]
            data["z_min"] = coordinates["z_min"][()]
            data["z_max"] = coordinates["z_max"][()]

        if "envelope" in f:
            envelope = f["envelope"]
            if "peak_rz" in envelope:
                data["peak_intensity"] = np.abs(envelope["peak_rz"][()] ** 2)

    return data


def _save_figure(fig, output_path: Path):
    """Save a Matplotlib figure and close it."""
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_peak_intensity(data: dict[str, Any], output_dir: Path):
    """Plot peak intensity over time."""
    if "peak_intensity" not in data:
        print("No intensity data available")
        return None

    peak_intensity = data["peak_intensity"]
    nr, nz = peak_intensity.shape

    r_grid = np.linspace(data["r_min"], data["r_max"], nr)
    z_grid = np.linspace(data["z_min"], data["z_max"], nz)

    r_grid_2d, z_grid_2d = np.meshgrid(r_grid, z_grid, indexing="ij")

    fig, ax = plt.subplots()

    mesh = ax.pcolormesh(z_grid_2d, r_grid_2d, peak_intensity)

    fig.colorbar(mesh, ax=ax, label="Intensity [W/m2]")
    ax.set(xlabel="z [m]", ylabel="r [m]")
    ax.set_title("Peak intensity over time")

    output_path = output_dir / "peak_intensity.png"
    _save_figure(fig, output_path)

    return z_grid


def plot_on_axis_peak_intensity(
    data: dict[str, Any], z_grid: np.ndarray, output_dir: Path
):
    """Plot on-axis peak intensity over time."""
    fig, ax = plt.subplots()

    peak_intensity = data["peak_intensity"]
    on_axis_peak_intensity = peak_intensity[0, :]

    ax.plot(z_grid, on_axis_peak_intensity)

    ax.set(xlabel="z [m]", ylabel="I(r=0,z) [W/m2]")
    ax.set_title("Peak on-axis intensity over time")

    output_path = output_dir / "peak_intensity_r0.png"
    _save_figure(fig, output_path)


def main():
    """Main function."""
    args = parse_cli_options()
    sim_dir = args.sim_path
    fig_dir = args.fig_path
    user_paths = get_user_paths(base_path=sim_dir, create=False)
    base_dir = user_paths["base_dir"]

    monitoring_file = sim_dir / MONITORING_FILENAME
    print(f"Loading data from file: {_display_path(monitoring_file, base_dir)}")

    output_dir = fig_dir / MONITORING_FIGURE_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving figures to file: {_display_path(output_dir, base_dir)}")

    if not monitoring_file.exists():
        print(f"File not found: {monitoring_file}")
        return

    data = load_monitoring_data(monitoring_file)
    if not data:
        return

    z_grid = plot_peak_intensity(data, output_dir)
    plot_on_axis_peak_intensity(data, z_grid, output_dir)


if __name__ == "__main__":
    main()
