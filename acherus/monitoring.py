"""
Python tool for monitoring some Acherus
data chosen to be saved while a simulation
is still running and did not finish.
"""

import matplotlib.pyplot as plt
import numpy as np
from h5py import File

from .data.paths import base_dir, fig_dir, sim_dir


def load_monitoring_data(file_path):
    """Load data from HDF5 file."""
    data = {}

    with File(file_path, "r") as f:
        if "coordinates" in f:
            coor = f["coordinates"]
            data["r_min"] = coor["r_min"][()]
            data["r_max"] = coor["r_max"][()]
            data["z_min"] = coor["z_min"][()]
            data["z_max"] = coor["z_max"][()]

        if "envelope" in f:
            envelope = f["envelope"]
            if "peak_rz" in envelope:
                data["peak_intensity"] = np.abs(envelope["peak_rz"][()] ** 2)

        return data


def plot_peak_intensity(data, save_dir):
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

    save_path = save_dir / "peak_intensity.png"
    plt.savefig(save_path)
    plt.close(fig)

    return z_grid


def plot_on_axis_peak_intensity(data, z_grid, save_dir):
    """Plot on-axis peak intensity over time."""
    fig, ax = plt.subplots()

    peak_intensity = data["peak_intensity"]
    peak_intensity_r0 = peak_intensity[0, :]

    ax.plot(z_grid, peak_intensity_r0)

    ax.set(xlabel="z [m]", ylabel="I(r=0,z) [W/m2]")
    ax.set_title("Peak on-axis intensity over time")

    save_path = save_dir / "peak_intensity_r0.png"
    plt.savefig(save_path)
    plt.close(fig)


def main():
    """Main function."""
    mon_file = sim_dir / "acherus_monitoring.h5"
    print(f"Loading data from file: {mon_file.relative_to(base_dir)}")

    save_file = fig_dir / "acherus_monitoring"
    save_file.mkdir(parents=True, exist_ok=True)
    print(f"Saving figures to file: {save_file.relative_to(base_dir)}")

    if not mon_file.exists():
        print(f"File not found: {mon_file}")
        return

    data = load_monitoring_data(mon_file)
    if not data:
        return

    z_grid = plot_peak_intensity(data, save_file)
    plot_on_axis_peak_intensity(data, z_grid, save_file)


if __name__ == "__main__":
    main()
