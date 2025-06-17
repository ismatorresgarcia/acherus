"""
Python tool for monitoring some ACHERUS
data chosen to be saved while a simulation
is still running and did not finish.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from h5py import File


def load_monitoring_data(file_path):
    """Load data from HDF5 file."""
    data = {}  # Empty dictionary to extract the data

    with File(file_path, "r") as f:
        if "coordinates" in f:
            coords = f["coordinates"]
            data["r_min"] = coords["r_min"][()]
            data["r_max"] = coords["r_max"][()]
            data["z_min"] = coords["z_min"][()]
            data["z_max"] = coords["z_max"][()]

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
    r_nodes, z_nodes = peak_intensity.shape

    r_grid = np.linspace(data["r_min"], data["r_max"], r_nodes)
    z_grid = np.linspace(data["z_min"], data["z_max"], z_nodes)

    r_grid_2d, z_grid_2d = np.meshgrid(r_grid, z_grid, indexing="ij")

    fig, axis = plt.subplots()

    im = axis.pcolormesh(z_grid_2d, r_grid_2d, peak_intensity)

    cbar = fig.colorbar(im, ax=axis)
    cbar.set_label("Intensity [W/m2]")
    axis.set_xlabel("z [m]")
    axis.set_ylabel("r [m]")
    axis.set_title("Peak intensity over time")

    save_path = save_dir / "peak_intensity.png"
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")

    return z_grid


def plot_on_axis_peak_intensity(data, z_grid, save_dir):
    """Plot on-axis peak intensity over time."""
    fig, ax = plt.subplots()

    peak_intensity = data["peak_intensity"]
    peak_intensity_r0 = peak_intensity[0, :]

    ax.plot(z_grid, peak_intensity_r0)

    ax.set_xlabel("z [m]")
    ax.set_ylabel("I(r=0,z) [W/m2]")
    ax.set_title("Peak on-axis intensity over time")

    save_path = save_dir / "peak_intensity_r0.png"
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    """Main function."""
    base_dir = Path("./path_to_base_directory")

    sim_dir = base_dir / "sim_par_folder" / "data" / "sim_folder_name"
    fig_dir = base_dir / "sim_par_folder" / "figures" / "sim_folder_name"
    diag_file = sim_dir / "acherus_monitoring.h5"
    print(f"Loading data from: {diag_file.relative_to(base_dir)}")

    save_dir = fig_dir / "acherus_monitoring"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots to: {save_dir.relative_to(base_dir)}")

    if not diag_file.exists():
        print(f"File not found: {diag_file}")
        return

    data = load_monitoring_data(diag_file)
    if not data:
        return

    z_grid = plot_peak_intensity(data, save_dir)
    plot_on_axis_peak_intensity(data, z_grid, save_dir)


if __name__ == "__main__":
    main()
