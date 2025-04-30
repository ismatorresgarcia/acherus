"""
Python quick tool for plotting short NumPy arrays
saved while simulations are being executed
on-the-fly.
"""

import os

import h5py
import matplotlib.pyplot as plt
import numpy as np


def load_simulation_data_on_the_fly(file_path):
    """Load envelope data from HDF5 file."""
    data = {}  # Empty dictionary to extract the data

    with h5py.File(file_path, "r") as f:
        if "coordinates" in f:
            coords = f["coordinates"]
            data["r_min"] = coords["r_min"][()]
            data["r_max"] = coords["r_max"][()]
            data["z_min"] = coords["z_min"][()]
            data["z_max"] = coords["z_max"][()]

        if "envelope" in f:
            envelope = f["envelope"]
            if "peak_rz" in envelope:
                data["peak_intensity"] = 1e-4 * np.abs(envelope["peak_rz"][()] ** 2)

        return data


def plot_peak_intensity(data, save_dir):
    """Plot peak intensity over time values."""
    if "peak_intensity" not in data:
        print("No intensity data available")
        return None

    peak_intensity = data["peak_intensity"]
    r_nodes, z_nodes = peak_intensity.shape

    r_grid = np.linspace(1e3 * data["r_min"], 1e3 * data["r_max"], r_nodes)
    z_grid = np.linspace(data["z_min"], data["z_max"], z_nodes)

    r_grid_2d, z_grid_2d = np.meshgrid(r_grid, z_grid, indexing="ij")

    fig, axis = plt.subplots(figsize=(12, 6))

    im = axis.pcolormesh(r_grid_2d, z_grid_2d, peak_intensity.T, cmap="turbo")

    cbar = fig.colorbar(im, ax=axis)
    cbar.set_label("Intensity [W/cm2]")
    axis.set_xlabel("z [m]")
    axis.set_ylabel("r [mm]")
    axis.set_title("Peak intensity over time")

    save_path = os.path.join(save_dir, "peak_intensity.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")

    return z_grid


def plot_on_axis_peak_intensity(data, z_grid, save_dir):
    """Plot on-axis peak intensity over time values."""
    fig, ax = plt.subplots(figsize=(10, 6))

    peak_intensity = data["peak_intensity"]
    peak_intensity_r0 = peak_intensity[0, :]

    ax.plot(z_grid, peak_intensity_r0)

    ax.set_xlabel("z [m]")
    ax.set_ylabel("I(r=0,z) [W/cm2]")
    ax.set_title("Peak on-axis intensity over time")

    save_path = os.path.join(save_dir, "peak_intensity_r0.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    """Main function."""
    sim_dir = "./"
    fig_dir = "./"
    diag_file = os.path.join(sim_dir, "temp_diagnostic.h5")

    save_dir = os.path.join(fig_dir, "figures_temp")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving plots to: {os.path.abspath(save_dir)}")

    if not os.path.exists(diag_file):
        print(f"File not found: {diag_file}")
        return

    data = load_simulation_data_on_the_fly(diag_file)
    if not data:
        return

    z_grid = plot_peak_intensity(data, save_dir)
    plot_on_axis_peak_intensity(data, z_grid, save_dir)


if __name__ == "__main__":
    main()
