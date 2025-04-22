"""
Python tool for plotting NumPy arrays saved while
simulations are being executed.
"""

import os

import h5py
import matplotlib.pyplot as plt
import numpy as np


def load_diagnostic_data(file_path):
    """Load envelope data from HDF5 file."""
    data = {}

    with h5py.File(file_path, "r") as f:
        if "coordinates" in f:
            coords = f["coordinates"]
            data["r_min"] = coords["r_min"][()]
            data["r_max"] = coords["r_max"][()]
            data["z_min"] = coords["z_min"][()]
            data["z_max"] = coords["z_max"][()]

        if "envelope" in f:
            env = f["envelope"]
            if "peak_rz" in env:
                data["intensity"] = 1e-4 * np.abs(env["peak_rz"][()] ** 2)

        print(f"Successfully loaded data from {file_path}")

        return data


def plot_max_intensity(data, save_dir):
    """Plot intensity peak against r and z coordinates."""
    if "intensity" not in data:
        print("No intensity data available")
        return None

    intensity = data["intensity"]
    r_nodes, z_nodes = intensity.shape

    r_grid = np.linspace(1e3 * data["r_min"], 1e3 * data["r_max"], r_nodes)
    z_grid = np.linspace(data["z_min"], data["z_max"], z_nodes)

    r_grid_2d, z_grid_2d = np.meshgrid(r_grid, z_grid, indexing="ij")

    fig, axis = plt.subplots(figsize=(12, 6))

    im = axis.pcolormesh(r_grid_2d, z_grid_2d, intensity, cmap="plasma")

    cbar = fig.colorbar(im, ax=axis)
    cbar.set_label("Intensity [W/cm^2]")
    axis.set_xlabel("z [m]")
    axis.set_ylabel("r [mm]")
    axis.set_title("Peak evolution over time along z")

    save_path = os.path.join(save_dir, "peak_intensity.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {save_path}")

    return z_grid


def plot_on_axis_max_intensity(data, z_grid, save_dir):
    """Plot on-axis peak intensity against the z coordinate."""
    if "intensity" not in data:
        print("No intensity data available")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    peak_intensity = data["intensity"]
    on_axis_peak_intensity = peak_intensity[0, :]

    ax.plot(z_grid, on_axis_peak_intensity)

    ax.set_xlabel("z [m]")
    ax.set_ylabel("I(r=0,z) [W/cm2]")
    ax.set_title("Peak evolution over time on-axis along z")

    save_path = os.path.join(save_dir, "peak_intensity_on_axis.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    """Main function."""
    sim_dir = "./"
    diag_file = os.path.join(sim_dir, "temp_diagnostic.h5")

    save_dir = os.path.join(sim_dir, "temp_plots")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving plots to: {os.path.abspath(save_dir)}")

    if not os.path.exists(diag_file):
        print(f"File not found: {diag_file}")
        return

    data = load_diagnostic_data(diag_file)
    if not data:
        return

    z_grid = plot_max_intensity(data, save_dir)
    plot_on_axis_max_intensity(data, z_grid, save_dir)


if __name__ == "__main__":
    main()
