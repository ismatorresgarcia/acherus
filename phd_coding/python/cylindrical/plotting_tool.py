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
                data["intensity"] = np.abs(env["peak_rz"][()] ** 2) * 1e-4

        print(f"Successfully loaded data from {file_path}")

        return data


def plot_intensity(data, save_dir):
    """Plot 2D intensity distribution."""
    if "intensity" not in data:
        print("No intensity data available")
        return

    r_min = data["r_min"] * 1e3
    r_max = data["r_max"] * 1e3
    z_min = data["z_min"]
    z_max = data["z_max"]

    intensity = data["intensity"]
    r_nodes, z_nodes = intensity.shape

    r_array = np.linspace(r_min, r_max, r_nodes)
    z_array = np.linspace(z_min, z_max, z_nodes)

    r_grid_2d, z_grid_2d = np.meshgrid(r_array, z_array, indexing="ij")

    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.pcolormesh(r_grid_2d, z_grid_2d, intensity, cmap="plasma")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Intensity [W/cm^2]")
    ax.set_xlabel("z [m]")
    ax.set_ylabel("r [mm]")
    ax.set_title("Peak intensity")

    save_path = os.path.join(save_dir, "peak_intensity.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {save_path}")

    return z_array


def plot_on_axis_intensity(data, z_array, save_dir):
    """Plot on-axis intensity."""
    if "intensity" not in data:
        print("No intensity data available")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    intensity = data["intensity"]
    on_axis = intensity[0, :]

    ax.plot(z_array, on_axis)

    ax.set_xlabel("z [m]")
    ax.set_ylabel("I(r=0,z) [W/cm2]")
    ax.set_title("On-axis Peak Intensity")

    save_path = os.path.join(save_dir, "on-axis_intensity.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    """Main function."""
    sim_dir = "./"  # Change this to your simulation directory
    diag_file = os.path.join(sim_dir, "temp_diagnostic.h5")

    save_dir = os.path.join(sim_dir, "plots")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving plots to: {os.path.abspath(save_dir)}")

    if not os.path.exists(diag_file):
        print(f"File not found: {diag_file}")
        return

    data = load_diagnostic_data(diag_file)
    if not data:
        return

    z_array = plot_intensity(data, save_dir)
    plot_on_axis_intensity(data, z_array, save_dir)


if __name__ == "__main__":
    main()
