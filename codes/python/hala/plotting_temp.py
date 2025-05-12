"""
Python quick tool for plotting short NumPy arrays
saved while simulations are being executed
on-the-fly.
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def load_ofdiagnostic_data(file_path):
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

    im = axis.pcolormesh(r_grid_2d, z_grid_2d, peak_intensity, cmap="turbo")

    cbar = fig.colorbar(im, ax=axis)
    cbar.set_label("Intensity [W/cm2]")
    axis.set_ylabel("z [m]")
    axis.set_xlabel("r [mm]")
    axis.set_title("Peak intensity over time")

    save_path = save_dir / "peak_intensity.png"
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

    save_path = save_dir / "peak_intensity_r0.png"
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    """Main function."""
    base_dir = Path("./path_to_base_directory")

    sim_dir = base_dir / "data" / "sim_folder_name"
    fig_dir = base_dir / "figures" / "sim_folder_name"
    diag_file = sim_dir / "temp_ofdiagnostic.h5"
    print(f"Loading data from: {diag_file.relative_to(base_dir)}")

    save_dir = fig_dir / "figures_ofdiagnostic"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots to: {save_dir.relative_to(base_dir)}")

    if not diag_file.exists():
        print(f"File not found: {diag_file}")
        return

    data = load_ofdiagnostic_data(diag_file)
    if not data:
        return

    z_grid = plot_peak_intensity(data, save_dir)
    plot_on_axis_peak_intensity(data, z_grid, save_dir)


if __name__ == "__main__":
    main()
