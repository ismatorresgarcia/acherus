"""Module for global variables used in the results subpackage."""

from pathlib import Path

base_dir = Path("./path_to_base_directory")

sim_dir = base_dir / "data" / "sim_folder_name"
Path(sim_dir).mkdir(parents=True, exist_ok=True)
