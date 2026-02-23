"""Module for global variables used in the results subpackage."""

import os
from pathlib import Path

DEFAULT_BASE_DIR = Path("./results")


def set_base_dir(path) -> Path:
    """Set user choice base directory through environment variable."""
    base_path = Path(path)
    os.environ["ACHERUS_BASE_DIR"] = str(base_path)
    return base_path


def _resolve_base_dir(base_path=None) -> Path:
    """Resolve base directory from explicit path or environment."""
    if base_path is not None:
        return Path(base_path)
    return Path(os.environ.get("ACHERUS_BASE_DIR", str(DEFAULT_BASE_DIR)))


def get_base_dir(base_path=None) -> Path:
    """Get user base directory for data and figures."""
    return _resolve_base_dir(base_path)


def get_sim_dir(base_path=None) -> Path:
    """Get simulation data directory."""
    root = _resolve_base_dir(base_path)
    return root


def get_fig_dir(base_path=None) -> Path:
    """Get figures output directory."""
    root = _resolve_base_dir(base_path)
    return root / "figures"


def get_user_paths(base_path=None, create=False):
    """Get all user-selected output paths and optionally create them."""
    root = _resolve_base_dir(base_path)
    sim_path = get_sim_dir(root)
    fig_path = get_fig_dir(root)

    if create:
        sim_path.mkdir(parents=True, exist_ok=True)
        fig_path.mkdir(parents=True, exist_ok=True)

    return {
        "base_dir": root,
        "sim_dir": sim_path,
        "fig_dir": fig_path,
    }


def __getattr__(name):
    """Compatibility aliases for code importing base_dir/sim_dir/fig_dir."""
    if name == "base_dir":
        return get_base_dir()
    if name == "sim_dir":
        return get_sim_dir()
    if name == "fig_dir":
        return get_fig_dir()

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "DEFAULT_BASE_DIR",
    "set_base_dir",
    "get_base_dir",
    "get_sim_dir",
    "get_fig_dir",
    "get_user_paths",
]
