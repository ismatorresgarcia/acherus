"""
Main Module for Beam Propagation Simulation.

This module serves as the entry point for the beam propagation simulation.
It orchestrates:
    - Configuration setup for beam and grid parameters
    - Simulation execution using Crank-Nicolson method
    - Results visualization through intensity profile plots
"""

from .beam_config import BeamConfig
from .grid_config import GridConfig
from .plotting import BeamPlotter
from .simulation import GaussianBeamSimulation


def main():
    """
    Main function to run the beam propagation simulation.

    This function:
    1. Creates beam and grid configurations with default parameters
    2. Initializes and runs the Gaussian beam simulation
    3. Generates intensity profile plots comparing numerical and analytical solutions

    Example usage:
        $ python -m beam_simulation.main
    """
    # Create configurations
    beam_config = BeamConfig(wavelength=800e-9, waist=9e-3)
    grid_config = GridConfig(fin_radi=2e-2, radi_nodes=1000)

    # Create and run simulation
    simulation = GaussianBeamSimulation(beam_config, grid_config)
    simulation.run_simulation()

    # Plot results
    plotter = BeamPlotter(simulation)
    plotter.plot_intensity_profiles(simulation.envelope, simulation.envelope_s)


if __name__ == "__main__":
    main()
