# v0.7.0 Draft

In preparation!

## 🚀 New Features
 
- ⚙️ Configuration

  - Added a new `gpu` boolean variable as an entry to the dictionary from the `config` folder, in preparation for future GPU-acceleration option.
  - Removed `method_raman` string variable as an entry to the dictionary from the `config` folder, since a unique trapezoidal integration method is used instead.


- 🖼️ Plotting

  - Removed, renamed and simplified some variables and classes for simplicity.
  - `Plot2D` now supports using log-scale for plotting radial vs time images using the `countourf` function.
  - `Plot2D` now supports new colormaps through the third party library `cmasher`,
  which provides better color gradients and takes into consideration color vision deficiency users.
  - Modified the `monitoring` module to import the input/output data paths from the `paths.py` submodule in `data`.
 
 - 🧊 Mesh
 
   - Modified the radial grid index to go from 0 to N, instead of from 0 to N + 1 for matching boundary conditions. 

- ⚡ Solvers 

  - Modified `base` solver now initializes all arrays to zero values to make sure no trash memory values are being used. 
  - Refactored the creation of both Crank-Nicolson matrices to use one single `compute_matrices()` routine.
  - Modified `fcn` solver, now it precomputes and stores the matrices in parallel using the `ThreadPoolExecutor()` routine from `concurrent.futures` for faster execution.
  - Modified `fcn` solver's `compute_envelope()`, now uses `ThreadPoolExecutor()` for parallel computation of each frequency slice.

- 🧲 Physics

  - Renamed `materials` module to `media` module. The entry for average air at 775 nm has been removed.
  - `media` module now includes `constant_k1` for group velocity inverse, in preparation for full dispersion relations. All the associated variables have been renamed accordingly.
  - `optics` now uses the variable `wavenumber_0` for the gaussian pulse wave vector module in the chosen material.
  - New `ppt_rate` module in substitution of previous `ionization` module. It computes the PPT ionization rates for a given array `field_str` of peak intensities, instead of computing them for every element of the pulses `envelope` array.
 
- 🧠 Functions

  - New `functions` folder which contains all the individual modules in the previous `mathematics` folder for simplicity. The latter folder has therefore been removed.
  - Modified the import process in `fourier.py` to use `gpu` boolean variable from the `config_options()`. Now CuPy's backend `cufft` will be used only if the user selected `gpu` to be `True` and the library is available, otherwise falling back to SciPy's `fft()` and `ifft()` functions.
  - Modified `compute_raman()` function. Now, a second-order accurate trapezoidal integration method is used for evaluating the delayed molecular Raman scattering convolution integral. It relies on a simple loop in the time coordinate. Removed the remaining methods from SciPy's `solve_ivp()` routines, no longer needed. 

- 🛡️ Security & Documentation

  - Moved and updated `README.md` to parent folder repository.
  - Renamed and moved `codes` folder to parent repository. Now named `acherus` folder.

## 🏷️ Other tag features

- Removed `old` directory folder.
- ... sphinx! readthedocs
- ... pypi package!
 
## 🐛 Bug-fixing

  - Fixed error in `compute_envelope()` function from `fcn` module where the RHS matrix
  `mats_right` was used for solving the linear system Ax = b with `solve_banded()` 
  instead of `mats_left`, which is the correct matrix for the Crank-Nicolson scheme.
  - Fixed error in `fourier` module where the initial import statements failed if the 
  user did have `CuPy` library available. The error source was trying to use the global
  backend `cufft` with the `workers` option at the same time, which belongs exclusively
  to SciPy's functions.
