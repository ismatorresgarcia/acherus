## Unreleased

- Added a production-ready CLI/config flow in `__main__` with TOML loading, optional `--output` override, and centralized solver initialization.
- Added `data_output_path` and `figure_output_path` support in `ConfigOptions`, plus a root `config.toml` template with examples and documented defaults.
- Reworked output path management with environment-aware helpers (`get_base_dir`, `get_user_paths`, etc.) and standardized data/figure save locations.
- Moved progressive monitoring writes into `OutputManager.monitoring_diagnostics`, removed legacy diagnostics-side monitoring code, and routed shared propagation through output-managed monitoring.
- Upgraded the `monitoring` CLI with `--config`, `--sim-path`, and `--fig-path`, including clear precedence rules and robust path display/creation behavior.
- Upgraded the `plotting` CLI and internals with config-aware path defaults, simpler defaults (`--dimensions 1d,2d`, `--scale linear`), direct `--dpi`, explicit log controls (`--log-y-range`, `--log-rt-levels`), `--camera-view`, and stricter validation.
- Simplified plotting color configuration: `--colors-1d` now accepts a strict RGB triplet in `[0,1]` (defaulting to Matplotlib blue), and `--colors-2d` accepts direct Matplotlib colormap names.
- Removed plot titles across 1D/2D/3D outputs and fixed plotting details (small-`z` filename precision, flat-ended 2D `rt` log colorbars, and beam-radius computation using full radial fluence with axial slicing).
- Improved plotting readability by standardizing naming (`magnitude`, `z_idx`, `z_coor`) and clarifying helper argument names.
- Improved plotting data-loading robustness by centralizing HDF5 group extraction helpers and keeping path/default handling consistent across config, CLI overrides, and generated output names.
- Unified FCN solvers into a single `FCN` class and SSCN solvers into a single `SSCN` class with initialization-time Raman/non-Raman strategy binding; removed duplicated `r*`/`nr*` solver files and updated exports/imports.
- Reduced FCN linear overhead by reusing persistent thread pools and replacing `Ax=b` solves with pre-factored tridiagonal LAPACK on the left-hand side plus pure-NumPy tridiagonal RHS assembly (instead of DIA sparse matrices and the `@` operator).
- Added optional radial PML-aware matrix assembly for both FCN and SSCN (`grid_parameters.PML`), with strict `pml_width` validation and interior-only radial coefficient construction for numerical safety near boundaries.
- Ported SSCN linear propagation to the same tridiagonal LAPACK workflow used in FCN (left-matrix factorization + explicit tridiagonal right-hand assembly), and aligned boundary enforcement so the last radial node remains consistent with Dirichlet conditions throughout propagation and diagnostics.
- Added configurable dispersion modeling via `[dispersion.*]`: `FULL` preserves Sellmeier-based behavior, while `PARTIAL` uses user-provided `k2` with optional `k3`, `k4`, and `k5` for truncated second- through fifth-order dispersion terms.
- Added optional electron recombination support in density evolution (`-k_rec * rho^2`) via medium key `recombination_rate`, dedicated density paths (`compute_density_nr`/`compute_density_r`), and initialization-time strategy binding in shared solver state.
- Updated `functions` exports to the explicit density API (`compute_density_nr`, `compute_density_r`, `compute_density_rk4`) and removed the ambiguous legacy `compute_density` alias.
- Added a numerical-robustness guard for low-intensity MPA division using thresholded element-wise division to avoid overflow/invalid contributions near zero intensity.
- Simplified the Raman kernel JIT decorator to `@njit(parallel=True)` and aligned helper naming/formatting cleanups across function modules.
- Applied minor consistency fixes, including duplicate ion-rate removal in SSCN paths, Sellmeier validity-range text cleanup, and style/readability normalization in touched modules.

# v0.8.0

## 🚀 New Features

- 🧩 Configuration

  - Replaced the `options` configuration module with a new `config` class that unifies all user-related options. Input options are now provided through a dictionary at the `__main__` entry.
  - Added a new `gpu` boolean variable as an entry in the `config` dictionary, in preparation for future GPU-acceleration.
  - Removed the `method_raman` string variable from the `config` dictionary, since only a single trapezoidal integration method is now used.


- 🧩 Plotting

  - Simplified and renamed some variables and classes for simplicity.
  - `Plot1D` now plots beam radius using a new `compute_radius()` function and beam fluence loaded `fluence_rz` data.
  - `Plot2D` now supports log-scale plotting for radial vs time images using the `countourf` function.
  - `Plot2D` now supports new colormap options via the third party library `cmasher`,
  which provides better color gradients and considers color vision deficiencies.
  - The `monitoring` module now imports input and output data paths from the `paths` submodule in `data`.

- 🧩 Mesh

  - Modified the radial grid index to go from 0 to N (instead of from 0 to N + 1) to matching boundary conditions.

- 🧩 Solvers

  - Rewired the `FSS` and `FCN` solvers to use two dedicated nonlinear functions `compute_nonlinear()`, depending on whether they include or not Raman molecular scattering: `r` indicates the solver includes the Raman molecular scattering term. `nr` indicates propagation without the Raman term. This avoids conditionals during the propagation loop and prevents introduction of zeros when the Raman is not computed.
  - Renamed the `base` submodule to `shared` to match the solver structure.
  - Added a new Adams-Bashforth two-step (AB2) integrator for nonlinear terms. All solvers now use AB2 to preserve Crank-Nicolson accuracy; the first propagation step is computed using the Forward Euler method.
  - `shared` module now initializes all arrays to zero to prevent use of uninitialized memory or empty values.
  - Refactored Crank-Nicolson matrix creation into a single `compute_matrices()` routine.
  - `fcn` solver now precomputes and stores the matrices in parallel using `ThreadPoolExecutor()` for faster execution.
  - `fcn` solver now uses `ThreadPoolExecutor()` for parallel computation across frequency slides in `compute_envelope()`.
  - Every single z-independent coefficient which depends on frequency, such as constants optical shock operators, is now precomputed and stored as a 1D array and passed to `compute_nonlinear()` functions for faster computation and reduced floating-point errors.

- 🧩 Physics

  - Added a new `keldysh` module replacing previous `ionization` approaches. It computes ionization rates over a desired intensity interval and provides an interpolating object to convert intensity arrays into ionization rates. Supports gaseous and condensed media, including the multiphoton limit and the full generalized Keldysh theory.
  - Added a new `medium` module storing medium properties from `config` and dispersive properties from `sellmeier`.
  - Refactored `laser` and `grid` modules to use input options from the `config` class.
  - Added molecular correction and electron momentum dependence for ionization rates of gases based on [Mishima et al.](https://link.aps.org/doi/10.1103/PhysRevA.66.033401) in the `keldysh` module.

- 🧩 Functions

  - Removed the `mathematics` folder; replaced with a `functions` folder containing all previous modules.
  - Moved `compute_radius()` to `plotting` module to increase execution speed by avoiding unnecessary computation inside the propagation loop and storage in diagnostics files.
    - Refactored `fourier` imports to use the `gpu` boolean from `computing_backend` option in `config`. CuPy's backend `cufft` is now used only `computin
    =gpu` and the library is available; otherwise SciPy `fft()` and `ifft()` is used.
  - `compute_raman()` now uses a second-order trapezoidal integration method for the Raman integral. Removed unused SciPy `solve_ivp()` routines.
  - `compute_nonlinear()` now uses z-independent frequency tables for faster broadcasting computations.
  - Optimized `compute_density()` and `compute_nonlinear()` with in-place operations using temporary buffer arrays `_nlin_tmp_t`, `_nlin_tmp_w`, and `dens_tmp_buf` to improve speed.
  - Replaced `RegularGridInterpolator` for intensity/ionization 2D arrays with SciPy's linear `make_interp_spline` interpolation along the time axis only.
  - Replaced `interp1d` in `keldysh` with SciPy's cubic `PchipInterpolator` for monotonic, shape-preserving interpolation.
  - Added `keldysh_rates` and `keldysh_sum` functions for truncated series computation of Keldysh ionization rates.
  - Added `sellmeier` functions providing complete Sellmeier formulas for [air](https://opg.optica.org/josa/abstract.cfm?uri=josa-62-8-958), [water](https://opg.optica.org/ao/abstract.cfm?uri=ao-17-18-2875), and [silica](https://opg.optica.org/josa/abstract.cfm?uri=josa-55-10-1205).

- 🧩 Data

  - `validate_step` in `diagnostics` now saves propagation results in case of failure. `Output_manager` is initialized in `__main__` and accessible from `shared` to save results when density or envelope values overflow.
  - Removed profiling capabilities from `diagnostics`, as they are no longer needed.

## 🏷️Other Features

- 🔁 Tests

  - Added a `tests` folder for development and quality control of modules and new features.

- 📚 Documentation and Examples

  - Added `docs` folder for [Read The Docs](https://acherus.readthedocs.io/en/latest/) powered by Sphinx! This hosts the official, updated documentation.
  - Acherus is now an official PyPI package!
  - Moved `README.md` to the repository root and updated badges.

- 🐍 Build and Compatibility

  - Renamed `main` entry as `__main__` to comply with Python CLI standards.
  - Removed `old` directory in favor of Git version control.

- 🎨 Code Style

  - Improved various helper module descriptions.
  - Configuration options now accept uppercase and lowercase strings.

- 🛡️Security

 - Added `SECURITY.md` file to the repository root.

## 🐛 Bug-fixing

  - Fixed `compute_envelope()` in `fcn` module: `mats_right` was incorrectly used in `solve_banded()` instead of `mats_left`.
  - Fixed `fourier` import error when CuPy was unavailable; previously, global backend `cufft` was used alongside SciPy `workers` option.
  - Fixed axis node conversion from float to integer in `init_grid_nodes` within the `plotting` module.
  - Fixed `--radial-limit` parser argument value not being converted to float.
  - Fixed duplicated axis data in `init_sliced_arrays` and `flip_radial_data` when plotting radially symmetric figures.
  - Fixed overflow in `snapshot_z_index`; data type changed from `int16` to `uint16`.

## 📝 Full Changelog
Full changelog: https://github.com/ismatorresgarcia/acherus/compare/v0.7.0...v0.8.0
