# 🔎 Overview

## Welcome to `acherus` documentation

> Open-source **laser** pulse **filamentation** (2+1)-dimensional solver

Developed for **ultrashort laser filamentation modeling**, Acherus is a **(2+1)-spatiotemporal laser filamentation** package for Python. It solves the Nonlinear Envelope Equation (NEE) for ultrashort laser pulses in optically transparent and dispersive media---gases (air), liquids (water), and solids (silica)---using the pseudo-spectral Fourier-Crank-Nicolson (FCN) method. The package computes the temporal evolution of the plasma **electron density** generated during propagation, as well as the pulse **intensity**, **fluence**, and **width** spatiotemporal profiles. Acherus enables accurate prediction and reproduction of numerical or experimental ultrashort laser filamentation scenarios under moderate input powers, providing a versatile tool for modeling a wide range of nonlinear optical phenomena, including **supercontinuum generation**, **conical emission**, and **X-wave formation**, among others.

🚀 Key features of `acherus`:
* Input beam models: Gaussian spatiotemporal profiles including chirped and lens-focused beams.
* Dispersion models: full or partial chromatic dispersion using Sellmeier semi-empirical equations.
* Ionization models: multiphoton ionization (MPI) or general Keldysh-PPT theory rate predictions.
* Python implementation with exposed API: access to modules, classes, functions, and core simulation objects.
* High performance matrix operations using `numpy` and `scipy` compiled libraries for fast, GIL-free computation.
* Built-in 1D-3D visualization tools for post-processing and on-the-fly simulation monitoring with Matplotlib.
* Optimized memory management and multithreading using `ThreadPoolExecutor` shared work pools.

🧩 Other features of `acherus`:
* Decoupled split-step Crank-Nicolson (SSCN) solver for simpler scenarios.
* Optimized output storage (`HDF5` format) and post-processing with `h5py` library.

📣 Tag and version updates are described in each `acherus` [GitHub Release](https://github.com/ismatorresgarcia/acherus/releases)
* The source code is available on the `acherus` [GitHub](https://github.com/ismatorresgarcia/acherus) repository.
* For specific queries, please contact the developer 👩‍💻👋 through this [Email](mailto:i.torresg@upm.es).

```{toctree}
:maxdepth: 2
:caption: Table of Contents

self
installation.md
userguide.md
physics.md
releases.md
acherus.rst
```
