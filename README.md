[![Acherus logo](https://raw.githubusercontent.com/ismatorresgarcia/acherus/master/docs/images/acherus-logo-b.png)](https://github.com/ismatorresgarcia/acherus)

> Open-source **laser** pulse **filamentation** solver

[![Documentation Status](https://readthedocs.org/projects/acherus/badge/?version=latest)](https://acherus.readthedocs.io/en/latest/)
[![Tests Python 3-14](https://github.com/ismatorresgarcia/acherus/actions/workflows/nightly_tests_CPU_p3.14.yml/badge.svg)](https://github.com/ismatorresgarcia/acherus/actions/workflows/nightly_tests_CPU_p3.14.yml)
[![Codecov](https://codecov.io/gh/ismatorresgarcia/acherus/graph/badge.svg?token=7QPYJC23A0)](https://codecov.io/gh/ismatorresgarcia/acherus)
[![LoC](https://raw.githubusercontent.com/ismatorresgarcia/acherus/gh-pages/badge.svg)](https://github.com/ismatorresgarcia/acherus)
[![LoD](https://raw.githubusercontent.com/ismatorresgarcia/acherus/gh-pages/badge-docs.svg)](https://github.com/ismatorresgarcia/acherus)

![PyPI - Version](https://img.shields.io/pypi/v/acherus?style=flat-square&color=fuchsia)
![PyPI - Downloads](https://img.shields.io/pypi/dm/acherus)
[![PyPI - License](https://img.shields.io/pypi/l/acherus?style=flat-square&color=orange)](https://www.apache.org/licenses/LICENSE-2.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18864578.svg)](https://doi.org/10.5281/zenodo.18864578)

## About
Developed for **ultrashort laser filamentation modeling**, Acherus is a **(2+1)-spatiotemporal laser filamentation** package for Python. It solves the Nonlinear Envelope Equation (NEE) for ultrashort laser pulses in optically transparent and dispersive media&mdash;gases (air), liquids (water), and solids (silica)&mdash;using the pseudo-spectral Fourier-Crank-Nicolson (FCN) method. The package computes the temporal evolution of the plasma **electron density** generated during propagation, as well as the pulse **intensity**, **fluence**, and **width** spatiotemporal profiles. Acherus enables accurate prediction and reproduction of numerical or experimental ultrashort laser filamentation scenarios under moderate input powers, providing a versatile tool for modeling a wide range of nonlinear optical phenomena, including **supercontinuum generation**, **conical emission**, and **X-wave formation**, among others.

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
* For specific queries, please contact the developer 👩‍💻👋 through this [Email](mailto:i.torresg@upm.es).

## How to use
📘 Documentation, powered by `sphinx`, is available at https://acherus.readthedocs.io/

Look at 📁 `examples/` for different physical applications:
  * 1500 picosecond filamentation of an IR (1032 nm) pulse in air.
  * 100 femtosecond filamentation of an IR (800 nm) pulse in air.
  * 130 femtosecond filamentation of an IR (800 nm) pulse in water.
  * 90 femtosecond filamentation of a UV (400 nm) pulse in water.

## Installation
Acherus supports `Python 3.11 - 3.14` and may be installed using any `venv` or `conda` environment.

📘 Complete installation details can be checked in our [Installation Guide](https://acherus.readthedocs.io/en/latest/installation.html).

### Install with PyPI
To install `acherus`, simply run:
```bash
pip install acherus
```
To install `acherus` from the source, clone the repository and install it in *editable* mode:
```bash
git clone https://github.com/ismatorresgarcia/acherus.git
cd acherus
pip install -e .
```
📌 **Have a bug, feature request, or suggestion?** Open a [GitHub Issue](https://github.com/ismatorresgarcia/acherus/issues) so the community can track it.

👥 **Want to contribute?**  To merge your changes into `main`, create a **Pull Request (PR)** following this [PR template](https://github.com/ismatorresgarcia/acherus/tree/master/.github/pull_request_template.md).

## Motivation

🚀 **HASTUR Project** - *Harnessing Atmospheric Lasing: Towards Ultrasensitive Detection of Toxic Agents and Pathogens*

🏢 **ETSII-UPM. Instituto de Fusión Nuclear**

🌱 This project is part of a research carried out during the academic years 2024–2026 at the Universidad Politécnica de Madrid.

🎯 The main goal of this thesis is to study the detection of toxic agents and pathogens in the upper layers of the atmosphere by exploiting the presence of molecular nitrogen. These nitrogen molecules can act as an _active medium_, which amplifies radiation of a specific frequency when interacting with nitrogen, generating laser emission. The interaction between laser light and surrounding matter in the atmosphere can be used to determine the hidden presence of undesired contaminants and to study their physical properties.

🤔 So, what are the main activities or tasks carried out in this project?
  * 💻 Developing Maxwell-Bloch numerical codes, as well as studying the propagation of intense infrared plasma lasers through plasma channels&mdash;using Particle-in-Cell codes (PIC)&mdash;and atomic processes in plasmas.
  * 🧪 Developing preprocessing and postprocessing tools to study the data generated by the numerical codes.
  * 🌀 Using the previous numerical codes to study the amplification of ultraviolet (UV) radiation in nitrogen plasma filaments.

## Citing `acherus`
🔖 All `acherus` releases are linked automatically to a [Zenodo](https://doi.org/10.5281/zenodo.18864578) publication under a unique [DOI](https://doi.org/10.5281/zenodo.18864578). If you use `acherus` in your work, please star this [repository](https://github.com/ismatorresgarcia/acherus) so we can track adoption and improve the project. Additionally, if you use `acherus` in a scientific publication, please consider citing this work:

> \[1\] I. Torres García et al., "Acherus". Zenodo, 2026. https://doi.org/10.5281/zenodo.18864578

---
### Tests badges
[![Tests Python 3-11](https://github.com/ismatorresgarcia/acherus/actions/workflows/nightly_tests_CPU_p3.11.yml/badge.svg)](https://github.com/ismatorresgarcia/acherus/actions/workflows/nightly_tests_CPU_p3.11.yml)
[![Tests Python 3-12](https://github.com/ismatorresgarcia/acherus/actions/workflows/nightly_tests_CPU_p3.12.yml/badge.svg)](https://github.com/ismatorresgarcia/acherus/actions/workflows/nightly_tests_CPU_p3.12.yml)
[![Tests Python 3-13](https://github.com/ismatorresgarcia/acherus/actions/workflows/nightly_tests_CPU_p3.13.yml/badge.svg)](https://github.com/ismatorresgarcia/acherus/actions/workflows/nightly_tests_CPU_p3.13.yml)
[![Tests Python 3-14](https://github.com/ismatorresgarcia/acherus/actions/workflows/nightly_tests_CPU_p3.14.yml/badge.svg)](https://github.com/ismatorresgarcia/acherus/actions/workflows/nightly_tests_CPU_p3.14.yml)
