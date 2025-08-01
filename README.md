![image](./docs/images/acherus-logo-b.png)

> Open-source **Acherus** code for laser filamentation

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0) ![Lines of Code](https://tokei.rs/b1/github/ismatorresgarcia/acherus) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15924923.svg)](https://doi.org/10.5281/zenodo.15924923)

🚀 **HASTUR Project** - *Harnessing Atmospheric Lasing: Towards Ultrasensitive Detection of Toxic Agents and Pathogens*

🏢 **ETSII-UPM. Instituto de Fusión Nuclear _Guillermo Velarde_**

## Overview
🌱 This project is part of my PhD thesis during the academic years 2024–2027 at the Polytechnic University of Madrid.

🎯 The main goal of this thesis is to study the detection of toxic agents and pathogens in the upper layers of the atmosphere by exploiting the presence of molecular nitrogen. These nitrogen molecules can act as an _active medium_, which amplifies radiation of a specific frequency when interacting with nitrogen, generating laser emission. The interaction between laser light and surrounding matter in the atmosphere can be used to determine the hidden presence of undesired contaminants and to study their physical properties.

🤔 So, what are the main activities or tasks carried out in this project?

- 💻 Developing Maxwell-Bloch numerical codes, as well as studying the propagation of intense infrared plasma lasers through plasma channels &mdash;using Particle-in-Cell codes (PIC)&mdash; and atomic processes in plasmas
- 🧪 Developing preprocessing and postprocessing tools to study the data generated by the numerical codes
- 🌀 Using the previous numerical codes to study the amplification of ultraviolet (UV) radiation in nitrogen plasma filaments

## About
`Acherus` is a **3D spatiotemporal filamentation code** that solves the Nonlinear Envelope Equation (NEE) for ultrashort and ultraintense cylindrically symmetric laser pulses propagating through optically transparent media using various numerical schemes. It computes the laser pulse **intensity and fluence distributions, as well as its radius**, together with the generated plasma **electron density**. It is capable of reproducing both numerical and experimental results in different scenarios, allowing the simulation of **condensed dielectric, liquid, and gaseous media**.

For example, **atmospheric** filamentation can be studied thanks to the interaction of laser pulses with nitrogen and oxygen diatomic molecules. Another common medium where filamentation has been reported experimentally is **water** (or in any other aqueous media), as well as dense dielectrics like **fused silica**, since this phenomenon was first discovered by M. Hercher (1964) when laser-induced damage tracks were found in glass during an experiment.

## Citing `Acherus`
🔖 All `Acherus` releases will be linked to a [Zenodo](https://doi.org/10.5281/zenodo.15924923) publication automatically under a unique [DOI](https://doi.org/10.5281/zenodo.15924923). If you were using `Acherus` in your scientific research, please cite this work:
> \[1\] I. Torres García et al., "Acherus". Zenodo, 2025. https://doi.org/10.5281/zenodo.15924923
