# 🔎 Introduction

## Overview of `Acherus` documentation

> Open-source **laser** pulse **filamentation** solver

`Acherus` is a **3D spatiotemporal filamentation code** that solves the Nonlinear Envelope Equation (NEE) for ultrashort and ultraintense cylindrically symmetric laser pulses propagating through optically transparent media using various numerical schemes. It computes the laser pulse **intensity** and **fluence distribution**, as well as its **radius**, together with the generated plasma **electron density**. It is capable of reproducing both numerical and experimental results in different scenarios, allowing the simulation of **condensed dielectric**, **liquid**, and **gaseous media**.

For example, **atmospheric** filamentation can be studied thanks to the interaction of laser pulses with nitrogen and oxygen diatomic molecules. Another common medium where filamentation has been reported experimentally is **water** (or in any other aqueous media), as well as dense dielectrics like **fused silica**, since this phenomenon was first discovered by [M. Hercher (1964)](https://link.springer.com/10.1007/978-0-387-34727-1) (see p. 280) when laser-induced damage tracks were found in glass during an experiment.

:::{toctree}
:maxdepth: 2
:caption: Table of Contents

self
installation.md
userguide.md
physics.md
releases.md
acherus.rst
:::