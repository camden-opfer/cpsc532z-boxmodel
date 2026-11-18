# cpsc532z-boxmodel

This repository contains a group of Python scripts used to create climate box model output, train a PINN to replicate it, and run that PINN. The goal of this is to create a fast model that can determine the habitability of exoplanets based on stellar radiation, planet size, land-to-ocean ratio, and atmospheric composition.

## Box Model

The box model is based on TODO

## PINN

The physics-informed neural network (PINN) was originally proposed in [[Raissi+ 2019]](https://doi.org/10.1016/j.jcp.2018.10.045). In this implementation, there are nodes representing the temperature and carbon content of the atmosphere, land, and ocean. Some asymmetric nodes, like moisture of the land surface and atmosphere but not the ocean, are also included.

The PINN is trained on input and output pairs of the original model created with randomized metaparameters (stellar radiation, planet size, etc.) and input data. This leads to a versatile PINN prepared to model exotic planets, not just Earth-like ones.

Once trained, the PINN can be run to equilibrium to determine habitability metrics like mean temperature and atmospheric humidity.