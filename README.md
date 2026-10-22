![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

# VloeiStof

*Vloeistof* is Dutch for “fluid.”

VloeiStof is a Python package for simulating two-dimensional incompressible fluid flow. It is an ongoing numerical-modelling project focused on implementing and exploring finite-difference methods for the Navier–Stokes equations. This is still very much a work in progress, check out the issues or releases tab for the most up to date information on the status of the project. 

## Overview

The solver numerically integrates the incompressible Navier–Stokes equations, coupling the momentum equations with a pressure Poisson equation to enforce mass conservation. Projecting it following the standard Chorin projection method. All grids are collocated, not staggered, so you'll likely get some artifacts when you run simulations. Don't worry, follow a staggered grid like Arakawa C-grid is on the list of stuff to do...

Current features include:

- Finite-difference discretisation of the governing equations.
- Iterative solution of the pressure Poisson equation using Jacobi iteration.
- Pressure–velocity projection to enforce incompressibility.
- Configurable domain dimensions, grid resolution, and fluid properties.
- Simulation output and visualisation using NumPy and Matplotlib.

## Installation

Requires Python 3.11 or later.

```bash
git clone <repository-url>
cd VloeiStof
python -m pip install -e .
```

## Usage

For best usage, checkout the `showcase.ipynb` example jupyter notebook in `examples/`.

## Testing

Run the test suite with:

```bash
pytest
```

Check code quality with Ruff:

```bash
ruff check .
ruff format --check .
```

## License

This project is released under the MIT License.

