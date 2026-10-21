import numpy as np
from fluids.navier import pressure_poisson_jacobi

def test_zero_rhs():
    """A zero RHS and zero initial pressure should return zero pressure."""
    Nx, Ny = 101, 101
    dx, dy = 0.1, 0.1

    p = np.zeros((Nx, Ny))
    RHS = np.zeros((Nx, Ny))

    p_computed = pressure_poisson_jacobi(p, dx, dy, RHS)

    assert np.allclose(p_computed, 0.0), (
        "Pressure should remain zero when the RHS is zero."
    )