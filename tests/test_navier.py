import numpy as np
import pytest

from fluids.navier import NavierStokesSolver


@pytest.fixture
def solver():
    """Create a solver with a square grid and stationary fluid."""
    nx, ny = 21, 21
    lx, ly = 2.0, 2.0

    u = np.zeros((nx, ny))
    v = np.zeros((nx, ny))
    p = np.zeros((nx, ny))

    return NavierStokesSolver(
        Pmax=1.0,
        p=p,
        u=u,
        v=v,
        nu=1e-3,
        rho=1000.0,
        Lx=lx,
        Ly=ly,
        Nx=nx,
        Ny=ny,
    )


def test_initialization(solver):
    """Check that the solver stores its parameters and grid spacing."""
    assert solver.Nx == 21
    assert solver.Ny == 21
    assert solver.dx == pytest.approx(0.1)
    assert solver.dy == pytest.approx(0.1)
    assert solver.dt > 0.0


def test_zero_rhs_preserves_zero_pressure(solver):
    """A zero RHS and zero initial pressure should return zero pressure."""
    rhs = np.zeros_like(solver.p)

    p_computed = solver.pressure_poisson_jacobi(rhs)

    assert np.allclose(p_computed, 0.0)


def test_constant_pressure_is_preserved(solver):
    """A constant pressure field should remain constant for zero RHS."""
    solver.p.fill(5.0)
    rhs = np.zeros_like(solver.p)

    p_computed = solver.pressure_poisson_jacobi(rhs)

    assert np.allclose(p_computed, 5.0)


def test_courant_number_for_stationary_fluid(solver):
    """A stationary velocity field should have zero Courant number."""
    assert solver.calculate_courant_number() == pytest.approx(0.0)


def test_zero_velocity_remains_zero(solver):
    """Zero velocity should remain zero under the intermediate step."""
    u_old = solver.u.copy()
    v_old = solver.v.copy()

    u_new, v_new = solver.calculate_intermediate_velocity(u_old, v_old)

    assert np.allclose(u_new, 0.0)
    assert np.allclose(v_new, 0.0)


def test_zero_velocity_projection(solver):
    """Projecting a stationary velocity field should leave it stationary."""
    u_new, v_new = solver.project_velocity()

    assert np.allclose(u_new, 0.0)
    assert np.allclose(v_new, 0.0)


def test_pressure_rhs_for_stationary_fluid(solver):
    """A stationary velocity field should produce a zero pressure RHS."""
    rhs = np.zeros_like(solver.p)

    rhs_computed = solver.calculate_ppm_RHS_central(rhs)

    assert np.allclose(rhs_computed, 0.0)


def test_boundary_velocities_are_zero_after_projection(solver):
    """Projection should enforce zero velocity at all domain boundaries."""
    solver.u[5, 5] = 1.0
    solver.v[5, 5] = -1.0

    u_new, v_new = solver.project_velocity()

    for field in (u_new, v_new):
        assert np.allclose(field[0, :], 0.0)
        assert np.allclose(field[-1, :], 0.0)
        assert np.allclose(field[:, 0], 0.0)
        assert np.allclose(field[:, -1], 0.0)
