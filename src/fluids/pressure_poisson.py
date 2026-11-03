import numba
import numpy as np


# extract the numerics into its own function and use numba
@numba.njit
def jacobi_solver(
    p,
    RHS,
    boundary_mesh,
    dx,
    dy,
    rtol,
    max_iters,
):
    tol = np.inf
    itrs = 0

    p_old = p.copy()

    dx2 = dx**2
    dy2 = dy**2
    aspect = dx2 / dy2
    ratio = 1.0 / (2.0 + 2.0 * aspect)

    Nx, Ny = p.shape

    while tol > rtol and itrs < max_iters:
        itrs += 1

        p_old, p = p, p_old

        numerator = 0.0
        denominator = 0.0

        # discretized jacobi solver
        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                p[i, j] = ratio * (
                    p_old[i + 1, j]
                    + p_old[i - 1, j]
                    + aspect * (p_old[i, j + 1] + p_old[i, j - 1])
                    - dx2 * RHS[i, j]
                )

        # reinforce neumann conditions for walls, open otherwise
        for j in range(Ny):
            if boundary_mesh[0, j]:
                p[0, j] = p[1, j]

            if boundary_mesh[-1, j]:
                p[-1, j] = p[-2, j]

        for i in range(Nx):
            if boundary_mesh[i, 0]:
                p[i, 0] = p[i, 1]

            if boundary_mesh[i, -1]:
                p[i, -1] = p[i, -2]

        # calculate convergence residual
        for i in range(Nx):
            for j in range(Ny):
                diff = p[i, j] - p_old[i, j]

                numerator += diff**2
                denominator += p[i, j]**2

        tol = np.sqrt(numerator / max(denominator, 1e-20))

    return p, itrs, tol