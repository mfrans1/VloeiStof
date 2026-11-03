import numba


@numba.njit
def project_velocity(
    u,
    v,
    p,
    boundary_mesh,
    dt,
    rho,
    dx,
    dy,
):
    Nx, Ny = u.shape

    # project velocity field
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            u[i, j] -= dt * (1.0 / rho) * (
                (p[i + 1, j] - p[i - 1, j]) / (2.0 * dx)
            )

            v[i, j] -= dt * (1.0 / rho) * (
                (p[i, j + 1] - p[i, j - 1]) / (2.0 * dy)
            )

    # apply boundary conditions

    # dirichlet for walls
    for j in range(Ny):
        if boundary_mesh[0, j]:
            u[0, j] = 0.0
        if boundary_mesh[-1, j]:
            u[-1, j] = 0.0

    for i in range(Nx):
        if boundary_mesh[i, 0]:
            v[i, 0] = 0.0
        if boundary_mesh[i, -1]:
            v[i, -1] = 0.0

    # neumann for open boundaries
    for j in range(Ny):
        if not boundary_mesh[0, j]:
            u[0, j] = u[1, j]

        if not boundary_mesh[-1, j]:
            u[-1, j] = u[-2, j]

        if not boundary_mesh[0, j]:
            v[0, j] = 0.0

        if not boundary_mesh[-1, j]:
            v[-1, j] = 0.0

    for i in range(Nx):
        if not boundary_mesh[i, 0]:
            v[i, 0] = v[i, 1]

        if not boundary_mesh[i, -1]:
            v[i, -1] = v[i, -2]

        if not boundary_mesh[i, 0]:
            u[i, 0] = 0.0

        if not boundary_mesh[i, -1]:
            u[i, -1] = 0.0

    return u, v

@numba.njit
def calculate_intermediate_velocity(
    u,
    v,
    u_old,
    v_old,
    dt,
    nu,
    dx,
    dy,
):
    Nx, Ny = u_old.shape

    dx2 = dx**2
    dy2 = dy**2

    # calculate u-component
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):

            # u-component advection
            if u_old[i, j] > 0.0:
                du_dx = (u_old[i, j] - u_old[i - 1, j]) / dx
            elif u_old[i, j] < 0.0:
                du_dx = (u_old[i + 1, j] - u_old[i, j]) / dx
            else:
                du_dx = 0.0

            if v_old[i, j] > 0.0:
                du_dy = (u_old[i, j] - u_old[i, j - 1]) / dy
            elif v_old[i, j] < 0.0:
                du_dy = (u_old[i, j + 1] - u_old[i, j]) / dy
            else:
                du_dy = 0.0

            u_advection = u_old[i, j] * du_dx + v_old[i, j] * du_dy

            # u-component diffusion
            u_diffusion = nu * (
                (u_old[i + 1, j] - 2.0 * u_old[i, j] + u_old[i - 1, j]) / dx2
                + (u_old[i, j + 1] - 2.0 * u_old[i, j] + u_old[i, j - 1]) / dy2
            )

            u[i, j] = u_old[i, j] + dt * (u_diffusion - u_advection)

    # calculate v-component
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):

            # v-component advection
            if u_old[i, j] > 0.0:
                dv_dx = (v_old[i, j] - v_old[i - 1, j]) / dx
            elif u_old[i, j] < 0.0:
                dv_dx = (v_old[i + 1, j] - v_old[i, j]) / dx
            else:
                dv_dx = 0.0

            if v_old[i, j] > 0.0:
                dv_dy = (v_old[i, j] - v_old[i, j - 1]) / dy
            elif v_old[i, j] < 0.0:
                dv_dy = (v_old[i, j + 1] - v_old[i, j]) / dy
            else:
                dv_dy = 0.0

            v_advection = u_old[i, j] * dv_dx + v_old[i, j] * dv_dy

            # v-component diffusion
            v_diffusion = nu * (
                (v_old[i + 1, j] - 2.0 * v_old[i, j] + v_old[i - 1, j]) / dx2
                + (v_old[i, j + 1] - 2.0 * v_old[i, j] + v_old[i, j - 1]) / dy2
            )

            v[i, j] = v_old[i, j] + dt * (v_diffusion - v_advection)

    return u, v