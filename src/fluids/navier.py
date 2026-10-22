import numpy as np
import scipy.linalg as sl
from numpy.typing import NDArray


class NavierStokesSolver:
    """Class for solving the 2D incompressible navier stokes equations for a given
    sized domain. Resolution defined by the user and informs the timestep using
    the courant friedrich lewy condition. 
    """
    def __init__(self,
        Pmax: float,
        p: NDArray,
        u:NDArray,
        v:NDArray,
        nu:float,
        rho:float,
        Lx:float,
        Ly: float,
        Nx:int,	
        Ny:int,	
        courant_max:float=0.005, 
        rtol:float=1e-5,
        logs:bool=False
    ):
        """Initialize the solver with parameters. """
        # setting external params
        self.rtol = rtol
        self.logs = logs

        # setting the grid variables
        self.p = p
        self.u = u
        self.v = v
        
        # setting fluid characteristics
        self.nu = nu
        self.rho = rho
        
        # setting problem dimensions
        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx	
        self.Ny = Ny

        # deriving resolution and 
        self.dx = Lx / (Nx - 1)
        self.dy = Ly / (Ny - 1)

        # calculate theoretical max velocity to inform timestep
        v_max = Pmax * max(Lx,Ly)**2 / (8*rho*nu*min(Lx,Ly))

        # set the timestep using the CFL condition based on v_max
        self.dt = courant_max * min(self.dx, self.dy) / v_max
        

    def pressure_poisson_jacobi(
        self,
        RHS: NDArray
    ) -> NDArray:
        """Solve the pressure poisson equation using jacobi iterative
        method, assuming a horizontal and vertical resolution of dx and dy
        respectively. Using right hand side given as RHS, and assuming
        NEUMANN conditions at the boundaries.

        Args:
            RHS (NDArray): Right hand side of the equation
        Returns:
            NDArray: Pressure field after the Jacobi iterations converge.
        """

        tol = 10.0 * self.rtol

        itrs = 0  # iterations counter

        p_old = np.copy(self.p)  # deep copy

        while tol > self.rtol:
            itrs += 1

            # swap the arrays
            temp = p_old
            p_old = self.p
            self.p = temp

            # discretized jacobi solver
            self.p[1:-1, 1:-1] = (
                1.0
                / (2.0 + 2.0 * (self.dx**2) / (self.dy**2))
                * (
                    p_old[2:, 1:-1]
                    + p_old[:-2, 1:-1]
                    + (p_old[1:-1, 2:] + p_old[1:-1, :-2]) * (self.dx**2) / (self.dy**2)
                    - (self.dx**2) * RHS[1:-1, 1:-1]
                )
            )

            # reinforce neumann conditions
            self.p[0, :] = self.p[1, :]  # left
            self.p[-1, :] = self.p[-2, :]  # right
            self.p[:, 0] = self.p[:, 1]  # bottom
            self.p[:, -1] = self.p[:, -2]  # top

            tol = sl.norm(self.p - p_old) / np.maximum(1.0e-10, sl.norm(self.p))

        if self.logs:
            print(f"PPE solve iterations = {itrs:4d}")

        return self.p

    def calculate_ppm_RHS_central(
        self,
        RHS: NDArray,
    ) -> NDArray:
        """Central difference scheme for the right
        hand side terms of the Navier Stokes equations,
        this becomes the RHS of the poisson jacobi
        method. Constructed from derivatives of velocity

        Args:
            RHS (NDArray): right hand side

        Returns:
            NDArray: spatial derivative RHS of velocities
        """
        RHS[1:-1, 1:-1] = (self.rho / self.dt) * (
            (self.u[2:, 1:-1] - self.u[:-2, 1:-1]) / (2 * self.dx)
            + (self.v[1:-1, 2:] - self.v[1:-1, :-2]) / (2 * self.dy)
        )

        return RHS

    def project_velocity(
            self) -> tuple[NDArray, NDArray]:
        """Velocity update to guarantee non-divergence with new
        pressure field calculated by pressure_poisson_jacobi.

        Returns:
            tuple[NDArray, NDArray]: u-velocity and v-velocity
        """
        self.u[1:-1, 1:-1] = self.u[1:-1, 1:-1] - self.dt * (1 / self.rho) * (
            (self.p[2:, 1:-1] - self.p[:-2, 1:-1]) / (2 * self.dx)
        )
        self.v[1:-1, 1:-1] = self.v[1:-1, 1:-1] - self.dt * (1 / self.rho) * (
            (self.p[1:-1, 2:] - self.p[1:-1, :-2]) / (2 * self.dy)
        )

        # apply boundary conditions
        self.u[0, :], self.u[-1, :] = 0, 0
        self.u[:, 0], self.u[:, -1] = 0, 0
        self.v[0, :], self.v[-1, :] = 0, 0
        self.v[:, 0], self.v[:, -1] = 0, 0

        return self.u, self.v


    def calculate_intermediate_velocity(
        self,
        u_old: NDArray,
        v_old: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """Computes the intermediate u- and v-velocity fields by advancing the
        previous velocity fields by one time step, accounting for nonlinear
        advection and viscous diffusion. This is the predictor step of a
        projection method; the resulting velocities are not necessarily
        divergence-free, hence needing project_velocity()

        Args:
            u_old (NDArray): old u-velocity
            v_old (NDArray): old v-velocity

        Returns:
            tuple[NDArray, NDArray]: updated u- and v-velocities
        """
        # calculate u-component
        u_advection = u_old[1:-1, 1:-1] * np.select(
            [u_old[1:-1, 1:-1] > 0, u_old[1:-1, 1:-1] < 0],
            [
                np.diff(u_old[:-1, 1:-1], n=1, axis=0) / self.dx,
                np.diff(u_old[1:, 1:-1], n=1, axis=0) / self.dx,
            ],
        ) + v_old[1:-1, 1:-1] * np.select(
            [v_old[1:-1, 1:-1] > 0, v_old[1:-1, 1:-1] < 0],
            [
                np.diff(u_old[1:-1, :-1], n=1, axis=1) / self.dy,
                np.diff(u_old[1:-1, 1:], n=1, axis=1) / self.dy,
            ],
        )
        u_diffusion = self.nu * (
            np.diff(u_old[:, 1:-1], n=2, axis=0) / (self.dx**2)
            + np.diff(u_old[1:-1, :], n=2, axis=1) / (self.dy**2)
        )

        self.u[1:-1, 1:-1] = u_old[1:-1, 1:-1] + self.dt * (u_diffusion - u_advection)

        # calculate v-component
        v_advection = u_old[1:-1, 1:-1] * np.select(
            [u_old[1:-1, 1:-1] > 0, u_old[1:-1, 1:-1] < 0],
            [
                np.diff(v_old[:-1, 1:-1], n=1, axis=0) / self.dx,
                np.diff(v_old[1:, 1:-1], n=1, axis=0) / self.dx,
            ],
        ) + v_old[1:-1, 1:-1] * np.select(
            [v_old[1:-1, 1:-1] > 0, v_old[1:-1, 1:-1] < 0],
            [
                np.diff(v_old[1:-1, :-1], n=1, axis=1) / self.dy,
                np.diff(v_old[1:-1, 1:], n=1, axis=1) / self.dy,
            ],
        )
        v_diffusion = self.nu * (
            np.diff(v_old[:, 1:-1], n=2, axis=0) / (self.dx**2)
            + np.diff(v_old[1:-1, :], n=2, axis=1) / (self.dy**2)
        )

        self.v[1:-1, 1:-1] = v_old[1:-1, 1:-1] + self.dt * (v_diffusion - v_advection)

        return self.u, self.v

    def solve_NS(
        self,
        t_end: float,
        output: bool = False,
        output_freq: int = 5,
        output_dir: str = "./output",
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Solve the incompressible Navier-Stokes equations

        Args:
            t_end (float): end of integration
            output (bool): output flag
            output_freq (int): output frequency
            output_dir (str): directory for output files
        Returns:
            tuple(NDArray, NDArray, NDArray): u-, v-velocity and pressure field
        """
        t = 0

        u_old = self.u.copy()
        v_old = self.v.copy()
        p_RHS = np.zeros_like(self.u)

        n_steps = int(np.ceil(t_end / self.dt))

        for step in range(1, n_steps + 1):
            t = step * self.dt

            if self.logs:
                print(f"\nTime = {t:.8f}")

            # calculate intermediate velocities
            self.u, self.v = self.calculate_intermediate_velocity(self.nu, self.u, self.v, u_old, v_old, self.dt, self.dx, self.dy)

            # calculate RHS for the pressure Poisson problem
            p_RHS = self.calculate_ppm_RHS_central(self.rho, self.u, self.v, p_RHS, self.dt, self.dx, self.dy)

            # compute pressure
            p = self.pressure_poisson_jacobi(self.p, self.dx, self.dy, p_RHS, self.rtol, self.logs)

            # project velocity
            self.u, self.v = self.project_velocity(self.rho, self.u, self.v, self.dt, self.dx, self.dy, self.p)

            if self.logs and output and step % output_freq == 0:
                print(f"norm(u) = {sl.norm(self.u):.8f}, norm(v) = {sl.norm(self.v):.8f}")
                print(
                    f"Courant number: {np.max(np.sqrt(self.u**2 + self.v**2)) * self.dt / min(self.dx, self.dy):.8f}"
                )

                # save current state to a file
                np.savez(f"{output_dir}/navier_{step:05d}.npz", u=self.u, v=self.v, p=self.p)

            # if the residuals in velocities between new and old are sufficiently small,
            # we can say we have reached steady state, as needed
            res_u = np.linalg.norm(self.u - u_old)/ (np.linalg.norm(self.u) + 1e-10)
            res_v = np.linalg.norm(self.v - v_old) / (np.linalg.norm(self.v) + 1e-10)
            if self.logs:
                print(f"velocity residual = {res_u + res_v}")

            if np.sqrt(res_u**2 + res_v**2) < 1e-7:
                print(f"Flow achieved steady state at t = {t:.2f}s")
                print(f"velocity residual = {res_u + res_v}")
                print(f"norm(u) = {sl.norm(self.u):.8f}, norm(v) = {sl.norm(self.v):.8f}")
                print(
                    f"Courant number: {np.max(np.sqrt(self.u**2 + self.v**2)) * self.dt / min(self.dx, self.dy):.8f}"
                )
                break

            # swap pointers without copying data
            temp = u_old
            u_old = self.u
            self.u = temp

            temp = v_old
            v_old = self.v
            self.v = temp
        return self.u, self.v, self.p
