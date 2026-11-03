import numpy as np
import scipy.linalg as sl
from numpy.typing import NDArray

import fluids.pressure_poisson as pp
import fluids.velocity as vel


class NavierStokesSolver:
    """Class for solving the 2D incompressible navier stokes equations for a given
    sized domain. The timestep is adapted every step from the advective (CFL)
    and viscous stability limits.
    """

    def __init__(
        self,
        boundary_mesh: NDArray,
        Pmax: float,
        p: NDArray,
        u: NDArray,
        v: NDArray,
        nu: float,
        rho: float,
        Lx: float,
        Ly: float,
        Nx: int,
        Ny: int,
        # standard args
        courant_max: float = 0.5,
        dt_max: float = 1e-3,
        viscous_safety: float = 0.9,
        dt_growth: float = 1.1,
        dt_floor: float = 1e-12,
        rtol: float = 1e-5,
        logs: bool = False,
    ):
        """Initialize the solver with parameters.

        Args:
            courant_max: target CFL number for the advective limit (< 1).
            dt_max: hard upper bound on the timestep.
            viscous_safety: fraction of the explicit-diffusion stability limit to use.
            dt_growth: max factor dt may grow by from one step to the next.
            dt_floor: abort if dt drops below this (the run is blowing up).
        """
        # setting external params
        self.rtol = rtol
        self.logs = logs

        # creating the boundary mesh
        self.boundary_mesh = boundary_mesh

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

        self.dx = Lx / (Nx - 1)
        self.dy = Ly / (Ny - 1)

        # timestep control parameters
        self.courant_max = courant_max
        self.dt_max = dt_max
        self.dt_growth = dt_growth
        self.dt_floor = dt_floor

        # viscous (explicit diffusion) limit: dt <= 1 / (2 nu (1/dx^2 + 1/dy^2)).
        self.dt_visc = viscous_safety / (
            2.0 * nu * (1.0 / self.dx**2 + 1.0 / self.dy**2)
        )

        # theoretical (Poiseuille) max velocity
        self.v_max = Pmax * (min(Lx, Ly) ** 2) / (8 * rho * nu * max(Lx, Ly))
        dt_adv0 = (
            courant_max * min(self.dx, self.dy) / self.v_max
            if self.v_max > 0
            else np.inf
        )
        self.dt = min(dt_adv0, self.dt_visc, dt_max)

        self._dt_stable = self.dt

    def compute_dt(self) -> float:
        """Stable timestep for the current velocity field.

        Uses the 2D advective CFL condition dt * (|u|/dx + |v|/dy) <= C,
        the explicit viscous limit, a hard cap, and a growth limiter.
        """
        adv_rate = np.max(np.abs(self.u)) / self.dx + np.max(np.abs(self.v)) / self.dy
        dt_adv = self.courant_max / adv_rate if adv_rate > 0 else np.inf

        dt = min(dt_adv, self.dt_visc, self.dt_max, self.dt_growth * self._dt_stable)

        if not np.isfinite(dt) or dt < self.dt_floor:
            raise RuntimeError(
                f"Timestep collapsed to {dt:.3e}; the solution is likely diverging "
                f"(max|u| = {np.max(np.abs(self.u)):.3e}, "
                f"max|v| = {np.max(np.abs(self.v)):.3e})."
            )
        return dt

    def calculate_courant_number(self) -> float:
        """Calculate the courant number for the current velocity field and dt.

        Returns:
            float: Courant number
        """
        return self.dt * (
            np.max(np.abs(self.u)) / self.dx + np.max(np.abs(self.v)) / self.dy
        )

    def pressure_poisson_jacobi(self, RHS: NDArray) -> NDArray:
        """Solve the pressure poisson equation using jacobi iterative
        method, assuming a horizontal and vertical resolution of dx and dy
        respectively. Using right hand side given as RHS, and assuming
        NEUMANN conditions at the boundaries.
        """

        max_iters = 100000

        self.p, itrs, tol = pp.jacobi_solver(
            self.p,
            RHS,
            self.boundary_mesh,
            self.dx,
            self.dy,
            self.rtol,
            max_iters,
        )

        if self.logs:
            print(f"PPE solve iterations = {itrs:4d}")

        if itrs == max_iters and tol > self.rtol:
            raise RuntimeError(
                f"Pressure solver did not converge after {max_iters} iterations. "
                f"Residual = {tol:.3e}"
            )

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

    def project_velocity(self) -> tuple[NDArray, NDArray]:
        """Velocity update to guarantee non-divergence with new
        pressure field calculated by pressure_poisson_jacobi.

        Returns:
            tuple[NDArray, NDArray]: u-velocity and v-velocity
        """
        self.u, self.v = vel.project_velocity(
            self.u,
            self.v,
            self.p,
            self.boundary_mesh,
            self.dt,
            self.rho,
            self.dx,
            self.dy,
        )

        # current max speed (diagnostic only; dt comes from compute_dt)
        self.v_max = float(np.max(np.sqrt(self.u**2 + self.v**2)))

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
        self.u, self.v = vel.calculate_intermediate_velocity(
            self.u,
            self.v,
            u_old,
            v_old,
            self.dt,
            self.nu,
            self.dx,
            self.dy,
        )

        return self.u, self.v

    def solve_NS(
        self,
        t_end: float,
        output: bool = False,
        output_dt: float = 0.1,
        output_dir: str = "./output",
        steady_tol: float = 1e-4,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Solve the incompressible Navier-Stokes equations with adaptive dt.

        Args:
            t_end (float): end of integration
            output (bool): output flag
            output_dt (float): simulation-time interval between saved snapshots
            output_dir (str): directory for output files
            steady_tol (float): steady-state tolerance on the relative velocity change
        Returns:
            tuple(NDArray, NDArray, NDArray): u-, v-velocity and pressure field
        """
        t = 0.0
        step = 0
        out_idx = 0
        next_output = output_dt

        u_old = self.u.copy()
        v_old = self.v.copy()
        p_RHS = np.zeros_like(self.u)

        du = self.u.copy()
        dv = self.v.copy()

        if output:
            # save initial state
            np.savez(
                f"{output_dir}/navier_{out_idx:05d}.npz",
                u=self.u, v=self.v, p=self.p, t=t, dt=self.dt, step=step,
            )
            out_idx += 1

        while t < t_end - 1e-12 * t_end:
            self.u, self.v = u_old, v_old
            dt = self.compute_dt()
            self._dt_stable = dt

            t_stop = min(next_output, t_end) if output else t_end
            remaining = t_stop - t
            if dt >= remaining:
                dt = remaining
            elif dt > 0.5 * remaining:
                dt = 0.5 * remaining

            self.dt = dt
            self.u, self.v = du, dv
            step += 1

            # calculate intermediate velocities
            self.u, self.v = self.calculate_intermediate_velocity(u_old, v_old)

            # calculate RHS for the pressure Poisson problem
            p_RHS = self.calculate_ppm_RHS_central(p_RHS)

            # compute pressure
            self.p = self.pressure_poisson_jacobi(p_RHS)

            # project velocity
            self.u, self.v = self.project_velocity()

            t += dt

            if self.logs:
                print(f"\nStep {step}, t = {t:.8f}, dt = {dt:.3e}")
                print(
                    f"norm(u) = {sl.norm(self.u):.8f}, norm(v) = {sl.norm(self.v):.8f}"
                )
                print(f"Courant number: {self.calculate_courant_number():.8f}")
                print(f"Maximum velocity: {self.v_max:.8f}")

            if output and t >= next_output - 1e-12 * max(1.0, next_output):
                # save current state to a file, with its simulation time
                np.savez(
                    f"{output_dir}/navier_{out_idx:05d}.npz",
                    u=self.u, v=self.v, p=self.p, t=t, dt=dt, step=step,
                )
                out_idx += 1
                next_output += output_dt

            res_u = sl.norm(self.u - u_old) / ((sl.norm(self.u) + 1e-10) * dt)
            res_v = sl.norm(self.v - v_old) / ((sl.norm(self.v) + 1e-10) * dt)
            res = np.sqrt(res_u**2 + res_v**2)

            if self.logs:
                print(f"velocity residual (per unit time) = {res:.3e}")

            if res < steady_tol:
                # exit early if the solution has reached a steady state
                print(f"Flow achieved steady state at t = {t:.4f}s (step {step})")
                print(f"velocity residual = {res:.3e}")
                print(
                    f"norm(u) = {sl.norm(self.u):.8f}, norm(v) = {sl.norm(self.v):.8f}"
                )
                print(f"Courant number: {self.calculate_courant_number():.8f}")
                return self.u, self.v, self.p

            u_old, du = self.u, u_old
            v_old, dv = self.v, v_old

        self.u, self.v = u_old, v_old
        return self.u, self.v, self.p