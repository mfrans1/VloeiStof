import numpy as np
from numpy.typing import NDArray
import scipy.linalg as sl

def pressure_poisson_jacobi(p: NDArray, 
							dx: float, 
							dy: float, 
							RHS: NDArray, 
							rtol: float = 1.e-5, 
							logs: bool = False) -> NDArray:
	"""Solve the pressure poisson equation using jacobi iterative
	method, assuming a horizontal and vertical resolution of dx and dy 
	respectively. Using right hand side given as RHS, and assuming 
	NEUMANN conditions at the boundaries.

    Args:
        p (NDArray): Initial pressure field.
        dx (float): Grid spacing in the x direction.
        dy (float): Grid spacing in the y direction.
        rhs (NDArray): Right-hand side of the Poisson equation.
        rtol (float): Relative convergence tolerance. Defaults to 1e-5.
        logs (bool): Whether to log solver progress. Defaults to False.

    Returns:
        NDArray: Pressure field after the Jacobi iterations converge.
    """
	
	tol = 10.*rtol

	itrs = 0 # iterations counter
	
	p_old = np.copy(p) # deep copy

	while tol > rtol:
		itrs += 1

		# swap the arrays
		temp = p_old
		p_old = p
		p = temp

		# discretized jacobi solver
		p[1:-1, 1:-1] = 1.0/(2.0+2.0*(dx**2)/(dy**2))*(p_old[2:, 1:-1] + p_old[:-2, 1:-1] +
										(p_old[1:-1, 2:] + p_old[1:-1,:-2])*(dx**2)/(dy**2)
										- (dx**2)*RHS[1:-1, 1:-1])

		# reinforce neumann conditions
		p[0, :] = p[1, :] # left
		p[-1, :] = p[-2, :] # right
		p[:, 0] = p[:, 1] # bottom
		p[:, -1] = p[:, -2] # top

		tol = sl.norm(p - p_old)/np.maximum(1.e-10, sl.norm(p))

	if logs:
		print("PPE solve iterations = {:4d}".format(itrs))

	return p

def calculate_ppm_RHS_central(rho: float, 
                              u: NDArray, 
							  v: NDArray, 
							  RHS: NDArray, 
							  dt: float, 
							  dx: float, 
							  dy: float)->NDArray:
	"""Central difference scheme for the right
	hand side terms of the Navier Stokes equations,
	this becomes the RHS of the poisson jacobi
	method. Constructed from derivatives of velocity

	Args:
		rho (float): fluid density
		u (NDArray): fluid u-velocity
		v (NDArray): fluid v-velocity
		RHS (NDArray): right hand side
		dt (float): time step
		dx (float): resolution x-dimension
		dy (float): resolution y-dimension

	Returns:
		NDArray: spatial derivative RHS of velocities
	"""
	RHS[1:-1, 1:-1] = rho * (
		(1.0/dt) * ((u[2:, 1:-1] - u[:-2, 1:-1]) / (2*dx)
		            (v[1:-1, 2:] - v[1:-1, :-2]) / (2*dy))
	)

	return RHS

def project_velocity(rho: float,
                     u: NDArray, 
					 v: NDArray,
					 dt: float,
					 dx: float,
					 dy: float,
					 p: NDArray)->tuple[NDArray, NDArray]:
	"""Velocity update to guarantee non-divergence with new 
	pressure field calculated by pressure_poisson_jacobi.

	Args:
		rho (float): fluid density
		u (NDArray): grid of u-velocity
		v (NDArray): grid of v-velocity
		dt (float): timestep
		dx (float): x-dimension resolution
		dy (float): y-dimension resolution
		p (NDArray): grid of pressure

	Returns:
		tuple[NDArray, NDArray]: u-velocity and v-velocity
	"""
	u[1:-1, 1:-1] = u[1:-1, 1:-1] - dt * (1/rho) * (
		(p[2:, 1:-1] - p[:-2, 1:-1])/(2*dx)
	)
	v[1:-1, 1:-1] = v[1:-1, 1:-1] - dt * (1/rho) * (
		(p[1:-1, 2:] - p[1:-1, :-2])/(2*dy)
	)

	# apply boundary conditions
	u[0, :], u[-1, :] = 0, 0
	u[:, 0], u[:, -1] = 0, 0
	v[0, :], v[-1, :] = 0, 0
	v[:, 0], v[:, -1] = 0, 0

	return u, v

def calculate_intermediate_velocity(nu: float,
                                    u: NDArray,
									v: NDArray,
									u_old: NDArray,
									v_old: NDArray,
									dt: float,
									dx: float,
									dy: float)-> tuple[NDArray, NDArray]:
	"""Computes the intermediate u- and v-velocity fields by advancing the
    previous velocity fields by one time step, accounting for nonlinear
    advection and viscous diffusion. This is the predictor step of a
    projection method; the resulting velocities are not necessarily
    divergence-free, hence needing project_velocity()

	Args:
		nu (float): diffusivity constant
		u (NDArray): u-velocity
		v (NDArray): v-velocity
		u_old (NDArray): old u-velocity
		v_old (NDArray): old v-velocity
		dt (float): timestep
		dx (float): x-dimension resolution
		dy (float): y-dimension resolution

	Returns:
		tuple[NDArray, NDArray]: updated u- and v-velocities
	"""
	# calculate u-component
	u_advection = u_old[1:-1, 1:-1] * np.select(
		[u_old[1:-1, 1:-1] > 0, u_old[1:-1, 1:-1] < 0],
		[np.diff(u_old[:-1, 1:-1], n=1, axis=0)/dx, 
		 np.diff(u_old[1:, 1:-1], n=1, axis=0)/dx]
		) + \
		v_old[1:-1, 1:-1] * np.select(
			[v_old[1:-1, 1:-1] > 0, v_old[1:-1, 1:-1] < 0],
			[np.diff(u_old[1:-1, :-1], n=1, axis=1)/dy, 
			 np.diff(u_old[1:-1, 1:], n=1, axis=1)/dy]
		)
	u_diffusion = nu * (
		np.diff(u_old[:, 1:-1], n=2, axis=0)/(dx**2) + \
		np.diff(u_old[1:-1, :], n=2, axis=1)/(dy**2)
		)

	u[1:-1, 1:-1] = u_old[1:-1, 1:-1] + dt * (u_diffusion - u_advection)

	# calculate v-component
	v_advection = u_old[1:-1, 1:-1] * np.select(
		[u_old[1:-1, 1:-1]>0, u_old[1:-1, 1:-1] < 0],
		[np.diff(v_old[:-1, 1:-1], n=1, axis=0)/dx,
		 np.diff(v_old[1:, 1:-1], n=1, axis=0)/dx]
		) + \
		v_old[1:-1, 1:-1] * np.select(
			[v_old[1:-1, 1:-1] > 0, v_old[1:-1, 1:-1] < 0],
			[np.diff(v_old[1:-1, :-1], n=1, axis=1)/dy,
			 np.diff(v_old[1:-1, 1:], n=1, axis=1)/dy]
		)
	v_diffusion = nu * (
		np.diff(v_old[:, 1:-1], n=2, axis=0)/(dx**2) + \
		np.diff(v_old[1:-1, :], n=2, axis=1)/(dy**2)
		)

	v[1:-1, 1:-1] = v_old[1:-1, 1:-1] + dt * (v_diffusion - v_advection)

	return u, v

def solve_NS():
	None
