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

	Returns:

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

def calculate_intermediate_velocity():
	None

def solve_NS():
	None
