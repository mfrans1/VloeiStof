import numpy as np
from numpy.typing import NDArray
import scipy.linalg as sl

def pressure_poisson_jacobi(p: NDArray, 
							dx: float, 
							dy: float, 
							RHS: NDArray, 
							rtol: float = 1.e-5, 
							logs: bool = False) -> NDArray:
	"""Solve the pressure poisson equation (PPE) using jacobi iterative
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

def calculate_ppm_RHS_central():
	None

def project_velocity():
	None

def calculate_intermediate_velocity():
	None

def solve_NS():
	None
