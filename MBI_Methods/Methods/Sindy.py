import numpy as np
import time
#import pdb

from .Splines import spline_Differencing as SD
from .Splines import Build_Design_Matrix as BDM
from .Quadratic_Programming import run_qp

def SINDY(E, T, Design_Blocks, fit = None, weights = None, sum_params = 1000):
	'''
    @brief perform a linear least square minimization for the linear MBI method
    @param K_ini : initial guess                                    array           (#params)
    @param E : moments                                              array           (#moments,)
    @param T : time values of snapshots                             array           (#moments,)
    @param Design_Blocks : A block from the desing matrix           ndarray         (#params, #moments, #moments + #highermoments) 
    @param Moments_Fit : boolean moments                            array           (#moments + #highermoments)
    @param Moments_Spline : boolean higher order moments            array           (#moments + #highermoments)
    @param Spline_der_bool: boolean                                 boolean                           
    @param weights : weight considered on residuals of moments      ndarray         (#moments, #moments)
        
	'''
	tik = time.time()
	if fit is None:
		fit = np.array([True]*E.shape[1])

	derivatives = SD(E[:,fit],T)
	Design = BDM(E,Design_Blocks)

	if weights is not None:
		Design = np.einsum('tdk,dd-> tdk', Design, weights)
		derivatives = np.einsum('td,dd -> td', derivatives, weights)

	b = derivatives.flatten()
	A = Design.reshape(-1, Design.shape[-1])

	#pdb.set_trace()
	'''
    Using quadratic programing to find the solution
    '''
	sol = run_qp(b, A, sum_params)
	tock = time.time()

	print('Sindy took %f'%(tock-tik))
	print(sol['status'])
	return np.array(sol['x']).flatten(),sol["status"]
