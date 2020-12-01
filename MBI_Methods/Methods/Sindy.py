import numpy as np
import time
import pdb

from .Splines import spline_Differencing as SD
from .Splines import Build_Design_Matrix as BDM
from .Quadratic_Programming import run_qp

def SINDY(E, T, Design_Blocks, fit = None, weights = None, sum_params = 1000):
	'''
	E = numpy array Time X Species
	'''
	tik = time.time()
	if fit is None:
		fit = np.array([True]*E.shape[1])

	derivatives = SD(E[:,fit],T)
	Design = BDM(E,Design_Blocks)

	if weights is not None:
		Design = np.einsum('tdk,dd-> tdk', Design, weights)
		derivatives = np.einsum('td,dd -> td', derivatives,weights)

	b = derivatives.flatten()
	A = Design.reshape(-1, Design.shape[-1])

	#pdb.set_trace()
	
	sol = run_qp(b, A, sum_params)
	tock = time.time()

	print('Sindy took %f'%(tock-tik))
	print(sol['status'])
	return np.array(sol['x']).flatten(),sol["status"]
