import numpy as np
import time
import pdb

from .Splines import spline_Differencing as SD
from .Splines import Build_Design_Matrix as BDM
from .Quadratic_Programming import run_qp

def SINDY(E, T, Design_Blocks, keeps = None, sum_params = 1000):
	'''
	E = numpy array Time X Species
	'''
	tik = time.time()
	if keeps is None:
		keeps = np.array([True]*E.shape[1])

	derivatives = SD(E[:,keeps],T)
	Design = BDM(E,Design_Blocks)
	b = derivatives.flatten()
	A = Design.reshape(-1, Design.shape[-1])
	
	sol = run_qp(b, A, sum_params)
	tock = time.time()

	print('Sindy took %f'%(tock-tik))
	print(sol['status'])
	return np.array(sol['x']).flatten()