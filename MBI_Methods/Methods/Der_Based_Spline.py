import numpy as np
from scipy.interpolate import CubicSpline
from functools import partial 
from scipy.integrate import solve_ivp

import pdb 

def der(t, E, K_fit, A_Blocks, B_Blocks, E_Splines):
	'''
	@brief Computes the derivative given the design blocks
	@param t time			scalar
	@param E moments 		array 		(#moments,)
	@param K_fit parameter		array		(#params,)
	@param A_Blocks 	 	dnarray		(#params, #moments, #moments)
	@param B_Blocks         dnarray     (#params, #moments, #highermoments)
	@param E_Splines 		func        t -> array(#highermoments,)
	'''

	A = np.sum(A_Blocks*K_fit[:,np.newaxis,np.newaxis],axis=0)

	B = np.sum(B_Blocks*K_fit[:,np.newaxis,np.newaxis], axis = 0)
	
	dEdt = A.dot(E) + B.dot(E_Splines(t))
	return dEdt



def Spline_With_Der(K, E, T, Design_Blocks, Moments_Spline):

	Design_4_Spline = np.sum(Design_Blocks[:,Moments_Spline[:Design_Blocks.shape[1]],:]*K[:,np.newaxis,np.newaxis], axis = 0)

	der_start = Design_4_Spline.dot(E[0,:].T)
	der_end = Design_4_Spline.dot(E[-1,:].T)

	der_based_Spline = CubicSpline(T, E[:,Moments_Spline].T,axis=1, bc_type=((1,der_start),(1,der_end)))

	return der_based_Spline

def Residual_Func(K, E, T, Design_Blocks, Moments_Fit, Moments_Spline, Spline_der_bool, Weights=None):

	##
	# Choosing the Spline Method
	##
	if Spline_der_bool:
		V = Spline_With_Der(K, E, T, Design_Blocks, Moments_Spline)
	else:
		V = CubicSpline(T, E[:,Moments_Spline].T,axis=1)

	##
	# Computing the Residuals
	##
	NumMoments = np.sum(Moments_Fit)
	NumOfJumps = len(T)-1 

	Residuals = np.zeros((NumOfJumps, NumMoments))  

	for i in range(NumOfJumps):
		# initialise the derivative
		dE_t = partial(der, K_fit = K, 
							A_Blocks = Design_Blocks[:,Moments_Fit[:Design_Blocks.shape[1]],:][:,:,Moments_Fit], 
							B_Blocks = Design_Blocks[:,Moments_Fit[:Design_Blocks.shape[1]],:][:,:,Moments_Spline], 
							E_Splines = V
						)
		# evolve the system
		Sol = solve_ivp(dE_t, np.array([T[i], T[i+1]]), E[i,Moments_Fit], t_eval = [T[i], T[i+1]])

		if Sol.status != 0:
			print('Integrator Failed')
			pdb.set_trace()

		if Weights is None:
			# calculate non weighted residual
			Residuals[i,:] = Sol.y[:, -1] - E[i+1,:][Moments_Fit]
		else:
			Residuals[i,:] = Weights.dot(Sol.y[:, -1] - E[i+1,:][Moments_Fit])


	return Residuals.flatten()




