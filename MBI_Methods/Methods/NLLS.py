import numpy as np
from scipy.optimize import least_squares
from .Der_Based_Spline import Residual_Func

def NLLS_Fit(K_ini, E, T, Design_Blocks, Moments_Fit, Moments_Spline, Spline_der_bool, weights=None):

	print('##############################')
	print("Initial (K SINDY) [%.4f, %.4f, %.4f, %.4f, %.4g, %.4g, %.4f, %.4f, %.4f, %.4f]"%tuple(K_ini))

	if Spline_der_bool:
		print('[Config] Using Derivative Adjusted Splines')
	else:
		print('[Config] Using Regular Splines')

	arg_stack = (E, T, Design_Blocks, Moments_Fit, Moments_Spline, Spline_der_bool, weights)
	res_lsq_new = least_squares(Residual_Func, K_ini, bounds=(0, np.inf), args=arg_stack)

	
	if Spline_der_bool:
		print("Fitted (K NLLS (with Der))    [%.4f, %.4f, %.4f, %.4f, %.4g, %.4g, %.4f, %.4f, %.4f, %.4f]"%tuple(res_lsq_new.x))
	else:
		print("Fitted (K NLLS)    [%.4f, %.4f, %.4f, %.4f, %.4g, %.4g, %.4f, %.4f, %.4f, %.4f]"%tuple(res_lsq_new.x))
	print('##############################')
	return res_lsq_new.x, res_lsq_new.status