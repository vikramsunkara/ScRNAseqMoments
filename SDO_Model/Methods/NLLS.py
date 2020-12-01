import numpy as np
from scipy.optimize import least_squares, minimize, fmin
from .Der_Based_Spline import Residual_Func


def To_Min(K, E, T, Design_Blocks, Moments_Fit, Moments_Spline, Spline_der_bool):
    return np.linalg.norm(Residual_Func(K, E, T, Design_Blocks, Moments_Fit, Moments_Spline, Spline_der_bool))


#epsilon = 0.5
#constraints_dict = [{"type": "ineq", "fun": lambda K: K[2] - K[10] - epsilon}, 
                   #{"type": "ineq", "fun": lambda K: K[3] - K[11] - epsilon},
                  #]

def NLLS_Fit(K_ini, E, T, Design_Blocks, Moments_Fit, Moments_Spline, Spline_der_bool):

	if Spline_der_bool:
		print('[Config] Using Derivative Adjusted Splines')
	else:
		print('[Config] Using Regular Splines')

	arg_stack = (E, T, Design_Blocks, Moments_Fit, Moments_Spline, Spline_der_bool)
	res_lsq_new = least_squares(Residual_Func, K_ini, bounds=(0, np.inf), args=arg_stack)
	#res_lsq_new = minimize(To_Min, K_ini, args = arg_stack, bounds=[(0, np.inf)]*len(K_ini), constraints=constraints_dict)
	
	#res_lsq_new = minimize(To_Min, K_ini, args = arg_stack, bounds=[(0, np.inf)]*len(K_ini))#, options = {"maxiter": 1e4})
	
	print('##############################')
	print("Initial (-----) [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]"%tuple(K_ini))
	if Spline_der_bool:
		print("Fitted (K NLLS (with Der))    [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]"%tuple(res_lsq_new.x))
	else:
		print("Fitted (K NLLS)    [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]"%tuple(res_lsq_new.x))
	#print("Tests ----- ", "(k_3 - k_11, epsilon) = ", (res_lsq_new.x[2] - res_lsq_new.x[10], epsilon), "(k_4 - k_12, epsilon) = ", (res_lsq_new.x[3] - res_lsq_new.x[11], epsilon))

	print('##############################')
	return res_lsq_new.x
