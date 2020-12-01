import pandas as pand
import numpy as np
import pdb
from functools import partial 
from scipy.integrate import solve_ivp, odeint
import time
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares, basinhopping

import matplotlib.pyplot as plt
from Two_Dim_NonLinear_Sys_Design_Blocks_5_Reaction import *
from Residual_and_Jac_Funs import *
from Evolve_2Dim import *


def NLLS_Fit(K, pt_index, spline_der_bool,  Design_Blocks, B_Blocks, Add_B_Blocks, title = ""):
    Index_time = pt_index
    Ebar_new = np.array([E_X_X[Index_time], E_X_Y[Index_time], E_Y_Y[Index_time]])
    
    V_full = CubicSpline(Time[Index_time], Ebar_new, axis = 1)
         
    E_0_list_new = np.array([[1, E_X[j], E_Y[j]] for j in Index_time[:-1]])
    E_1_list_new = np.array([[1, E_X[j], E_Y[j]] for j in Index_time[1:]])
    
    
    arg_stack = (E_0_list_new, E_1_list_new, Time[Index_time], Index_time, Design_Blocks, B_Blocks, Add_B_Blocks, V_full, spline_der_bool)
    res_lsq_new = least_squares(Residual_Func, K, bounds=(0, np.inf), args=arg_stack)
    
    print(title)
    print("Initial (K SINDY) [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]"%tuple(K))
    print("Fitted (K NLLS) [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]"%tuple(res_lsq_new.x))
    print("")
    return res_lsq_new.x



### Test runs ###

# Dependence of first momements on first moments
Design_Blocks2 = Design_Blocks[:, 0:3, 0:3]

# Dependence of first momements on second moments
B_Blocks2 = Design_Blocks[:, 0:3, 3:6]

# Dependence of second moments on first moments 
Add_Design_Blocks = Design_Blocks[:, 3:6, 0:3]

# Dependence of second moments on second moments
Add_B_Blocks_self = Design_Blocks[:, 3:6, 3:6]

# Dependence of second moments on third moments
Add_B_Blocks_higher = B_Blocks[:, 3:6, :]

# Combine the Add_B_Blocks to reflect the Dependence of sencond moments on [E_X_X, E_X_X, E_X_Y, E_Y_Y, E_X_X_Y, E_Y_Y_X, E_X_X_X, E_Y_Y_Y]
Add_B_Blocks = np.zeros((13, 3, 3+3+4))

for i in range(13):
    Add_B_Blocks[i, :, 0:3] = Add_Design_Blocks[i, :, :]
    Add_B_Blocks[i, :, 3:6] = Add_B_Blocks_self[i, :, :]
    Add_B_Blocks[i, :, 6:10] = Add_B_Blocks_higher[i, :, :]

#pdb.set_trace()

K_0 = np.array([13.6242,  0.0000,  0.1445,  20.7022,  0.4549,  0.0000,  0.0000,  0.0044,  0.0000,  0.0006,  0.0000,  20.1233,  0.0000]) # Found by SINDY
Shift_data = 10 # To start NNLS processing from a future time step
Cut_off = 141
Sub_Samples = 1 # used in K_Evolve
Sub_Samples_nlls = 20 # used in NLLS

used_time_index = np.arange(0,len(Time),1,dtype=np.int)[Shift_data:][:Cut_off][::Sub_Samples_nlls]

spline_der_bool = True
whichK_init = "initial"
whichK_fit = "fit"

K_fit = NLLS_Fit(K_0,used_time_index, spline_der_bool, Design_Blocks2, B_Blocks2, Add_B_Blocks, "NLLS with derivative")
pdb.set_trace()
# Evolve for the whole time using all the data sub sampled
#K_Evolve(K_fit, Sub_Samples, "Evolve with K fitted (spline with derivatives)",3, Draw, whichK_fit, True, used_time_index)
#K_Evolve(K_0, Sub_Samples, "Evolve with K Sindy (spline with derivatives)", 4, Draw, whichK_init, True, used_time_index)

# True reaction params
#K_real = np.array([4, 3.0, 0, 0.7, 0, 0, 0, 0, 0, 0.04, 1.25, 0, 0])


