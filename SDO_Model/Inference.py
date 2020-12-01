import numpy as np 
import pickle
import pdb
import time

import sys, os

np.set_printoptions(precision=2)

### Import data and all necessary arguments for SINDY and NLLS ###
from Data_Loader_PP import *

### Inference ####
print("================================================= NEW RUNS =================================================")
sub_list = np.array([1, 2, 4, 8, 10, 12, 16, 20, 25, 28, 32])
time_sindy = []
time_nlls = []
for n in range(len(sub_list)):
    sub_sample = sub_list[n]
    E = Mom_list[::sub_sample,:]
 
    T_fit = TT[::sub_sample]
    #####
    # SINDY CALL
    #####
    
    #pdb.set_trace()
    print("Subsample = %d"%(sub_sample))
    tick = time.time()
    blockPrint()
    K_sindy = SINDY(E, T_fit, Design_Blocks, keeps=Sindy_Moments_Fit)
    enablePrint()
    tock = time.time()
    
    t_sindy = tock - tick
    time_sindy.append(t_sindy)
    print("Sindy took", t_sindy)
    K_sindy = np.abs(K_sindy) #SINDY return positive values but sometimes there are -0.0000 which are treated as negative which is not accepted in the bounds of least square
    
    obj_sindy = {"Times_for_fit":T_fit,
                    "Parameters":K_sindy,
            "Subsample_for_fit" :sub_sample
                                }     
    
    pickle.dump(obj_sindy, open("./Data_final/K_sindy/Values/"+"K_%d.pck"%sub_sample, "wb"))
    #####
    # NLLS CALL
    #####
    if sub_sample>=16:
        k2b= open("./Data_final/K_nlls_der/Values/"+"K_%d.pck"%8, "rb") # initialize NLLS with the fit found with previous subsamples
        stuff2b = pickle.load(k2b)
        k2b.close()
        K_init = np.array(stuff2b["Parameters"])
        
        to_save = open("./Data_final/K_nlls_der/Values/"+"K_%d.pck"%sub_sample, "wb") # open("./Data_final/K_nlls_der/Values/"+"K_%d_start_true.pck"%sub_sample, "wb")
    else:
        K_init = K_sindy
        to_save = open("./Data_final/K_nlls_der/Values/"+"K_%d.pck"%sub_sample, "wb")
    
    tick = time.time()
    K_nlls_der = NLLS_Fit(K_init, E, T_fit, Design_Blocks, NLLS_Moments_Fit, Moments_Spline, Spline_der_bool)
    tock = time.time()
    t_nlls = tock - tick
    time_nlls.append(t_nlls)
    print("Nlls took", t_nlls)
    obj_nlls_with_der = {"Times_for_fit":T_fit,
                            "Parameters":K_nlls_der,
                    "Subsample_for_fit" :sub_sample,
                    "Starting_params" : K_init
                                   }
    
    pickle.dump(obj_nlls_with_der, to_save)
    #if n!=6:
    print("----------------------------------------------------------------------------------------------------")

Dt = np.array([TT[::sub][1] - TT[::sub][0] for sub in sub_list])
for s in range(len(Dt)):
    print("Dt = %.3f"%Dt[s], "Sindy run time = %.3f"%time_sindy[s], "Nlls run time = %.3f"%time_nlls[s])
