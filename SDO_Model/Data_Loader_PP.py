import numpy as np 
import pickle
import pdb
import pylab as pl 

import sys, os
### Turn print on and off ###
# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

np.set_printoptions(precision=2)

K_true = np.array([4.0, 3.0, 0.0, 0.7, 0., 0., 0., 0., 0., 0.04, 1.25, 0.0, 0.0])

### Load the data ###
f = open("./Data_final/K_true/SSA/"+"SSA_true_pred_prey_N_%d.pck"%10000,"rb")
stuff = pickle.load(f)
f.close()

Mom_list = np.array(stuff["SSA_means"])
T = np.array(stuff["Times"])

Shift_data = 1 #0 To start processing data later
Cut_off = 130 #129

TT_0 = T[:][:Cut_off] # start simulation from t = 0 (needed in SSA_runs)

Mom_list = Mom_list[Shift_data:,:][:Cut_off,:]
TT = T[Shift_data:][:Cut_off]

### General moments setups ###
hash_lookUp = {(0,0):0, (1,0):1, (0,1):2, # 1st moments
    				(2,0):3, (1,1):4, (0,2):5, # 2nd moments
    				(3,0):6, (2,1):7, (1,2):8, (0,3):9, # 3rd moments
    				(4,0):10, (3,1):11, (2,2):12, (1,3):13, (0,4):14 # 4th moments
    				}

moms_lookUp_list = [(0, 0), (1, 0), (0, 1), # 1st moments
                        (2, 0), (1, 1), (0, 2), # 2nd moments
                        (3, 0), (2, 1), (1, 2), (0, 3), # 3rd moments
                        (4,0), (3,1), (2,2), (1,3), (0,4) # 4th moments
                         ]


### SSA arguments needed in Sim_SSA ###        

moms_lookUp = np.array(moms_lookUp_list) # needed in SSA_runs

Moments_labels = np.array(["1", "E_X", "E_Y", # order 1
                            "E_X_X", "E_X_Y", "E_Y_Y", # order 2
                            "E_X_X_X", "E_X_X_Y", "E_X_Y_Y", "E_Y_Y_Y", # order 3
                            "E_X_X_X_X", "E_X_X_X_Y", "E_X_X_Y_Y", "E_X_Y_Y_Y", "E_Y_Y_Y_Y" # order 4
                            ])
# Moments to collect
Moments_collect = np.array([True, True, True,
                            True, True, True, 
                            True, True, True, True,
                            False, False, False, False, False
                            ])

### SINDY arguments ###
from Methods.Sindy import SINDY
Sindy_Moments_Fit = np.array([True, True, True, 
                              True, True, True, 
                              True, True, True, True, 
                              False, False, False, False, False
                              ])                        

### NLLS arguments ###
from Methods.NLLS import NLLS_Fit
NLLS_Moments_Fit = np.array([True, True, True,
                             True, True, True,
                             False, False, False, False,
                             False, False, False, False, False
                             ])

Moments_Spline = np.array([False, False, False,
                           False, False, False,
                           True, True, True, True,
                           False, False, False, False, False
                           ])
Spline_der_bool = True

### Load the Feature Matricies
blockPrint()

from Methods.Two_Dim_Design_Blocks import generate_design_blocks as GDB
#Design_Blocks = np.array(Reaction_Mapping_Mat).astype(np.float)

inds = np.where(NLLS_Moments_Fit+Moments_Spline)[0]
moms_lookUp_list = [moms_lookUp_list[i] for i in inds]

Design_Blocks = GDB(hash_lookUp, moms_lookUp_list,Verbose=True)
enablePrint()


