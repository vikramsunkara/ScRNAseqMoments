import numpy as np 
import pickle
import pdb

import sys, os
### Turn print on and off ###
# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
# Restore
def enablePrint():
    sys.stdout = sys.__stdout__  
 
### DATA relevant to this SSA ###
K_true = np.array([4.0, 3.0, 0.0, 0.7, 0., 0., 0., 0., 0., 0.04, 1.25, 0.0, 0.0])

"""
f = open("./Data/PP_model_Centres_And_Times_v2.pck","rb")
stuff = pickle.load(f)
f.close()

TT = np.array(stuff["Times"])

dt = TT[1]-TT[0]
TT = np.arange(TT[0], 25 + dt, dt)
"""

TT = np.arange(0.0, 25 + 0.05, 0.05)
 
### SSA arguments ###        
from Sim_SSA import * # Labels of moments
#hash_lookUp = {(0,0):0, (1,0):1, (0,1):2, 
#				(2,0):3, (1,1):4, (0,2):5, 
#				(3,0):6, (2,1):7, (1,2):8, (0,3):9,
#                               (4,0):10, (3,1):11, (2,2):12, (1,3):13, (0,4):14
#				} # order of the moments 

moms_lookUp = np.array([(0, 0), (1, 0), (0, 1), 
                        (2, 0), (1, 1), (0, 2), 
                        (3, 0), (2, 1), (1, 2), (0, 3), 
                        (4,0), (3,1), (2,2), (1,3), (0,4)])

Moments_labels_bis = np.array(["1", "E_X", "E_Y", # order 1
                            "E_X_X", "E_X_Y", "E_Y_Y", # order 2
                            "E_X_X_X", "E_X_X_Y", "E_X_Y_Y", "E_Y_Y_Y", # order 3
                            "E_X_X_X_X", "E_X_X_X_Y", "E_X_X_Y_Y", "E_X_Y_Y_Y", "E_Y_Y_Y_Y" # order 4
                                 ])
# Moments to collect
Moments_collect_bis = np.array([True]*len(Moments_labels_bis))      

# Number of sample trajectories to simulate in SSA
N_samples = 10000
# Number of parallel codes to run
ntasks = 20 # max 4 on computer, max 48 on cluster  

### SSA K_true ###
### True parameter ###
K_true = np.array([4.0, 3.0, 0.0, 0.7, 0., 0., 0., 0., 0., 0.04, 1.25, 0.0, 0.0])
SSA_list_true = SSA_moms_std(K_true, TT, Moments_collect_bis, Num_samples = N_samples, moms_lookUp_val = moms_lookUp, ntasks = ntasks)
obj_true = {"SSA_means":SSA_list_true[:, :, 0], 
              "SSA_std":SSA_list_true[:, :, 1],
                "Times":TT,
           "Parameters":K_true,
        "species_label":Moments_labels_bis[Moments_collect_bis]}

pickle.dump(obj_true, open("./Data_final/K_true/SSA/"+"SSA_true_pred_prey_N_%d.pck"%N_samples, "wb"))
#pickle.load(open("./Data/K_true/SSA/"+"SSA_true_N_%d.pck"%N_samples, "rb"))
#pdb.set_trace()


    
