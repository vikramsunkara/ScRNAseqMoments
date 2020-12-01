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
    
### Import data and all necessary arguments for SINDY and NLLS ###
from Data_Loader_PP import *   

# SSA generating function
from Sim_SSA import *     

# Number of sample trajectories to simulate in SSA
N_samples = 1000 #1000 #1000
# Number of parallel codes to run
ntasks = 20 # max 4 on computer, max 20 on one gpu node, check memory allocations in nodes 

from matplotlib import pyplot as plt
### Figure for quick check only one moment ###
plt.figure(figsize = (15, 7.5))
sub1 = plt.subplot(221)
plt.title("SSA K true")
sub2 = plt.subplot(222)
plt.title("SSA K sindy")
sub3 = plt.subplot(223)
plt.title("SSA K nlls with derivatives")
fsize = 14

try:
    plt.suptitle("%d sample trajectories"%N_samples, fontsize = fsize) 
except:
    plt.suptitle("%d sample trajectories"%100, fontsize = fsize) 

### SSA K_true ###
### True parameter ###
"""
SSA_list_true = SSA_moms_std(K_true, T, Moments_collect, Num_samples = N_samples, moms_lookUp_val = moms_lookUp, ntasks = ntasks)
obj_true = {"SSA_means":SSA_list_true[:, :, 0], 
              "SSA_std":SSA_list_true[:, :, 1],
                "Times":T,
           "Parameters":K_true,
        "species_label":Moments_labels[Moments_collect]}
pickle.dump(obj_true, open("./Data_final/K_true/SSA/"+"SSA_true.pck", "wb"))
sub1.plot(T, SSA_list_true[:, 1, 0], "-", label = "SSA E_X, K true")
sub1.plot(T, Mom_list[:, 1], "x", color = "black",label = "Data E_X")
sub1.legend(fontsize = fsize)
"""
sub_list = np.array([1, 2, 4, 8, 10, 12, 16, 20, 25, 28, 32]) 
for n in range(len(sub_list)): 
    sub_sample = sub_list[n]
    print(sub_sample)
    #####
    # SSA K_sindy
    #####
    #### Load the Infered K #####
    
    k1 = open("./Data_final/K_sindy/Values/"+"K_%d.pck"%sub_sample, "rb")
    stuff1 = pickle.load(k1)
    k1.close()
    
    K_sindy = np.array(stuff1["Parameters"])
    print("n=%d SINDY"%n, K_sindy)

    SSA_list_sindy = SSA_moms_std(K_sindy, TT_0, Moments_collect, K_lab = "K_sindy/SSA_samples/", subfit = sub_sample, Num_samples = N_samples, moms_lookUp_val = moms_lookUp, ntasks = ntasks)
    #SSA_moms_std(K_sindy, TT_0, Moments_collect, K_lab = "K_sindy/SSA_samples/", subfit = sub_sample, Num_samples = N_samples, moms_lookUp_val = moms_lookUp, ntasks = ntasks) # in case it returns None and I only saved trajectory samples
    
    obj_sindy = {"SSA_means":SSA_list_sindy[:, :, 0],
                   "SSA_std":SSA_list_sindy[:, :, 1],
                     "Times":TT_0,
                "Parameters":K_sindy,
        "Subsample_for_fit" :sub_sample,
             "species_label":Moments_labels[Moments_collect]}
    pickle.dump(obj_sindy, open("./Data_final/K_sindy/SSA/"+"SSA_%d.pck"%sub_sample, "wb"))
    
    
    ####
    # SSA K_nlls_der
    ####
    #### Load the Infered K #####
    k2 = open("./Data_final/K_nlls_der/Values/"+"K_%d.pck"%sub_sample, "rb")
    #k2 = open("./Data_final/K_nlls_der/Values/"+"K_%d_start_true.pck"%sub_sample, "rb")
    
    stuff2 = pickle.load(k2)
    k2.close()
    
    K_nlls_der = np.array(stuff2["Parameters"])
    print("n=%d NLLS"%n, K_nlls_der)
    
    SSA_list_nlls_der = SSA_moms_std(K_nlls_der, TT_0, Moments_collect, K_lab = "K_nlls_der/SSA_samples/",subfit = sub_sample, Num_samples = N_samples, moms_lookUp_val = moms_lookUp, ntasks = ntasks)
    #SSA_moms_std(K_nlls_der, TT_0, Moments_collect, K_lab = "K_nlls_der/SSA_samples/",subfit = sub_sample, Num_samples = N_samples, moms_lookUp_val = moms_lookUp, ntasks = ntasks)
    
    obj_nlls = {"SSA_means":SSA_list_nlls_der[:, :, 0],
                  "SSA_std":SSA_list_nlls_der[:, :, 1],
                    "Times":TT_0,
               "Parameters":K_nlls_der,
      "Subsample_for_fit" :sub_sample,
           "species_label":Moments_labels[Moments_collect]}
    pickle.dump(obj_nlls, open("./Data_final/K_nlls_der/SSA/"+"SSA_%d.pck"%sub_sample, "wb"))
    
    
    """
    if n == 6:
        sub2.plot(T,SSA_list_sindy[:, 1, 0], "-", label = "SSA E_X, n =%d"%n)
        sub2.plot(T, Mom_list[:, 1], "x", color = "black")
        #sub2.legend(fontsize = fsize)
        
        sub3.plot(T,SSA_list_nlls_der[:, 1, 0], "-", label = "SSA E_X, n =%d"%n)
        sub3.plot(T, Mom_list[:, 1], "x", color = "black")

        sub3.legend(fontsize = fsize, loc = (1.5, 0.1))
    else:
        sub2.plot(T,SSA_list_sindy[:, 1, 0], "-", label = "SSA E_X, n =%d"%n)
        sub3.plot(T,SSA_list_nlls_der[:, 1, 0], "-", label = "SSA E_X, n =%d"%n)
        print("------------------------------------------------------------------")
    
plt.subplots_adjust(bottom = 0.06, right = 0.95, left = 0.06, top = 0.89, wspace = 0.15, hspace = 0.20)
plt.savefig("SSA_test_fig_2.png")
#plt.show()
"""
    
