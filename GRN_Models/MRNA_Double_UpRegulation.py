import numpy as np
import joblib as jb
from functools import partial 
import pickle

sigma_1 = 0.01875
sigma_2 = 0.5
sigma_3 = 0.01875
sigma_4 = 0.5
rho_1 = 6.0
rho_2 = 1.0
rho_3 = 6.0
rho_4 = 1.0
theta = 5.0
delta = 0.1
k = 5.0

def propensities(x, sigma_1 = sigma_1, sigma_2 = sigma_2):
    return np.array([
        sigma_1*x[3]*(1-x[0]),
		sigma_2*x[0],
		rho_1*(1.0-x[0]),
		rho_2*x[0],
		theta*x[4],
		k*x[2],
		delta*x[4],
				
		sigma_3*x[2]*(1-x[1]),
		sigma_4*x[1],
		rho_3*(1.0-x[1]),
		rho_4*x[1],
		theta*x[5],
		k*x[3],
		delta*x[5]
            ])
transitions = np.array([  
			(1,0,0,-1,0,0), # G_1_off + P2 --> G_1_on 
			(-1,0,0,0,0,0), # G_1_on --> G_1_off
			(0,0,0,0,1,0),  # G_1_off --> G_1_off + M1
			(0,0,0,0,1,0),	# G_1_on --> G_1_on + M1
			(0,0,1,0,0,0),  # M_1 --> (M_1) + P_1
			(0,0,-1,0,0,0), # P_1 --> 0
			(0,0,0,0,-1,0), # M_1 --> 0			
			
			(0,1,-1,0,0,0), # G_2_off + P1 --> G_2_on
			(0,-1,0,0,0,0), # G_2_on --> G_off
			(0,0,0,0,0,1),  # G_2_off --> G_2_off + M2
			(0,0,0,0,0,1),  # G_2_on --> G_2_on + M2
			(0,0,0,1,0,0),  # M_2 --> (M_2) + P_2
			(0,0,0,-1,0,0), # P_2 --> 0
			(0,0,0,0,0,-1)  # M_2 --> 0
			]).T
			
initial_state = np.array([0,0,0,0,0,0])
species = ('G1','G2','P1','P2', 'M1','M2')

delta_t = 0.5
T = np.arange(0.0,80.0,delta_t)
N =100000
ntasks = 40
##############
## SSA
##############

##############
## SSA
##############

import SSA

def Parrallel(SSAp, N, ntasks, i):
    q,r = divmod(N,100)
    if r == 0:
        break_num_samples = [100]*q
    else:
        break_num_samples = [100]*q+[r]

    for n1 in range(len(break_num_samples)):
        print("%d out of %d"%(n1, len(break_num_samples)))
        Ns = break_num_samples[n1]
        Resamples = jb.Parallel(n_jobs = ntasks)(jb.delayed(SSAp)() for sp in range(Ns))
    
        X_SSA = np.array(Resamples).T
    
        f = open('/nfs/datanumerik/people/araharin/Data_032021/two_MRNA_Double_Up/data_N_%d_%d.pck'%(n1, i),'wb')
        pickle.dump({'Obs': X_SSA ,'Time': T, 'dim_order':'Time, Dim, Repeat'},f)
        f.close()
        

import time
tick = time.time()

sigma_1_list = sigma_1*np.array([1/2, 2, 2**2, 2**3, 2**4]) # to run sensitivity to sigma_1
sigma_2_list = sigma_2*np.array([1/2, 2, 2**2, 2**3, 2**4]) # to run sensitivity to sigma_1

for i in range(len(sigma_1_list)):
    sigma1 = sigma_1_list[i]
    sigma2 = sigma_2_list[i]
    propensities_p = partial(propensities, sigma_1 = sigma1, sigma_2 = sigma2)

    SSAp = partial(SSA.SSA_Fixed_Width_Trajectory, Stochiometry = transitions, Propensities = propensities_p, X_0 = initial_state,T_Obs_Points=T)
    Parrallel(SSAp, N, ntasks, i)
    
    tock = time.time()
    print(tock-tick)





