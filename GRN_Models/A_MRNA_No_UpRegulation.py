import numpy as np
import joblib as jb
from functools import partial 
import pickle

sigma_1 = 0.125 # Adjusted it so that we don't use the true mean
sigma_2 = 0.5
sigma_3 = 0.125 # Adjusted it so that we don't use the true mean
sigma_4 = 0.5
rho_1 = 4.75
rho_2 = 1.0
rho_3 = 4.75
rho_4 = 1.0
theta = 5.0
delta = 0.1
k = 5.0


def propensities(x):
	return np.array(
		[sigma_1*(1-x[0]),
		sigma_2*x[0],
		rho_1*(1.0-x[0]),
		rho_2*x[0],
		theta*x[4],
		k*x[2],
		delta*x[4],

		#sigma_3*(1-x[1]), # removed the loop back
		#sigma_4*x[1],
		#rho_3*(1.0-x[1]),
		#rho_4*x[1],
		#theta*x[5],
		#k*x[3],
		#delta*x[5]
                ])
transitions = np.array([ (1,0,0,0,0,0),  # G_1_off + P2 --> G_1_on 
				(-1,0,0,0,0,0), # G_1_on --> G_1_off
				(0,0,0,0,1,0),  # G_1_off --> G_1_off + M1
				(0,0,0,0,1,0),	# G_1_on --> G_1_on + M1
				(0,0,1,0,0,0),  # M_1 --> (M_1) + P_1
				(0,0,-1,0,0,0), # P_1 --> 0
				(0,0,0,0,-1,0), # M_1 --> 0
				
				#(0,1,0,0,0,0),  # G_2_off --> G_2_off
				#(0,-1,0,0,0,0), # G_2_on --> G_off
				#(0,0,0,0,0,1),  # G_2_off --> G_2_off + M2
				#(0,0,0,0,0,1),  # G_2_on --> G_2_on + M2
				#(0,0,0,1,0,0),  # M_2 --> (M_2) + P_2
				#(0,0,0,-1,0,0), # P_2 --> 0
				#(0,0,0,0,0,-1)  # M_2 --> 0
				]).T
  
initial_state = np.array([0,0,0,0,0,0])
species = ('G1','G2','P1','P2', 'M1','M2')

delta_t = 0.5
T = np.arange(0.0,80.0,delta_t)
N =100000
#delta_t = 0.25
#T = np.arange(0.0,60.0,delta_t) # test data 
#N =10000
ntasks = 40 
##############
## SSA
##############

##############
## SSA
##############

import SSA

def Parrallel(SSAp, N, ntasks):
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
        
        f = open('/nfs/datanumerik/people/araharin/Data_032021/A_MRNA_No_Up/data_N_%d.pck'%n1,'wb')

        pickle.dump({'Obs': X_SSA ,'Time': T, 'dim_order':'Time, Dim, Repeat'},f)
        f.close()
       

import time

tick = time.time()
SSAp = partial(SSA.SSA_Fixed_Width_Trajectory, Stochiometry = transitions, Propensities = propensities, X_0 = initial_state,T_Obs_Points=T)

Parrallel(SSAp, N, ntasks)
tock = time.time()

print("Runs took", tock-tick)

