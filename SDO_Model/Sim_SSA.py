import scipy.stats as stats
from matplotlib import pyplot as plt
import pickle
import pdb

import time
import numpy as np
import joblib as jb
from functools import partial 

"""
SSA of Stochatic Damped Oscillator model 
evolve the system forward in time 
and return the moments of interests
"""

# All reactions
Reactions = [
"* -> A, #1",          # true k_1 = 4.000
"* -> B, #2",          # true k_2 = 3.000
"A -> *, #3",          # true k_3 = 0.000
"B -> *, #4",          # true k_4 = 0.700
"A -> B, #5",          # true k_5 = 0.000
"B -> A, #6",          # true k_6 = 0.000
"A + B -> B, #7",    # true k_7 = 0.000
"A + B -> A, #8",    # true k_8 = 0.000
"A + B -> *, #9",     # true k_9 = 0.000
"A + B -> B + B, #10",    # true k_10 = 0.04
"A -> A + A, #11",    # true k_11 = 1.25
"B -> B + B, #12",    # true k_12 = 0.00
"A + B -> A + A, #13"    # true k_13 = 0.00
]
# Stoichiometries
S = np.array([      [1,0], #1
                    [0,1], #2
                    [-1,0], #3
                    [0,-1],#4
                    [-1,1],#5
                    [1,-1],#6
                    [-1,0],#7
                    [0,-1],#8
                    [-1,-1],#9
                    [-1,1],#10
                    [1,0],#11
                    [0,1],#12SSA_moms_std(K_sindy, T, Moments_collect, K_lab = "K_sindy/SSA_samples/", subfit = sub_sample, Num_samples = N_samples, moms_lookUp_val = moms_lookUp, ntasks = ntasks)
                    [1,-1]#13  # Stoichiometries for each reactions
                    
                    ])

# Number of reactions
N_r = len(Reactions)
# Number of species
N_s = 2

# Initial population configuration of the system
X_0 = 30 #30
Y_0 = 20 #20

XY_0 = np.array([X_0, Y_0])                       

#hash_lookUp = {(0,0):0, (1,0):1, (0,1):2, 
#                (2,0):3, (1,1):4, (0,2):5, 
#                (3,0):6, (2,1):7, (1,2):8, (0,3):9,
#                               (4,0):10, (3,1):11, (2,2):12, (1,3):13, (0,4):14
#                } # order of the moments default for SSA without specific arguments
# Exponents used to compute moments (for species 1, for species 2) default 
moms_lookUp = np.array([(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3)]) # default

# Find a sample trajectory that correponds to a particular parameter K
def SSA(K_val, time_val, maxTime = 10):
    k_1, k_2, k_3, k_4, k_5, k_6, k_7, k_8, k_9, k_10, k_11, k_12, k_13 = K_val 
    t_initial = time_val[0]
    t_final = time_val[-1]
    
    XY = []
    Delta_ts = [0]
    
    # Initial values
    XY.append(XY_0)
    Delta_ts.append(t_initial)
    xy = XY_0.copy()
    t = t_initial
    
    t_list = [t_initial]
    # Computing trajectory until t_final
    tik = time.time()
    limit = 0
    while t < t_final and limit < maxTime:
        # compute propensities
        X = xy[0]
        Y = xy[1]
        a_i = np.array([k_1, k_2, k_3*X, k_4*Y, k_5*X, k_6*Y, k_7*X*Y, k_8*X*Y, k_9*X*Y, k_10*X*Y, k_11*X, k_12*Y, k_13*X*Y])
        a_0 = sum(a_i)
        if a_0 == 0:
            t = t_final
            t_list.append(t)
            XY.append(xy.copy())
            #print("STOP 1 happened, sum(propensities) = 0")
            break
        else:
            # choose/sample when reaction occures/fires (we don't know which reaction it is)
            tau_1 = stats.uniform.rvs()
            while tau_1 == 0: # making sure that tau_1 is never zero
                tau_1 = stats.uniform.rvs() 
            
            Dt  = (1/a_0)*np.log(1/tau_1)
            if t + Dt > t_final:
                t = t_final
                t_list.append(t)
                #print("STOP 2 happened current_time + Dt > t_final")
                XY.append(xy.copy())
                break
            else:
                t +=Dt
                t_list.append(t)
                # choose/sample which reaction has occured/fired
                tau_2 = stats.uniform.rvs()
                while tau_2 == 0:
                    tau_2 = stats.uniform.rvs()
                cum_prop = np.cumsum(a_i)
                chosen_j= np.sum(cum_prop < tau_2*cum_prop[-1])              
                # Update the trajectory
                xy += S[chosen_j, :]
                XY.append(xy.copy())
        limit += time.time() - tik
        tik = time.time()
    
    # adjusting trajectory for the time interval
    if t >= t_final:     # only return the SSA that reached final time else return none
        XY = np.array(XY)
        XY_adjusted = np.zeros((len(time_val), 2))
        for n in range(len(time_val)):
            t_target = time_val[n]
            if t_target in t_list:
                #print("In", n, np.where(t_list==t_target)[0][0])
                index_target = np.where(t_list==t_target)[0][0] # np.where gives a tuple with first element an array of all the index
            else: # take the time value directly bellow t_target because the system jumps in the next time step from there
                #print("Not in", n, np.where(t_list<=t_target)[0][-1])
                index_target = np.where(t_list<=t_target)[0][-1]
            XY_adjusted[n, :] = XY[index_target, :]
        return list(XY_adjusted)#np.array(XY_adjusted)
    else:
        "SSA exceeded maximal run time allowed"
             

# Find the samples that correspond to a target time t_target (t_target should always be smaller than t_final)
def SampleXY_moms(n, Alltraj, Collect_moms, moms_lookUp1 = moms_lookUp):
    Sample = np.zeros((len(Alltraj), N_s)) # N_s is number of species 
    for j in range(len(Alltraj)):
        SecXY = np.array(Alltraj[j])
        Sample[j, :] = SecXY[n, :]  
    
    index_moms = np.where(Collect_moms == True)[0]
    Moms_std = []
    for ms in index_moms:
        d1, d2 = moms_lookUp1[ms]
        #if d1+d2>2:
            #print("WRONG", d1, d2)
        order_ms = (Sample[:, 0]**d1)*(Sample[:, 1]**d2)
        Moms_std.append(np.array([np.mean(order_ms), np.std(order_ms)]))
    
    return np.array(Moms_std)

def SSA_moms_std(K, T, Moments_collect, K_lab = "SSA_samples_tests/", subfit = 1 , Num_samples = 100, moms_lookUp_val = moms_lookUp, maxTime = 10, ntasks = 4): 

    Div = 100
    q,r = divmod(Num_samples,Div)
    
    if r == 0:
        break_num_samples = [Div]*q
    else:
        break_num_samples = [Div]*q+[r]
        
    SSAp = partial(SSA, K_val = K, time_val = T, maxTime = maxTime)
    
    TRAJ = []
    for n1 in range(len(break_num_samples)): 
        print(n1+1, "out of", len(break_num_samples), "subsample", subfit)
        Ns = break_num_samples[n1]
        ### Save each Ns trajectories ###
        
        """
        TRAJ_Ns = jb.Parallel(n_jobs = ntasks)(jb.delayed(SSAp)() for sp in range(Ns))
        #TRAJ += list(TRAJ_Ns)
        TRAJ_Ns = list(filter(None, TRAJ_Ns)) # remove the unsucceful runs, i.e., those which exceeded the maximum run time allowed (maxTime = 10s)
        if len(TRAJ_Ns)!=0:
            obj = {"SSA_sample": list(TRAJ_Ns),
               "Num_sample": Ns,
               "Times": T,
               "Parameters":K,
               "Subsample_for_fit":subfit,
               "percentage_success":100*len(TRAJ_Ns)/Num_samples,
               "Num_samples": Num_samples,
                }
            pickle.dump(obj, open("./Data_final/"+K_lab+"SSA_N_%d_%d.pck"%(subfit, n1), "wb"))

        """
        ### Open trajectories to compute the moments ####
        
        f = open("./Data_final/"+K_lab+"SSA_N_%d_%d.pck"%(subfit, n1), "rb")
        sample = pickle.load(f)
        f.close()
        TRAJ +=sample["SSA_sample"]
        
    
    Alltraj_val = np.array(TRAJ)
    SampleXYp = partial(SampleXY_moms, Alltraj = Alltraj_val, Collect_moms = Moments_collect, moms_lookUp1 = moms_lookUp_val)
    SAMPL = jb.Parallel(n_jobs = ntasks)(jb.delayed(SampleXYp)(n) for n in range(len(T)))
    Moms_and_std = np.array(list(SAMPL)) # of shape (len(T), np.sum(Moments_collect), 2)
    
    return Moms_and_std
    










