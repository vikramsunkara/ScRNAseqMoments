####
# Code For Constructing Realisations of Kurtz Processes.
# Author The big V.
####

import numpy as np 
#np.set_printoptions(precision=1)
import pdb

def Time_To_Next_Reaction(lam):
    """
    @brief The function samples from an exponential distribution with rate "lam". 
    @param lam : real value positive.
    """

    # small hack as the numpy uniform random number includes 0
    r = np.random.rand()
    while r == 0:
        r = np.random.rand()

    return (1.0/lam)*np.log(1.0/r)

def Find_Reaction_Index(a):
    """    
    @brief The function takes in the propensity vector and returns the index of a possible reaction candidate.
    @param a : Array (num_reaction,1) 

    """
    # small hack as the numpy uniform random number includes 0
    r = np.random.rand()
    
    while r == 0:
        r = np.random.rand()
    
    cum_prop = np.cumsum(a)
    j_chosen = np.sum(cum_prop < r*cum_prop[-1])
    
    return j_chosen

def SSA(Stochiometry,Propensities,X_0, t_0, t_final):
    """
    @brief  The Stochastic Simulation Algorithm. Given the stochiometry, propensities and the initial state; the algorithm
            gives a sample of the Kurtz process at $t_final.$
    
    @param Stochiometry : Numpy Array (Num_species,Num_reaction).
    @param Propensities : Function which given a state, returns the respective reaction propensities.
    @param X_0            : Numpy Array (Num_species, 1).
    @param t_final        : positive number.

    """

    t = t_0
    x = X_0.copy()

    while t <= t_final:
        
        a = Propensities(x)

        # First Jump Time
        tau = Time_To_Next_Reaction(np.sum(a))

        # Test if we have jumped to far
        if (t + tau > t_final) or (np.sum(a) == 0):
            return x, t_final
        else:
            # Since we have not, we need to find the next reaction
            t = t + tau
            j = Find_Reaction_Index(a)
            x = x + Stochiometry[:,j]


def SSA_Fixed_Width_Trajectory(Stochiometry,Propensities,X_0,T_Obs_Points):
    X = [X_0]

    for i in range(len(T_Obs_Points)-1):
        x, t = SSA(Stochiometry,Propensities,X[-1],T_Obs_Points[i],T_Obs_Points[i+1])
        X.append(x)

    return np.array(X).T #, T_Obs_Points