from .Symbolic_Moment_Generator import Symbolic_Moments
import sympy as sy
import numpy as np
import pdb
from .util import decompose, make_poly
import pickle
from numpy import (array, dot, arccos, clip)
from numpy.linalg import norm

np.set_printoptions(formatter={'float': '{: 0.4f}'.format},linewidth=150)


def generate_design_blocks(hash_lookUp, Moment_Basis, Verbose=False):
    A, B = sy.var('A B')
    k_1, k_2, k_3, k_4, k_5, k_6, k_7, k_8, k_9, k_10, k_11, k_12, k_13 = sy.symbols('k_1, k_2, k_3, k_4, k_5, k_6, k_7, k_8, k_9, k_10, k_11, k_12, k_13', constant=True)
    
    var = [A,B]
    reaction_coeffs = [	k_1, 
    					k_2,
    					k_3, 
    					k_4, 
    					k_5,
    					k_6,
    					k_7, 
    				 	k_8, 
    					k_9,
    					k_10,
    					k_11,
    					k_12,
    					k_13
    					]
    #reaction_coeffs = [k_1, k_2, k_3, k_4, k_5]
    
    def T(A,B,power):
    	return lambda A,B : (A**power[0])*(B**power[1])
    
    ## ALl reactions
    #propensities =[k_1, k_2, k_3*A, k_4*B, k_5*A*B]
    
    propensities =[k_1, 
    				k_2, 
    				k_3*A, 
    				k_4*B, 
    				k_5*A, 
    				k_6*B, 
    				k_7*A*B,  
    				k_8*A*B,  
    				k_9*A*B,  
    				k_10*A*B,
    				k_11*A,
    				k_12*B,
    				k_13*A*B
    				]
    transitions = [ 	(1,0), #1
    					(0,1), #2
    					(-1,0), #3
    					(0,-1),#4
    					(-1,1),#5
    					(1,-1),#6
    					(-1,0),#7
    					(0,-1),#8
    					(-1,-1),#9
    					(-1,1),#10
    					(1,0),#11
    					(0,1),#12
    					(1,-1)#13
    				]
    Moment_Obj = Symbolic_Moments(var,propensities,transitions,np.array([False,False]))
    Moment_Obj.Set_Moment_Degree_Function(T)
    
    '''
    
    hash_lookUp = {(0,0):0, (1,0):1, (0,1):2, # 1st moments
    				(2,0):3, (1,1):4, (0,2):5, # 2nd moments
    				(3,0):6, (2,1):7, (1,2):8, (0,3):9, # 3rd moments
    				(4,0):10, (3,1):11, (2,2):12, (1,3):13, (0,4):14 # 4th moments
    				}
    
    Moment_Basis = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), (3,0), (2,1), (1,2), (0,3)]
    '''
    
    ############ The Columns of the Design Matrix
    
    Reaction_Mapping_Mat = []
    Reaction_Labels = []
    for i in range(len(propensities)):
        Symbolic_Forms = Moment_Obj.Compute_Moments_for_Reaction(i, Moment_Basis)
        Reaction_Labels.append(Symbolic_Forms)
        if Verbose:
            print(Symbolic_Forms)
        Maps = Moment_Obj.Match_Terms_To_LookUp(Symbolic_Forms,hash_lookUp,reaction_coeffs[i])
    	# This generates a Matrix to which to multiple each k_i to. Sum of these is the Design matrix.
    	# These can be saved and loaded into Pyro.
        Reaction_Mapping_Mat.append(Maps)
        if Verbose:
            print(Maps)
        
    return np.array(Reaction_Mapping_Mat).astype(np.float)

	
