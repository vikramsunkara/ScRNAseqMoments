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
    '''
    # moms in columns  of design blocks  
    hash_lookUp 
    
    # moms in rows of design blocks
    Moment_Basis
    '''
    A, B = sy.var('A B')
    k_1, k_2, k_3, k_4, k_7, k_8, k_9, k_10, k_11, k_12, k_13 = sy.symbols('k_1, k_2, k_3, k_4, k_7, k_8, k_9, k_10, k_11, k_12, k_13', constant=True)

    var = [A,B]
    reaction_coeffs = [	
                        k_1, 
    					k_2,
    					k_3, 
    					k_4, 
    					#k_5,
    					#k_6,
    					k_7, 
    				 	k_8, 
    					#k_9,
    					k_10,
    					k_11,
    					k_12,
    					k_13
    					]
    
    def T(A,B,power):
    	return lambda A,B : (A**power[0])*(B**power[1])
    
    ## ALl reactions
    
    propensities =[ k_1, # A_birth
    				k_2, # B_birth
    				k_3*A, # A_death
    				k_4*B, # B_death
    				#k_5*A, 
    				#k_6*B, 
    				k_7*A*B,  # B_down_A
    				k_8*A*B,  # A_down_B
    				#k_9*A*B,  
    				k_10*B, # B_Up A
    				k_11*A, # A_Up
    				k_12*B, # B_Up
    				k_13*A  # A_Up_B
    				]
    transitions = [ 	(1,0), #1
    					(0,1), #2
    					(-1,0), #3
    					(0,-1),#4
    					#(-1,1),#5
    					#(1,-1),#6
    					(-1,0),#7
    					(0,-1),#8
    					#(-1,-1),#9
    					(1,0),#10
    					(1,0),#11
    					(0,1),#12
    					(0,1)#13
    				]
    Moment_Obj = Symbolic_Moments(var,propensities,transitions,np.array([False,False]))
    Moment_Obj.Set_Moment_Degree_Function(T)
    
    
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