import numpy as np 
import pickle
import pdb
import pylab as pl 

from Run_Multiple_mRNA_Inference import Batch_Inference

from Compute_Moms import *

"""
Multiple inference for the three models (Double Up, No Up, Single Up) with all simulations 
having initial condition (G1, G2, P1, P2, M1, M2) =(0, 0, 0, 0, 0, 0) (we only used M1 and M2)
Input files, please put your own data file paths
"""

File1 = "/2mRNA_100000/two_MRNA_Double_Up_data_100000.pck" 
File2 = "/2mRNA_100000/two_MRNA_No_Up_data_100000.pck" 
File3 = "/2mRNA_100000/two_MRNA_Single_Up_data_100000.pck" 

"""
Multiple inference for the model Single Up with all simulations 
having different conditions of the form (G1, G2, P1, P2, M1, M2) =(0, 0, 0, 0, M1, M2) (we only change M1 and M2)
Input files, please put your own data file paths
"""

File3a = "/2mRNA_100000/two_MRNA_Single_Up_data_70_0_100000.pck" 
File3b = "/2mRNA_100000/two_MRNA_Single_Up_data_0_70_100000.pck" 
File3c = "/2mRNA_100000/two_MRNA_Single_Up_data_70_70_100000.pck" 


DLab_m =["Double_up", "No_up", "Single_up", "Single_up_70_0 ", "Single_up_0_70", "Single_up_70_70"]


File_list = [File1, File2, File3, File3a, File3b, File3c]

indexes = [(1,0),(0,1),(2,0),(1,1),(0,2),(3,0),(2,1),(1,2),(0,3),(4,0),(3,1),(2,2),(1,3),(0,4)]
Moments = [np.array(item) for item in indexes]
Species_To_Store = np.array([False,False,False,False,True,True])

BatchNum = 40 # Number of replicate datasets


import joblib as jb
from functools import partial 
ntasks = 40

temp_corr = True # if True, then remove temporal correlation in the computation of the moments, i.e., take random cell sample at every timestep. If False, then take random trajectories instead.

for i in range(len(File_list)):
    input_file = File_list[i]
    
    # compute moments for BatchNum replicate datasets (without parallel computing)
    """
    Moms_time_data = []
    for n in range(BatchNum):
        if temp_corr:
            ''' 
            @brief remove temporal correlation
               
            ''' 
            data = Load_moms_time_diff(input_file, Moments, keep_species = Species_To_Store)
        else:
            '''
            @brief ignore temporal correlation
            '''
            data = Load_moms_time(input_file, Moments, keep_species = Species_To_Store)

        #Moms_time_data.append(data)
    """
    
    # compute moments for BatchNum replicate datasets (with parallel computing)
    if temp_corr:
        ''' 
        @brief remove temporal correlation
        '''
        Load_moms_time_p = partial(Load_moms_time_diff, input_filename = input_file, Moments = Moments, keep_species = Species_To_Store)
        
        Moms_time_data = jb.Parallel(n_jobs = ntasks)(jb.delayed(Load_moms_time_p)() for n in range(BatchNum))
    
        # preform GRN inference for each dataset in the Batch
        Batch_Inference(Moms_time_data, [DLab_m[i]+"(#%d)"%n for n in range(BatchNum)], DLab_m[i], shift = 30, sub_sample = 15, PDF_Save_dir = 'PDF_rand', GRN_Save_dir = 'GRNs_rand', indexes = indexes)
                
    else:
        '''
        @brief ignore temporal correlation
        '''
        Load_moms_time_p = partial(Load_moms_time, input_filename = input_file, Moments = Moments, keep_species = Species_To_Store)
        
        Moms_time_data = jb.Parallel(n_jobs = ntasks)(jb.delayed(Load_moms_time_p)() for n in range(BatchNum))
    
        # preform GRN inference for each dataset in the Batch
        Batch_Inference(Moms_time_data, [DLab_m[i]+"(#%d)"%n for n in range(BatchNum)], DLab_m[i], shift = 30, sub_sample = 15, PDF_Save_dir = 'PDF', GRN_Save_dir = 'GRNs', indexes = indexes)





