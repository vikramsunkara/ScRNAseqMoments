import numpy as np 
import pickle
import pdb
import pylab as pl 

from Run_Multiple_mRNA_Inference import Batch_Inference

from Compute_Moms import *

"""
Multiple inference for the three models (Double Up, Single Up) with all simulations 
having initial condition (G1, G2, P1, P2, M1, M2) =(0, 0, 0, 0, 0, 0) (we only used M1 and M2)
Input files, please put your own data file paths
"""
File_list = []

DLab_m =[]

sigma_1 = 0.01875
sigma_1_list = sigma_1*np.array([1/2, 2, 2**2, 2**3, 2**4])

"""
for i in range(len(sigma_1_list)):
    File = "/nfs/datanumerik/people/araharin/Data_032021/two_MRNA_Double_Up_data_%d.pck"%i
    Lab = "Double_Up_%d"%i
    File_list.append(File)
    DLab_m.append(Lab)
"""
"""
for i in range(2, len(sigma_1_list)):
    File = "/nfs/datanumerik/people/araharin/Data_032021/two_MRNA_Double_Up_data_%d_1chng.pck"%i
    Lab = "Double_Up_%d_1chng"%i
    File_list.append(File)
    DLab_m.append(Lab)
"""
for i in range(3,len(sigma_1_list)):
    File = "/nfs/datanumerik/people/araharin/Data_032021/two_MRNA_Single_Up_data_%d.pck"%i
    Lab = "Single_Up_%d"%i
    File_list.append(File)
    DLab_m.append(Lab)


indexes = [(1,0),(0,1),(2,0),(1,1),(0,2),(3,0),(2,1),(1,2),(0,3),(4,0),(3,1),(2,2),(1,3),(0,4)]
Moments = [np.array(item) for item in indexes]
Species_To_Store = np.array([False,False,False,False,True,True])

BatchNum = 400 # Number of replicate datasets

import joblib as jb
from functools import partial 
ntasks = 40
for i in range(len(File_list)):
    input_file = File_list[i]
    # compute moments for BatchNum replicate datasets
    #Moms_time_data = []
    #for n in range(BatchNum):
        #data = Load_moms_time(input_file, Moments, keep_species = Species_To_Store)
        #Moms_time_data.append(data)
    Load_moms_time_p = partial(Load_moms_time, input_filename = input_file, Moments = Moments, keep_species = Species_To_Store)    
    Moms_time_data = jb.Parallel(n_jobs = ntasks)(jb.delayed(Load_moms_time_p)() for n in range(BatchNum))
    
    # preform GRN inference for each dataset in the Batch
    Batch_Inference(Moms_time_data, [DLab_m[i]+"(#%d)"%n for n in range(BatchNum)], DLab_m[i], shift = 30, sub_sample = 15, PDF_Save_dir = '/nfs/datanumerik/people/araharin/Data_032021/PDF', GRN_Save_dir = '/nfs/datanumerik/people/araharin/Data_032021/GRNs', indexes = indexes)








