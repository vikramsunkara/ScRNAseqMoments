import numpy as np 
import pickle
import pdb
import pylab as pl 

from Run_Multiple_mRNA_Inference import Batch_Inference

from Compute_Moms import *

"""
Multiple inference for the three models (Double Up, No Up, Single Up) with all simulations 
having initial condition (G1, G2, P1, P2, M1, M2) =(0, 0, 0, 0, 0, 0) (we only used M1 and M2)
"""

File1 = "/2mRNA_100000/two_MRNA_Double_Up_data_100000.pck" # Set1
File2 = "/2mRNA_100000/two_MRNA_No_Up_data_100000.pck" # Set2 
File3 = "/2mRNA_100000/two_MRNA_Single_Up_data_100000.pck" # Set3

"""
Multiple inference for the model Single Up with all simulations 
having different conditions of the form (G1, G2, P1, P2, M1, M2) =(0, 0, 0, 0, M1, M2) (we only change M1 and M2)
"""

File3a = "/2mRNA_100000/two_MRNA_Single_Up_data_70_0_100000.pck" # Set3
File3b = "/2mRNA_100000/two_MRNA_Single_Up_data_0_70_100000.pck" # Set3
File3c = "/2mRNA_100000/two_MRNA_Single_Up_data_70_70_100000.pck" # Set3


DLab_m =["Double_up", "No_up", "Single_up", "Single_up_70_0 ", "Single_up_0_70", "Single_up_70_70"]


File_list = [File1, File2, File3, File3a, File3b, File3c]

indexes = [(1,0),(0,1),(2,0),(1,1),(0,2),(3,0),(2,1),(1,2),(0,3),(4,0),(3,1),(2,2),(1,3),(0,4)]
Moments = [np.array(item) for item in indexes]
Species_To_Store = np.array([False,False,False,False,True,True])

BatchNum = 400
for i in range(len(File_list)):
    input_file = File_list[i]
    Moms_time_data = []
    for n in range(BatchNum):
        data = Load_moms_time(input_file, Moments, keep_species = Species_To_Store)
        Moms_time_data.append(data)

    Batch_Inference(Moms_time_data, [DLab_m[i]+"(#%d)"%n for n in range(BatchNum)], DLab_m[i], shift = 30, sub_sample = 15, PDF_Save_dir = 'PDF', GRN_Save_dir = 'GRNs')





