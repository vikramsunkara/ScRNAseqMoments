import pickle
import pdb
import numpy as np
import joblib as jb

### Convert data to txt file for PIDC inference using Julia (Chan2017) ###
def write_to_file(Obs, Time, Repeat, output_file, species = ('M1','M2'), species_bool = [True, True]):
    # first line is repeated indexes of time
    species_used = np.array(species)[species_bool]
    out_file = open(output_file, "w")
    
    line1_kernel = "0\t" + "%.2f\t"*Repeat
    line1_kernel = "".join(line1_kernel)
    line1 = ""
    Tn = len(Time)
    for j in range(Tn):
        t = [Time[j]]*Repeat
        line1 = "".join(line1 + line1_kernel%tuple(t))
    out_file.write(''.join(line1+"\n"))
            
    # next lines are the repeated simulations for each species 
    for i in range(len(species_used)):
        out_file.write("\n")
        line_kernel = species_used[i] + "\t" + "%d\t"*Repeat
        line_kernel = "".join(line_kernel)
        line = ""
        for j in range(Tn):
            line = ''.join("" + line_kernel%tuple(Obs[j, list(species).index(species_used[i]), :]))
        out_file.write(''.join(line + "\n"))
    
    out_file.close()
    print('Saved to %s'%(output_file))   

def Convert(input_file, output_file, species, species_bool, subsamples = 1000):
    #### input_file : pickle file to convert with the keys: 'Obs': simulations, "Time":T,'dim_order':'Time, Dim, Repeat'
    #### output_file : txt file to save the data
    #### species, list of labels of dimension Dim
    #### species_bool, species t, shift = 0o include in the dataset
    
    f = open(input_file, "rb")
    input_dic = pickle.load(f)
    f.close()
    
    inds = np.arange(0,input_dic['Obs'].shape[-1],1,dtype=np.int)
    new_inds = np.random.choice(inds,size=subsamples,replace=False)
    
    Obs = input_dic["Obs"][0:120,:,new_inds] # only for time np.arange(0.0,60.0,delta_t), t in the simulations goes up to 80.0
    Time = input_dic["Time"][0:120]
    Repeat = Obs.shape[2]

    write_to_file(Obs, Time, Repeat, output_file, species, species_bool)
     
    
def Packing(input_file, output_file, species_bool, subsamples = 1000, Num_sets = 400):
    f = open(input_file, "rb")
    input_dic = pickle.load(f)
    f.close()
    T = input_dic["Time"][0:120] # 0:120 because only for time np.arange(0.0,60.0,0.5), t in the simulations goes up to 80.0
    
    inds = np.arange(0, input_dic['Obs'].shape[-1], 1, dtype=np.int)

    runs = []
    for n in range(Num_sets):
        new_inds = np.random.choice(inds,size=subsamples,replace=False)
        #runs.append(input_dic["Obs"][0:120, species_bool, :][:,:, new_inds]) 
        runs.append(input_dic["Obs"][0:120, species_bool, :][:,:, new_inds]) 
    
    packed =  open(output_file,"wb")
    pickle.dump({"Runs":runs, "Time":T, 'dim_order': "Time, Dim, Repeat"}, packed)
    packed.close()
    print('Packed to %s'%(output_file))    


ntasks = 40 # for the cluster
def PIDCMI_format(input_file, output_file_kernel, shift = 0):
    f = open(input_file, "rb")
    input_dic = pickle.load(f)
    f.close()
    
    Runs = input_dic["Runs"]
    Time = input_dic["Time"][shift:] 
    
    """
    for i in range(len(Runs)):
        output_file = output_file_kernel + "%d.txt"%i 
        Obs = Runs[i]
        Repeat = Obs.shape[2]
        write_to_file(Obs[shift:, :, :], len(Time), Repeat, output_file) 
    """
    #parallel
    jb.Parallel(n_jobs = ntasks)(jb.delayed(write_to_file)(Runs[i][shift:, :, :], Time, Runs[i].shape[2], output_file_kernel + "%d.txt"%i) for i in range(len(Runs)))
    
    
if __name__  == "__main__":
    ### Load the data ###
    species = ('G1','G2','P1','P2', 'M1','M2')
    species_bool = [False, False, False, False, True, True]
    
    sigma_1 = 0.01875
    sigma_1_list = sigma_1*np.array([1/2, 2, 2**2, 2**3, 2**4])
    
    
    
    for i in range(len(sigma_1_list)):
        ### Packing up Data #### this is fast
        in1 = "/nfs/datanumerik/people/araharin/Data_2021/two_MRNA_Double_Up_data_%d.pck"%i #Set 1
        out1 = "/nfs/datanumerik/people/araharin/Data_2021/Packed_two_MRNA_Double_Up_data_%d.pck"%i
        
        Packing(in1, out1, species_bool, subsamples = 10000, Num_sets = 400)
        
        in2 = "/nfs/datanumerik/people/araharin/Data_2021/two_MRNA_Double_Up_data_%d_1chng.pck"%i # Set 2
        out2 = "/nfs/datanumerik/people/araharin/Data_2021/Packed_two_MRNA_Double_Up_data_%d_1chng.pck"%i
        
        Packing(in2, out2, species_bool, subsamples = 10000, Num_sets = 400)
        
        in3 = "/nfs/datanumerik/people/araharin/Data_2021/two_MRNA_Single_Up_data_%d.pck"%i # Set3
        out3 = "/nfs/datanumerik/people/araharin/Data_20201/Packed_two_MRNA_Single_Up_data_%d.pck"%i
        
        Packing(in3, out3, species_bool, subsamples = 10000, Num_sets = 400)
      
        ### Convert into PIDC MI formats #### The result is not different from shift = 0
        shift = 30 # we had a shift of 30 in the Batch run for the moment based approach
        in1 = "/nfs/datanumerik/people/araharin/Data_2021/Packed_two_MRNA_Double_Up_data_%d.pck"%i 
        out1_kernel = "/nfs/datanumerik/people/araharin/Data_2021/Unpacked_MRNA_data_10000/txt_start30/Double_Up_data_%d_"%i
        PIDCMI_format(in1, out1_kernel, shift)
        
        in2 = "/nfs/datanumerik/people/araharin/Data_2021/Packed_two_MRNA_Double_Up_data_%d_1chng.pck"%i
        out2_kernel = "/nfs/datanumerik/people/araharin/Data_2021/Unpacked_MRNA_data_10000/txt_start30/Double_Up_data_%d_1chng_"%i
        PIDCMI_format(in2, out2_kernel, shift)
        
        in3 = "/nfs/datanumerik/people/araharin/Data_2021/Packed_two_MRNA_Single_Up_data_%d.pck"%i
        out3_kernel = "/nfs/datanumerik/people/araharin/Data_2021/Unpacked_MRNA_data_10000/txt_start30/Single_Up_data_%d_"%i
        PIDCMI_format(in3, out3_kernel, shift)
    
    
    
    
  
