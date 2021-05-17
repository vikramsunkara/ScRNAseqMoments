import numpy as np 
import pickle
import pdb
import pylab as pl 

def Sims_2_Moms(Sim_Block, Moments):
    '''
    @param Sim_Block: data                 numpy array           (#Time, #Dim, #Repeats)
    @param Moments:   exponents            List of arrays        [(#Dim,),...]
    '''
    num_Moms = len(Moments)
    Sim_Block = Sim_Block.astype(np.float)
    
    Moms = np.zeros((Sim_Block.shape[0],len(Moments)+1))

    Moms[:,0] = 1.0

    for i in range(num_Moms):
        Powered = np.power(Sim_Block,Moments[i][np.newaxis,:,np.newaxis])
        Produced = np.prod(Powered,axis=1)
        Moms[:,i+1] = np.average(Produced,axis=1)

    return Moms


def compute_moms(data, Moments, keep_species):
    '''
    @param data   : data from which to compute moments      ndarray              (#Time, #Dim, #Repeats)
    @param Moments: exponents                               List of arrays       [(#Dim),...]
    @param keep_species: species to compute moments         array of bool        (#Dim,)
    
    '''
    if keep_species is None: # process all species
        Moms = Sims_2_Moms(data,Moments)
    else:
        Moms = Sims_2_Moms(data[:,keep_species,:],Moments)
    return Moms


def Load_moms_time(input_filename, Moments, keep_species =None, sample_size = 10000):
    '''
    @brief load datafile and compute target moments (ingnore the possibility of temporal correlation)
    @param input_filename     : datafile                           pickle object containing ndarray of dimension (#Time, #Dim, #Repeats)
    @param Moments            : exponents                          List of arrays                                [(#Dim),...]
    @param keep_species       : species to compute moments         array of bools                                (#Dim,)
    @param sample_size        : 
    '''
    f = open(input_filename,'rb')
    input_dic = pickle.load(f)
    f.close()
    ### Subsampling for runs ###
    inds = np.arange(0,input_dic['Obs'].shape[-1],1,dtype=np.int)
    new_inds = np.random.choice(inds,size=sample_size,replace=False)
    #data = input_dic['Obs'][:120,:,new_inds] # only for time np.arange(0.0,60.0,delta_t), t in the simulations goes up to 80.0
    data = input_dic['Obs'][:,:,new_inds]
    
    #T = input_dic['Time'][:120] # only for time np.arange(0.0,60.0,delta_t), time in the simulations goes up to 80.0
    T = input_dic['Time'][:]
    
    Moms = compute_moms(data, Moments, keep_species)
    return {"centers":Moms, "Times":T}


###### New modules to remove the possibility of temporal correlation in the computation of the moments (Cf. Reviewer Nr.2) ######

def diff_time_samples(input_dic, sample_size):
    '''
    @brief remove the possibility of temporal correlation by sampling different cells at each time point
    '''
    Sim_data = input_dic['Obs'] # -> numpy array           (#Time, #Dim, #Repeats)
    data = np.zeros((Sim_data.shape[0], Sim_data.shape[1], sample_size))
    
    inds = np.arange(0,Sim_data.shape[-1],1,dtype=np.int)
    
    #for t in range(Sim_data.shape[0], 120): # only for time np.arange(0.0,60.0,delta_t), t in the simulations goes up to 80.0
    for t in range(Sim_data.shape[0]):
        ### Subsampling for runs ###
        new_inds = np.random.choice(inds,size=sample_size,replace=False)
        data[t, :, :] = Sim_data[t][:,new_inds] 
    
    return data

def Load_moms_time_diff(input_filename, Moments, keep_species =None, sample_size = 10000):
    '''
    @brief load datafile and compute target moments (remove the possibility of temporal correlation)
    @param input_filename     : datafile                           pickle object containing ndarray of dimension (#Time, #Dim, #Repeats)
    @param Moments            : exponents                          List of arrays                                [(#Dim),...]
    @param keep_species       : species to compute moments         array of bools                                (#Dim,)
    @param sample_size        : 
    '''
    f = open(input_filename,'rb')
    input_dic = pickle.load(f)
    f.close()
   
    data = diff_time_samples(input_dic, sample_size)
    
    #T = input_dic['Time'][:120] # only for time np.arange(0.0,60.0,delta_t), time in the simulations goes up to 80.0
    T = input_dic['Time'][:]
    
    Moms = compute_moms(data, Moments, keep_species)
    return {"centers":Moms, "Times":T}