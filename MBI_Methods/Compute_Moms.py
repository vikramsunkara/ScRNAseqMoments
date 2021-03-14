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
    @brief load datafile and compute target moments
    @param input_filename     : datafile                           pickle object containing ndarray of dimension (#Time, #Dim, #Repeats)
    @param Moments            : exponents                          List of arrays                                [(#Dim),...]
    @param keep_species       : species to compute moments         array of bools                                (#Dim,)
    @param sample_size        : 
    '''
    f = open(input_filename,'rb')
    input_dic = pickle.load(f)
    f.close()
    ### Subsampling for good runs ###
    inds = np.arange(0,input_dic['Obs'].shape[-1],1,dtype=np.int)
    new_inds = np.random.choice(inds,size=sample_size,replace=False)
    data = input_dic['Obs'][:,:,new_inds] # t in the simulations goes up to 80.0
    T = input_dic['Time'][:]
    
    Moms = compute_moms(data, Moments, keep_species)
    return {"centers":Moms, "Times":T}