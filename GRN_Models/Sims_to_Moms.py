###
# Module to read in Stochastic simulations and Generate Moments.
###

import numpy as np 
import pdb
import pickle

def Sims_2_Moms(Sim_Block, Moments):
    '''
    Sim_Block         numpy array      (Time, Dim, Repeats)
    Moments         List              numpy arrays [(Dim,),...]
    '''

    num_Moms = len(Moments)

    Moms = np.zeros((Sim_Block.shape[0],len(Moments)+1))

    Moms[:,0] = 1.0

    for i in range(num_Moms):
        Powered = np.power(Sim_Block,Moments[i][np.newaxis,:,np.newaxis])
        Produced = np.prod(Powered,axis=1)
        Moms[:,i+1] = np.average(Produced,axis=1)

    return Moms

def Load_and_Save(input_filename, output_filename, Moments, keep_species =None):

    f = open(input_filename,'rb')
    input_dic = pickle.load(f)
    f.close()
    ### Subsampling for 1000 good runs ###
    inds = np.arange(0,input_dic['Obs'].shape[-1],1,dtype=np.int)
    new_inds = np.random.choice(inds,size=100000,replace=False)
    T = input_dic["Time"][0:120] # only for time np.arange(0.0,60.0,delta_t), t in the simulations goes up to 80.0
    data = input_dic['Obs'][0:120,:,new_inds]
    
    if keep_species is None: # process all species
        Moms = Sims_2_Moms(data,Moments)
    else:
        Moms = Sims_2_Moms(data[:,keep_species,:],Moments)

    f = open(output_filename,"wb")
    pickle.dump({'centers': Moms, 'Times': T},f)
    f.close()
    print('Saved to %s'%(output_filename))
    


if __name__ == '__main__':
    from Plot_Moments import * # Plot
    
    indexes = [(1,0),(0,1),(2,0),(1,1),(0,2),(3,0),(2,1),(1,2),(0,3),(4,0),(3,1),(2,2),(1,3),(0,4)]
    Moments = [np.array(item) for item in indexes]
    Species_To_Store = np.array([False,False,False,False,True,True])
       
 
    input_file = "/nfs/datanumerik/people/araharin/2mRNA_100000/two_MRNA_Double_Up_data_100000.pck"
    out_outfile = '/nfs/datanumerik/people/araharin/2mRNA_100000/two_MRNA_Double_Up_data_Centres_And_Times_100000.pck'
    Load_and_Save(input_file,out_outfile,Moments,Species_To_Store)
    #load(out_outfile, "Double_up") #plot moment
    
    input_file = "/nfs/datanumerik/people/araharin/2mRNA_100000/two_MRNA_Single_Up_data_100000.pck"
    out_outfile = '/nfs/datanumerik/people/araharin/2mRNA_100000/two_MRNA_Single_Up_data_Centres_And_Times_100000.pck'
    Load_and_Save(input_file,out_outfile,Moments,Species_To_Store)
    #load(out_outfile ,"Single_up") #plot moment
    
    input_file = "/nfs/datanumerik/people/araharin/2mRNA_100000/two_MRNA_Single_Up_data_0_70_100000.pck"
    out_outfile = '/nfs/datanumerik/people/araharin/2mRNA_100000/two_MRNA_Single_Up_data_Centres_And_Times_0_70_100000.pck'
    Load_and_Save(input_file,out_outfile,Moments,Species_To_Store)

    input_file = "/nfs/datanumerik/people/araharin/2mRNA_100000/two_MRNA_Single_Up_data_70_0_100000.pck"
    out_outfile = '/nfs/datanumerik/people/araharin/2mRNA_100000/two_MRNA_Single_Up_data_Centres_And_Times_70_0_100000.pck'
    Load_and_Save(input_file,out_outfile,Moments,Species_To_Store)

    input_file = "/nfs/datanumerik/people/araharin/2mRNA_100000/two_MRNA_Single_Up_data_70_70_100000.pck"
    out_outfile = '/nfs/datanumerik/people/araharin/2mRNA_100000/two_MRNA_Single_Up_data_Centres_And_Times_70_70_100000.pck'
    Load_and_Save(input_file,out_outfile,Moments,Species_To_Store)

    input_file = "/nfs/datanumerik/people/araharin/2mRNA_100000/two_MRNA_No_Up_data_100000.pck"
    out_outfile = '/nfs/datanumerik/people/araharin/2mRNA_100000/two_MRNA_No_Up_data_Centres_And_Times_100000.pck'
    Load_and_Save(input_file,out_outfile,Moments,Species_To_Store)
    

    #load(out_outfile, "No_up") # plot moment