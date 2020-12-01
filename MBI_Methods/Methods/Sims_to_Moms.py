###
# Module to read in Stochastic simulations and Generate Moments.
###

import numpy as np 
import pdb
import pickle

def Sims_2_Moms(Sim_Block, Moments):
	'''
	Sim_Block 		numpy array  	(Time, Dim, Repeats)
	Moments 		List 	 		numpy arrays [(Dim,),...]
	'''

	num_Moms = len(Moments)

	Moms = np.zeros((Sim_Block.shape[0],len(Moments)+1))

	Moms[:,0] = 1.0

	for i in range(num_Moms):
		Powered = np.power(Sim_Block,Moments[i][np.newaxis,:,np.newaxis])
		Produced = np.prod(Powered,axis=1)
		Moms[:,i+1] = np.average(Produced,axis=1)

	return Moms

def Load_and_Save(input_filename, output_filename, Moments, keep_species = None):

	f = open(input_filename,'rb')
	input_dic = pickle.load(f)
	f.close()

	if keep_species is None: # process all species
		Moms = Sims_2_Moms(input_dic['Obs'].astype(np.float),Moments)
	else:
		Moms = Sims_2_Moms(input_dic['Obs'][:,keep_species,:].astype(np.float),Moments)

	f = open(output_filename,"wb")
	pickle.dump({'centers': Moms, 'Times': input_dic['Time']},f)
	f.close()

	print('Saved to %s'%(output_filename))


if __name__ == '__main__':

	'''
	# Testing Sims_2_Moms
	A = np.random.rand(2,3,4)
	Moments = [np.array([0,2,1])]
	Moms = Sims_2_Moms(A,Moments)
	print(Moms[0,1])
	print(np.average(np.power(A[0,1,:],2.0)*A[0,2,:]))
	pdb.set_trace()
	'''

	indexes = [(1,0),(0,1),(2,0),(1,1),(0,2),(3,0),(2,1),(1,2),(0,3),(4,0),(3,1),(2,2),(1,3),(0,4)]
	Moments = [np.array(item) for item in indexes]
	Species_To_Store = np.array([False,False,False,False,True,True])

	'''
	input_file = '/Users/sunkara/dev/GRN_Caus/Models/two_MRNA_No_Up_data_Test.pck'
	out_outfile = '/Users/sunkara/dev/GRN_Caus/Data/two_MRNA_No_Up_test_data_Centres_And_Times.pck'
	
	Load_and_Save(input_file,out_outfile,Moments,Species_To_Store)
	'''

	
	input_file = '/Users/sunkara/dev/GRN_Caus/Data/two_MRNA_Double_Up_data.pck'
	out_outfile = '/Users/sunkara/dev/GRN_Caus/Data/two_MRNA_Double_Up_data_Centres_And_Times.pck'
	
	Load_and_Save(input_file,out_outfile,Moments,Species_To_Store)


	input_file = '/Users/sunkara/dev/GRN_Caus/Data/two_MRNA_Single_Up_data.pck'
	out_outfile = '/Users/sunkara/dev/GRN_Caus/Data/two_MRNA_Single_Up_data_Centres_And_Times.pck'
	
	Load_and_Save(input_file,out_outfile,Moments,Species_To_Store)

	input_file = '/Users/sunkara/dev/GRN_Caus/Data/two_MRNA_No_Up_data.pck'
	out_outfile = '/Users/sunkara/dev/GRN_Caus/Data/two_MRNA_No_Up_data_Centres_And_Times.pck'
	
	Load_and_Save(input_file,out_outfile,Moments,Species_To_Store)


	input_file = '/Users/sunkara/dev/GRN_Caus/Data/two_MRNA_Single_Up_data_70_0.pck'
	out_outfile = '/Users/sunkara/dev/GRN_Caus/Data/two_MRNA_Single_Up_data_70_0_Centres_And_Times.pck'
	
	Load_and_Save(input_file,out_outfile,Moments,Species_To_Store)

	input_file = '/Users/sunkara/dev/GRN_Caus/Data/two_MRNA_Single_Up_data_0_70.pck'
	out_outfile = '/Users/sunkara/dev/GRN_Caus/Data/two_MRNA_Single_Up_data_0_70_Centres_And_Times.pck'
	
	Load_and_Save(input_file,out_outfile,Moments,Species_To_Store)

	input_file = '/Users/sunkara/dev/GRN_Caus/Data/two_MRNA_Single_Up_data_70_70.pck'
	out_outfile = '/Users/sunkara/dev/GRN_Caus/Data/two_MRNA_Single_Up_data_70_70_Centres_And_Times.pck'
	
	Load_and_Save(input_file,out_outfile,Moments,Species_To_Store)
	
	pdb.set_trace()

