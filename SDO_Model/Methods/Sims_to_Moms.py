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

def Load_and_Save(input_filename, output_filename, Moments, keeps =None):

	f = open(input_filename,'rb')
	input_dic = pickle.load(f)
	f.close()

	if keeps is None:
		Moms = Sims_2_Moms(input_dic['Obs'],Moments)
	else:
		Moms = Sims_2_Moms(input_dic['Obs'][:,keeps,:],Moments)

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

	input_file = '/Users/sunkara/dev/GRN_Caus/Models/two_MRNA_data.pck'
	out_outfile = '/Users/sunkara/dev/GRN_Caus/Data/MRNA_model_1_Centres_And_Times_v1.pck'

	indexes = [(1,0),(0,1),(2,0),(1,1),(0,2),(3,0),(2,1),(1,2),(0,3)]
	Moments = [np.array(item) for item in indexes]
	keeps = np.array([False,False,False,False,True,True])
	Load_and_Save(input_file,out_outfile,Moments,keeps)

	pdb.set_trace()

