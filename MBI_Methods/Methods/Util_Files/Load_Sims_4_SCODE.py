import numpy as np 
import pdb
import pickle

def Load_and_Save(input_filename, out_name):

	f = open(input_filename,'rb')
	input_dic = pickle.load(f)
	f.close()

	N = 1000

	Obs = input_dic['Obs'][1:,:,:]
	Times = np.array(input_dic['Time'])[1:]
	Indexes = np.arange(0,len(Times),1,dtype=np.int)

	M1 = Obs[:,-2,:N]
	M2 = Obs[:,-1,:N]

	Time = np.zeros((len(Times),N))
	Rep  = np.zeros((len(Times),N),dtype=np.int)
	Time += Times[:,np.newaxis]
	Rep  += Indexes[:,np.newaxis]

	M1 = M1.flatten()
	M2 = M2.flatten()
	Rep = Rep.flatten()
	Time = Time.flatten()

	Time = np.divide(Time,np.amax(Time))

	M = np.vstack((M1,M2)).astype(np.int)
	T = np.column_stack((Rep,Time))

	'''
	np.savetxt('Data/Moms_%s_N_%d.txt'%(out_name,N*len(Times)),M,delimiter='	',fmt='%d')
	np.savetxt('Data/Time_%s_N_%d.txt'%(out_name,N*len(Times)),T,delimiter='	',fmt='%0.3f')
	'''

	M_4_Sin = np.vstack((M1,M2,Rep))
	np.savetxt('Data/Moms_4_SIN_%s_N_%dK.txt'%(out_name,(N*len(Times))/1000),M_4_Sin.T,delimiter=',',header='Gene1,Gene2,h',fmt='%0.3f')


if __name__ == '__main__':

	'''
	input_file = '/Users/sunkara/dev/GRN_Caus/Models/two_MRNA_No_Up_data.pck'
	Load_and_Save(input_file, 'No_Up')
	input_file = '/Users/sunkara/dev/GRN_Caus/Models/two_MRNA_Double_Up_data.pck'
	Load_and_Save(input_file, 'Double_Up')
	input_file = '/Users/sunkara/dev/GRN_Caus/Models/two_MRNA_Single_Up_data.pck'
	Load_and_Save(input_file, 'Single_Up')
	'''
	
	input_file = '/Users/sunkara/dev/GRN_Caus/Models/two_MRNA_No_Up_data.pck'
	Load_and_Save(input_file, 'No_Up')
