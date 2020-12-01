import numpy as np 
import pdb
import pickle
import seaborn as sns
def Load_and_Save(input_filename, out_name):

	f = open(input_filename,'rb')
	input_dic = pickle.load(f)
	f.close()

	N = 1000

	Obs = input_dic['Obs'][1:,:,:]
	Times = np.array(input_dic['Time'])[1:]
	Indexes = np.arange(0,len(Times),1,dtype=np.int)

	pdb.set_trace()
	
	M1 = Obs[:,-2,:N]
	M2 = Obs[:,-1,:N]

	import pylab as pl
	'''
	pl.subplot(1,2,1)
	pl.plot(Times,np.average(M1,axis=1),label='A')
	pl.plot(Times,np.average(M2,axis=1),label='B')
	pl.subplot(1,2,2)
	'''
	pl.figure(num=3,figsize=(7,7))
	pl.suptitle(out_name)
	pl.scatter(M1[-1,:],M2[-1,:],alpha=0.3)
	sns.kdeplot(M1[-1,:],M2[-1,:], gridsize=50,n_levels=6,alpha=0.9)
	pl.plot(np.average(M1,axis=1)[::5],np.average(M2,axis=1)[::5], '-^', color=	'#df8137')
	#pl.xlim([0,70])
	#pl.ylim([0,70])
	pl.axis('equal')
	pl.show()


	Time = np.zeros((len(Times),N))
	Rep  = np.zeros((len(Times),N),dtype=np.int)
	Time += Times[:,np.newaxis]
	Rep  += Indexes[:,np.newaxis]

	M1 = M1.flatten()
	M2 = M2.flatten()
	Rep = Rep.flatten()
	Time = Time.flatten()


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

	input_file = '/Users/sunkara/Nextcloud/Alexia_Data_Share/two_MRNA_Single_Up_data_70_0.pck'
	Load_and_Save(input_file, 'Mono_I_70_0')

	input_file = '/Users/sunkara/Nextcloud/Alexia_Data_Share/two_MRNA_Single_Up_data_0_70.pck'
	Load_and_Save(input_file, 'Mono_I_0_70')

	input_file = '/Users/sunkara/Nextcloud/Alexia_Data_Share/two_MRNA_Single_Up_data_70_70.pck'
	Load_and_Save(input_file, 'Mono_I_70_70')
