
import numpy as np 
import pdb
import pickle
import pylab as pl
from scipy.stats import linregress as LR

def Load_Sims(input_filename, Species = None):

	f = open(input_filename,'rb')
	input_dic = pickle.load(f)
	f.close()

	data = input_dic['Obs']

	pl.figure(1)
	pl.subplot(1,3,1)
	pl.scatter(data[-1,-1,:],data[-1,-2,:])
	E_M_1 = np.average(data[:,-2,:],axis=1)
	E_M_2 = np.average(data[:,-1,:],axis=1)
	E_M_M_1 = np.average(np.power(data[:,-2,:],2.0),axis=1)
	E_M_M_2 = np.average(np.power(data[:,-1,:],2.0),axis=1)

	pl.subplot(1,3,2)
	pl.plot(input_dic['Time'],E_M_1,label='(1,0)')
	pl.plot(input_dic['Time'],E_M_2,label='(0,1)')
	pl.legend()
	pl.subplot(1,3,3)
	pl.plot(input_dic['Time'],E_M_M_1,label='(2,0)')
	pl.plot(input_dic['Time'],E_M_M_2,label='(0,2)')
	pl.legend()
	#pl.show()
	#pdb.set_trace()
	

	if Species is None:
		return input_dic['Obs'][-1,:,:]
	else:
		return input_dic['Obs'][-1,Species,:]

def Load_Sims_4_Exp(input_filename, time_slice = -1):

	f = open(input_filename,'rb')
	input_dic = pickle.load(f)
	f.close()
	
	data = input_dic['Obs']

	E_M_1 = np.average(data[:,-2,:],axis=1)
	E_M_2 = np.average(data[:,-1,:],axis=1)

	E_G_1 = np.average(data[:,0,:],axis=1)
	E_G_2 = np.average(data[:,1,:],axis=1)

	return E_M_1, E_M_2, data[-1,-2,:], data[-1,-1,:], input_dic['Time'], E_G_1, E_G_2, data[time_slice,-2:,:]



def Gene_cond_MRNA(G_X):

	u_X = np.unique(G_X[1,:])

	E_G_con_X = []

	A = np.zeros((2,u_X[-1]+1),np.int)

	#pdb.set_trace()
	for i in range(G_X.shape[1]):
		A[G_X[0,i],G_X[1,i]] += 1


	for x in u_X:
		x_active = G_X[1,:] == x
		E_G_con_X.append(np.average(G_X[0,x_active]))

	
	stuff = LR(u_X,E_G_con_X)  

	return u_X, np.array(E_G_con_X), A, stuff[0],stuff[1], np.average(G_X[0,:])


def Compute_Cond_Prob(input_filename, Species):
	return Gene_cond_MRNA(Load_Sims(input_filename,Species).astype(np.int))


if __name__ == '__main__':

	name = 'No'

	Species_G1_A = np.array([True,False,False,False,True,False])
	Species_G2_B = np.array([False,True,False,False,False,True])

	input_file = '/Users/sunkara/dev/GRN_Caus/Models/Parallel/two_MRNA_%s_Up_data_vik_1.pck'%(name)
	#input_file = '/Users/sunkara/dev/GRN_Caus/Models/two_MRNA_%s_Up_data.pck'%(name)

	u_A, u_G_cond_A, A, slope_A, inter_A, G_A = Compute_Cond_Prob(input_file,Species_G1_A)
	u_B, u_G_cond_B, B, slope_B, inter_B, G_B = Compute_Cond_Prob(input_file,Species_G2_B)

	p_G_A = np.divide(A,np.sum(A))
	p_A = np.sum(p_G_A,axis=0)
	p_G_cond_A = np.divide(p_G_A,p_A[np.newaxis,:],where=p_A[np.newaxis,:] > 0)
	E_G_A = p_G_cond_A[1,:]

	p_G_B = np.divide(B,np.sum(B))
	p_B = np.sum(p_G_B,axis=0)
	p_G_cond_B = np.divide(p_G_B,p_B[np.newaxis,:],where=p_B[np.newaxis,:] > 0)
	E_G_B = p_G_cond_B[1,:]

	pl.figure(2)
	pl.suptitle('%s UP'%(name))
	pl.subplot(1,3,1)
	pl.xlabel('mRNA_A')
	pl.ylabel('E[Gene_A | mRNA_A]')
	pl.plot(u_A, u_G_cond_A, 'rx')
	pl.plot(u_A , u_A*slope_A + inter_A,'r--')
	

	pl.subplot(1,3,2)
	pl.xlabel('mRNA_B')
	pl.ylabel('E[Gene_B | mRNA_B]')
	pl.plot(u_B, u_G_cond_B, 'gx')
	pl.plot(u_B , u_B*slope_B + inter_B,'g--')

	pl.subplot(1,3,3)
	pl.plot(u_A , u_A*slope_A + inter_A,'r--')
	pl.plot(u_B , u_B*slope_B + inter_B,'g--')

	'''
	pl.subplot(1,3,2)
	pl.imshow(A,aspect='auto',origin='low')
	pl.subplot(1,3,3)
	pl.imshow(B,aspect='auto',origin='low')
	'''
	pl.show()
	pdb.set_trace()

