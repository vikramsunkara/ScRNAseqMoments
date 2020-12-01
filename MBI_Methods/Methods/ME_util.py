'''
Some utility functions.
'''

import numpy as np
import pdb

def generate_moments_2D(states,p,indexes):

	num_States = states.shape[0]

	moments = []

	for index in indexes:
		total_prod = p.copy()
		for i in range(num_States):
			if index[i] != 0:
				total_prod *= np.power(states[i,:],index[i])
		moments.append(np.sum(total_prod))
	return moments

def evaluate_only_Y(x,Y,lam,indexes,prob=False):

	p_Y = np.ones((len(Y)))

	for i in range(len(indexes)):
		index = indexes[i]
		if index[1] != 0:
			p_Y += lam[i]*np.power(x,index[0])*np.power(Y,index[1])
	if prob == True:
		return np.exp(-p_Y)
	else:
		return -p_Y

def Center_Differencing(XX,TT):
	N = len(XX)
	diffs = []
	centers = []
	times = []
	for i in range(N-2):
		diffs.append( np.divide(np.subtract(XX[i+2],XX[i]),(TT[i+2]+TT[i])))
		centers.append(XX[i+1])
		times.append(TT[i+1])
	return diffs, centers, times

def Simple_Differencing(XX,TT):
	N = len(XX)
	diffs = []
	centers = []
	times = []
	for i in range(N-1):
		diffs.append( np.divide(np.subtract(XX[i+1],XX[i]),(TT[i+1]+TT[i])))
		centers.append(XX[i])
		times.append(TT[i])
	return diffs, centers, times

def real_Differencing(XX,TT):
	#Linear
	k_1         = 30.0
	gamma_1     = 1.0
	k_2         = 4.0
	gamma_2     = 1.0


	A = np.zeros((6,6))

	A[1,:] = 	[k_1,-gamma_1, 0, 0, 0, 0]
	A[2,:] =	[0,k_2, -gamma_2, 0, 0, 0]
	A[3,:] =	[0,gamma_1 + 2*k_1, 0, -2*gamma_1, 0, 0]
	A[4,:] =	[0,0, k_1, k_2, -gamma_1 - gamma_2, 0]
	A[5,:] =	[0,k_2, gamma_2, 0, 2*k_2, -2*gamma_2]

	N = len(XX)
	diffs = []
	centers = []
	times = []
	for i in range(N-1):
		diffs.append(A.dot(XX[i]))
		centers.append(XX[i])
		times.append(TT[i])
	return diffs, centers, times

def spline__Differencing(XX,TT,keeps=None):
	from scipy.interpolate import CubicSpline

	XX = np.array(XX).T # species X Obs
	TT = np.array(TT)

	mid_T = (TT[1:] + TT[:-1])/2

	diffs = []
	Weights = []
	second_diff = []
	import pylab as pl
	for i in range(XX.shape[0]):
		cs = CubicSpline(TT,XX[i,:])
		dspl = cs.derivative()
		diffs.append(dspl(TT[1:]))
		ddspl = dspl.derivative()
		second_diff.append(ddspl(mid_T))

		# Introducing Weights
		#Weights.append(np.where(np.abs(diffs[-1]) > 10, 1.0/np.sqrt(XX[i,1:]), 1.0))
		Weights.append(np.divide(1.0, 1.0 + np.abs(second_diff[i])))
		
		#pl.subplot(5,3,i+1)
		#pl.plot(TT[1:], diffs[i])
		#pl.plot(mid_T, np.abs(second_diff[i]),'--')
	#pl.show()
	#pdb.set_trace()

	# We will try to compute the log mapping to see if the derivative is different. (algorithm:arXiv:1709.02003v4)
	import scipy as sp
	import dspy.algorithms as algorithms
	import dspy.observables as observables
	Y = XX[:,1:]
	X = XX[:,:-1]
	p = 1 # maximum order of monomials
	psi = observables.monomials(p)

	YYY = psi(Y)
	XXX = psi(X)

	L_data = (1.0/(TT[-1]-TT[-2]))*sp.linalg.logm(YYY.dot(sp.linalg.pinv(XXX)))

	#L_data = L_data[1:,:][:,1:]
	#der_UpLift = L_data.dot(X)

	#der_UpLift = L_data.dot(XXX)[1:,:] # linear
	der_UpLift = L_data.dot(XXX)[1:X.shape[0]+1,:] # quandratic
	
	"""
	L_data_org = (1.0/(TT[-1]-TT[-2]))*sp.linalg.logm(Y.dot(sp.linalg.pinv(X)))
	der_UpLift_org = L_data_org.dot(X)
	"""
	#pdb.set_trace()
	
	if keeps is not None:
		diffs[0] = [0.0]*len(TT)
		return np.array(diffs)[keeps,:], XX[keeps,:], TT, der_UpLift[keeps,:]
	else:
		return np.array(diffs), XX[:,1:], TT[1:], np.array(Weights), der_UpLift

def normalised_spline_Differencing(XX,TT,Normals,keeps=None):
	from scipy.interpolate import CubicSpline

	XX = np.divide(np.array(XX).T,Normals[:,np.newaxis]) # species X Obs

	TT = np.array(TT)

	diffs = []

	for i in range(XX.shape[0]):
		cs = CubicSpline(TT,XX[i,:])
		dspl = cs.derivative()
		diffs.append(dspl(TT))

	if keeps is not None:
		diffs[0] = [0.0]*len(TT)
		return np.array(diffs)[keeps,:], XX[keeps,:], TT
	else:
		return np.array(diffs), XX, TT


def Generate_Jacobian(Gen,p):
	import dspy.observables as observables

	D = Gen.shape[0]
	N = Gen.shape[1]

	indexes = observables.allMonomialPowers(D,p).astype(np.uint8)

	Jacobian_Var = np.zeros((D,D,D))
	Jacobian_Cont = np.zeros((D,D)) 

	for j in range(D): # Over the Functions
		for k in range(D): # Over the Variables
			for i in range(N):
				# Find all positions which have a k term
				if indexes[k,i] != 0:
					number_terms = np.sum(indexes[k,i]!=0)

					if number_terms == 2:
						temp = indexes[:,i]
						temp[j] = 0
						ind = np.where(temp == 1)[0][0]

						Jacobian_Var[j,k,ind] += Gen[k,i] # Jacobian

					elif number_terms == 1:
						if indexes[k,i] == 2:
							# quadratic
							Jacobian_Var[j,k,k] += 2.0*Gen[k,i]
						else: 
							#linear Reduces to Constant
							if k == j:
								Jacobian_Cont[j,k] += Gen[k,i]
	return Jacobian_Var, Jacobian_Cont

def Evaluate_Jacobian(Jac_var, Jac_Const, x):
	D = Jac_var.shape[0]
	evaluates = np.zeros((D,D))

	for i in range(D):
		for j in range(D):
			evaluates[i,j] = np.dot(Jac_var[i,j,:],x)

	return evaluates + Jac_Const








