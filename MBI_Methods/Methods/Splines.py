# Computing the derivative for the splined data
import numpy as np

def spline_Differencing(XX, TT):
	from scipy.interpolate import CubicSpline

	# redundant code, do change
	if len(TT) == XX.shape[0]:
		XX = np.array(XX).T # convert to species X Obs

	TT = np.array(TT)

	diffs = []

	for i in range(XX.shape[0]):
		cs = CubicSpline(TT,XX[i,:])
		dspl = cs.derivative()
		diffs.append(dspl(TT[1:]))

	Arr = np.array(diffs)

	Arr[0,:] = 0.0 # correction for the constant.
	
	return Arr.T 

# Building the linear operator for the linear MBI method
def Build_Design_Matrix(XX, Design_Blocks):

	## XX is Obs x Species
	num_obs = XX.shape[0]-1
	num_moms = Design_Blocks.shape[1]
	num_params = Design_Blocks.shape[0]

	Design_Mat = np.zeros((num_obs,num_moms,num_params))

	for t in range(num_obs):
		for k in range(num_params):
			Design_Mat[t,:,k] = Design_Blocks[k,:,:].dot(XX[t,:])

	return Design_Mat