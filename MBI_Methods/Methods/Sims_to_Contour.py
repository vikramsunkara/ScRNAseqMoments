import numpy as np 

def make_contour_matrix(Sims_X_Y):

	X_max = 1 + np.amax(Sims_X_Y[0,:])
	Y_max = 1 + np.amax(Sims_X_Y[1,:])

	Z = np.zeros((X_max,Y_max))

	for i in range(Sims_X_Y.shape[1]):
		x,y = Sims_X_Y[:,i]
		Z[x,y] += 1

	Z_prob = np.divide(Z,Sims_X_Y.shape[1])

	return range(X_max), range(Y_max), Z_prob.T