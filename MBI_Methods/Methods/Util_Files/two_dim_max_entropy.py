## Maximal entropy for higher dimensions.
import numpy as np 
import pdb
import pylab as pl
from scipy.optimize import fmin


def give_rank2_tensor(indexes,X,Y):
	return np.outer(np.power(X,indexes[0]),np.power(Y,indexes[1]))

def build_prob(lam,index_list,X,Y,return_log=False):
	#p = np.exp(-lam[0])*np.ones((len(X),len(Y)))
	temp = np.ones((len(X),len(Y))) + lam[0]
	for i in range(len(lam)-1):
		#p *= np.exp(-lam[i+1]*give_rank2_tensor(index_list[i+1],X,Y))
		temp += lam[i+1]*give_rank2_tensor(index_list[i+1],X,Y)
	#return p 
	
	if return_log == True:
		return -temp
	else:
		return np.exp(-temp)

def give_Moment(p,indexes,X,Y):
	terms = give_rank2_tensor(indexes,X,Y)
	return np.sum(np.multiply(terms,p))

def sum_indexes(ind1,ind2):
	return(ind1[0]+ind2[0], ind1[1]+ind2[1])

def build_J(lam,index_list,X,Y,return_p=False):
	D = len(index_list)
	J = np.zeros((D,D))
	p = build_prob(lam,index_list,X,Y)
	# normalise
	p = np.divide(p,np.sum(p))
	for i in range(D):
		for j in range(D):
			if j >= i :
				new_ind = sum_indexes(index_list[i],index_list[j])
				J[i,j] = give_Moment(p,new_ind,X,Y)
				J[j,i] = J[i,j]

	pl.clf()
	pl.imshow(p,origin="low")
	pl.draw()
	pl.pause(0.001)
	pdb.set_trace()
	if return_p == True:
		return -J, p
	else:
		return -J

def step_forward(J,mu):
	return np.linalg.inv(J).dot(mu)

def progress(mu_target,index_list,tol,X,Y,lam_0=None):

	if lam_0 is None:
		lam_0 = np.zeros((len(index_list)))
		# testing if a better initial value helps convergence
		e_x = mu_target[1]
		e_y = mu_target[2]
		e_x_x = mu_target[3]
		e_y_y = mu_target[5]
		c_x_x = e_x_x - e_x*e_x
		c_y_y = e_y_y - e_y*e_y

		lam_0[0] = 0.5*(((e_x**2)/c_x_x) + ((e_y**2)/c_y_y))
		lam_0[1] = -e_x/(2.0*c_x_x)
		lam_0[2] = -e_y/(2.0*c_y_y)
		lam_0[3] = 1.0/(2.0*c_x_x)
		lam_0[5] = 1.0/(2.0*c_y_y)

	print(lam_0)
	#lam_0[0] = np.log(len(X)*len(Y))-1.0

	# find the highest ordered terms
	"""	highest_terms = []
	max_in = np.amax(index_list) 
	print "max_Index is %d"%(max_in)
	for i in range(len(index_list)):
		if np.sum(index_list[i]) >= max_in:
			highest_terms.append(i) 
	print "highest terms %s"%(str(highest_terms))"""
	#first step
	J_0 = build_J(lam_0,index_list,X,Y)
	mu_H = -J_0[0,:]
	delta_mu = mu_target - mu_H
	eps = delta_mu.T.dot(delta_mu)
	lam = lam_0

	_counter = 0

	while (eps > tol) and (_counter < 1000):
		#repeater step
		lam +=	step_forward(J_0,delta_mu)
		print(lam)
		"""		for ind in highest_terms:
			if lam[ind] < 0.0:
				lam[ind] = 0.0 
				#lam[ind] = -lam[ind] # we will flip the sign."""
		J_0 = build_J(lam,index_list,X,Y)
		mu_H = -J_0[0,:]
		delta_mu = mu_target - mu_H
		eps = delta_mu.T.dot(delta_mu)
		#print eps, _counter

		_counter += 1

	if _counter >= 1000:
		print('[WARNING] The entropy solver did not converge')

	return lam

def progress_test(mu_target,index_list,tol,X,Y,lam_0=None):

	from scipy.optimize import minimize

	if lam_0 is None:
		lam_0 = np.zeros((len(index_list)))
		# testing if a better initial value helps convergence
		lam_0[0] = 1.0
		'''
		e_x = mu_target[1]
		e_y = mu_target[2]
		e_x_x = mu_target[3]
		e_y_y = mu_target[5]
		c_x_x = e_x_x - e_x*e_x
		c_y_y = e_y_y - e_y*e_y

		lam_0[0] = 0.5*(((e_x**2)/c_x_x) + ((e_y**2)/c_y_y))
		lam_0[1] = -e_x/(2.0*c_x_x)
		lam_0[2] = -e_y/(2.0*c_y_y)
		lam_0[3] = 1.0/(2.0*c_x_x)
		lam_0[5] = 1.0/(2.0*c_y_y)
		'''
	#print lam_0

	def min_func(x):
		P = build_prob(x,index_list,X,Y,return_log=False)
		#pdb.set_trace()
		P = np.divide(P,np.sum(P))
		mu_H = give_all_moments(P,index_list,X,Y)
		delta_mu = mu_target - mu_H
		return delta_mu.T.dot(delta_mu)
	
	res = minimize(min_func, lam_0, method='Nelder-Mead', tol=1e-6)
	return res.x


def compute_2D_maximal_entropy_appx(Moms, index_list, X,Y, tol=1e-6, lam_0 = None):
	#pl.ion()
	#lam = progress(Moms,index_list,tol,X,Y,lam_0=None)
	lam = progress_test(Moms,index_list,tol,X,Y,lam_0=None)
	P = build_prob(lam,index_list,X,Y)
	return P, lam


def find_lam_Zero(lam, index_list, X, Y):
	log_coeffs = build_prob(lam, index_list, X, Y, return_log=True)
	def fun(x):
		new_coeff = log_coeffs - x
		return np.power(1.0 - np.sum(np.exp(new_coeff)),2.0)
	s = fmin(fun, 10.0,disp=False)
	return lam[0] - s[0]

def Normalise_Density(P):
	normalise_by = np.sum(P) 
	return -np.log(normalise_by) 

def give_all_moments(p,index_list,X,Y):
	Mom_H = []
	for index in index_list:
		Mom_H.append(give_Moment(p,index,X,Y))
	return np.array(Mom_H)


if __name__ == '__main__':
	################################
	### Example Code
	################################
	#mu_target = np.array([1.0,	33.25,	94.26,	3203.24,	1133.69,	9211.15,	318830.26,	39573.27,	111364.18,	928245.47])  #1129342.88,	1412921.52])
	mu_target = np.array([1.00000000e+00, 3.98733000e+01, 4.00583000e+01, 1.66115190e+03,
       1.59834800e+03, 1.67595530e+03, 7.18815723e+04, 6.66160452e+04,
       6.69137828e+04, 7.28045963e+04])
	index_list = [	(0,0),		(1,0),		‚(0,1),		(2,0),		(1,1),		(0,2),		(3,0),		(2,1),		(1,2),		(0,3)]		#(2,2),			(4,0)]

	tol = 1e-9
	X = np.arange(0,120,1, dtype=np.float64)
	Y = np.arange(0,120,1, dtype=np.float64)

	'''
	lam = progress(mu_target,index_list,tol,X,Y)
	P = build_prob(lam,index_list,X,Y)
	'''

	P, lam = compute_2D_maximal_entropy_appx(mu_target,index_list,X,Y,tol=tol)

	pdb.set_trace()

	pl.imshow(P.T,origin='low',aspect='auto')
	pl.show()

	pdb.set_trace()