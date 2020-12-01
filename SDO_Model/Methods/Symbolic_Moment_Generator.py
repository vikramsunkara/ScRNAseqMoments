import sympy as sy
import numpy as np
import pdb

def decompose(expr,variables):
	item = sy.Poly(expr, *variables).as_dict()
	pos = list(item.keys())
	#pdb.set_trace()
	return pos[0], item[pos[0]]

def make_poly(degree_list,symbols):
	sym_form = 1.0
	for i in range(len(degree_list)):
		#pdb.set_trace()
		sym_form = sym_form*(symbols[i]**degree_list[i])
	return sym_form


class Symbolic_Moments :

	def __init__(self,symbols,propensities,transitions,binary_species):

		self.symbols 		= symbols
		self.propensities 	= propensities
		self.transitions  	= transitions

		self.binary_species = binary_species

		self.T = None

	def Set_Moment_Degree_Function(self,T):
		self.T = T

	def Compute_Moments_for_Reaction(self, reaction_num, Moment_list):

		Contributions = []

		for Mom_degrees in Moment_list:
			T_degrees = self.T(* (self.symbols + [Mom_degrees]))
			der = sy.Symbol('der')
			der_list = []
			all_terms = []
			shift = np.array(self.symbols) + np.array(self.transitions[reaction_num])
			der_list.append( (T_degrees(*shift) - T_degrees(*self.symbols))*self.propensities[reaction_num]) # Formula from Engblom 06

			term = sy.simplify(der_list[-1].expand()).expand()
			terms = term.args

			if 'Add' in sy.srepr(term):
				for item in terms:
					all_terms.append(item)
			else:
				all_terms.append(term)
			Contributions.append(all_terms)

		return Contributions

	def Match_Terms_To_LookUp(self,Contributions,hash_lookUp,normalise_coeff):

		Maps = []
		for all_terms in Contributions:
			Vec = [0]*len(hash_lookUp)
			Extra = [0]
			for term in all_terms:
				if term != 0:
					ind, coeff = decompose(term, self.symbols)
					
					# We need to correct for Binary variables
					for k in range(len(ind)):
						if self.binary_species[k] == True:
							if ind[k] > 1:
								ind_list = list(ind)
								ind_list[k] = 1
								ind = tuple(ind_list)

					if ind in hash_lookUp:
						Vec[hash_lookUp[ind]] += coeff/normalise_coeff
					else:
						Extra.append([ind,1.0])
			Maps.append(Vec)
		return np.array(Maps) # this will give a matrix (degrees X LookUp)

	def Compute_Moment(self,degrees,hash_lookUp):

		T_degrees = self.T(* (self.symbols + [degrees]))

		der = sy.Symbol('der')
		der_list = []

		all_terms = []
		for i in range(len(self.transitions)):
			shift = np.array(self.symbols) + np.array(self.transitions[i])
			der_list.append( (T_degrees(*shift) - T_degrees(*self.symbols))*propensities[i]) # Formula from Engblom 06

			term = sy.simplify(der_list[-1].expand()).expand()
			terms = term.args

			if 'Add' in sy.srepr(term):
				for item in terms:
					all_terms.append(item)
			else:
				all_terms.append(term)

		# Match the terms of the derivative to the hash of the moments of interest.
		Vec = [0]*len(hash_lookUp)
		Extra = [0]
		for term in all_terms:
			if term != 0:
				ind, coeff = decompose(term, var)
				
				# We need to correct for Binary variables
				for k in range(len(ind)):
					if self.binary_species[k] == True:
						if ind[k] > 1:
							ind_list = list(ind)
							ind_list[k] = 1
							ind = tuple(ind_list)

				if ind in hash_lookUp:
					Vec[hash_lookUp[ind]] += coeff
				else:
					Extra.append([ind,coeff])

		print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
		#print zip(range(9),Vec)
		print(Vec)
		print(Extra)
		return Vec, Extra

if __name__ == "__main__":

	A, B = sy.var('A B')
	k_0, k_1, k_2, k_3, k_4, k_5, k_6, k_7, k_8, k_9, k_10, k_11, k_12, k_13 = sy.symbols('k_0 , k_1, k_2, k_3, k_4, k_5, k_6, k_7, k_8, k_9, k_10, k_11, k_12, k_13', constant=True)
	var = [A,B]
	reaction_coeffs = [k_0,k_1, k_2, k_3, k_4, k_5, k_6, k_7, k_8, k_9, k_10, k_11, k_12]

	def T(A,B,power):
		return lambda A,B : (A**power[0])*(B**power[1])

	## ALl reactions
	propensities =[k_0, k_1, k_2*A, k_3*B, k_4*A, k_5*B, k_6*A, k_7*B, k_8*A*B, k_9*A*B, k_10*A*B, k_11*A*B, k_12*A*B]
	transitions = [ 	(1,0),   #0
						(0,1),   #1
						(-1,0),  #2
						(0,-1),  #3
						(-1,1),  #4
						(1,-1),  #5
						(1,0),   #6
						(0,1),   #7
						(-1,0),  #8
						(0,-1),  #9
						(-1,-1), #10
						(-1,1),  #11
						(1,-1)   #12
					]
	reaction_name = [   '* -> A',	#0
						'* -> B',	#1
						'A ->  *',	#2
						'B -> *',	#3
						'A -> B',	#4
						'B -> A',	#5
						'A -> 2 A',	#6
						'B -> 2 B',	#7
						'A + B -> B',	#8
						'A + B -> A',	#9
						'A + B -> *',	#10
						'A + B -> 2B',	#11
						'A + B -> 2A'	#12
					]
	keeps = np.array([0,1,3,6,11],dtype=np.int)

	if len(keeps) > 0:
		pdb.set_trace()
		propensities = [propensities[i] for i in keeps]
		transitions = [transitions[i] for i in keeps]
		reaction_coeffs = [reaction_coeffs[i] for i in keeps]

	Moment_Obj = Symbolic_Moments(var,propensities,transitions,np.array([False,False]))
	Moment_Obj.Set_Moment_Degree_Function(T)

	hash_lookUp = {(0,0):0, (1,0):1, (0,1):2, 
					(2,0):3, (1,1):4, (0,2):5, (2,2):6 , 
					(3,0):7, (2,1):8, (1,2):9, (0,3):10, 
					(4,0):11, (3,1):12, (2,2):13, (1,3):14, (0,4):15,
					#(5,0):16, (4,1):17, (3,2):18, (2,3):19, (1,4):20, (0,5):21
					}

	#Check_Moments = [(1,0), (0,1), (2,0), (1,1), (0,2), (3,0), (2,1), (1,2), (0,3), (4,0), (3,1), (2,2), (1,3), (4,0)]
	Check_Moments = [(0,0),(1,0), (0,1), (2,0), (1,1), (0,2)]

	Reaction_Mapping_Mat = []
	Reaction_Labels = []
	for i in range(len(propensities)):
		stuff = Moment_Obj.Compute_Moments_for_Reaction(i, Check_Moments)
		Reaction_Labels.append(stuff)
		print('____ Reaction %d ____ %s'%(keeps[i],reaction_name[keeps[i]]))
		print(stuff)
		Maps = Moment_Obj.Match_Terms_To_LookUp(stuff,hash_lookUp,reaction_coeffs[i])
		print(Maps)
		Reaction_Mapping_Mat.append(Maps)
		#pdb.set_trace()
	'''
	f = open("reaction_Maps.pck","wb")
	pickle.dump({"Maps": Reaction_Mapping_Mat, "Labels": Reaction_Labels},f)
	f.close()
	'''

	#for degrees in Check_Moments:
	#	Moment_Obj.Compute_Moment(degrees,hash_lookUp)

	



