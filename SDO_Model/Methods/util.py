import sympy as sy
import pdb
import math
import numpy as np

def decompose(expr,variables):
	item = sy.Poly(expr, *variables).as_dict()
	pos = item.keys()

	return pos[0], item[pos[0]]

def make_poly(degree_list,symbols):
	sym_form = 1.0
	for i in range(len(degree_list)):
		#pdb.set_trace()
		sym_form = sym_form*(symbols[i]**degree_list[i])
	print(sym_form)

# auxiliary functions
def nchoosek(n, k):
    '''
    Computes binomial coefficients.
    '''
    return math.factorial(n)//math.factorial(k)//math.factorial(n-k) # integer division operator

def nextMonomialPowers(x):
    '''
    Returns powers for the next monomial. Implementation based on John Burkardt's MONOMIAL toolbox, see
    http://people.sc.fsu.edu/~jburkardt/m_src/monomial/monomial.html.
    '''
    m = len(x)
    j = 0
    for i in range(1, m): # find the first index j > 1 s.t. x[j] > 0
        if x[i] > 0:
            j = i
            break
    if j == 0:
        t = x[0]
        x[0] = 0
        x[m - 1] = t + 1
    elif j < m - 1:
        x[j] = x[j] - 1
        t = x[0] + 1
        x[0] = 0
        x[j-1] = x[j-1] + t
    elif j == m - 1:
        t = x[0]
        x[0] = 0
        x[j - 1] = t + 1
        x[j] = x[j] - 1
    return x


def allMonomialPowers(d, p):
    '''
    All monomials in d dimensions of order up to p.
    '''
    # Example: For d = 3 and p = 2, we obtain
    #[[ 0  1  0  0  2  1  1  0  0  0]
    # [ 0  0  1  0  0  1  0  2  1  0]
    # [ 0  0  0  1  0  0  1  0  1  2]]
    n = nchoosek(p + d, p) # number of monomials
    x = np.zeros(d) # vector containing powers for the monomials, initially zero
    c = [] # matrix containing all powers for the monomials
    for i in range(1, n):
        c.append(tuple(nextMonomialPowers(x).astype(np.int)))
    return c
