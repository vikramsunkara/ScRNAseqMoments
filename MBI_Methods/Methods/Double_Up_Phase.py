import numpy as np
import pdb
import pylab as pl
import seaborn as sns

sigma_1 = 0.125 # Adjusted it so that we don't use the true mean
sigma_2 = 0.5
sigma_3 = 0.125 # Adjusted it so that we don't use the true mean
sigma_4 = 0.5
rho_1 = 4.75
rho_2 = 1.0
rho_3 = 4.75
rho_4 = 1.0
theta = 5.0
delta = 0.1
k = 5.0

from Sims_to_Cond_Exp import Compute_Cond_Prob, Load_Sims_4_Exp

name = 'No'

#input_file_A = '/Users/sunkara/dev/GRN_Caus/Models/two_MRNA_%s_Up_data.pck'%(name)
input_file_A = '/Users/sunkara/dev/GRN_Caus/Models/Parallel/two_MRNA_%s_Up_data_vik.pck'%(name)
input_file = '/Users/sunkara/dev/GRN_Caus/Models/Parallel/two_MRNA_%s_Up_data_vik_0.pck'%(name)

Species_G1_A = np.array([True,False,False,False,True,False])
Species_G2_B = np.array([False,True,False,False,False,True])

u_A, u_G_cond_A, A, slope_A, inter_A, G_A = Compute_Cond_Prob(input_file,Species_G1_A)
u_B, u_G_cond_B, B, slope_B, inter_B, G_B = Compute_Cond_Prob(input_file,Species_G2_B)

E_M_A, E_M_B, s_A, s_B, T, E_G_A, E_G_B, sims_A_B = Load_Sims_4_Exp(input_file)
_, _, _, _, _, _, _, sims_A_B = Load_Sims_4_Exp(input_file_A)

#AA = np.arange(0,np.amax(u_A)+1,1).astype(np.int)
#BB = np.arange(0,np.amax(u_B)+1,1).astype(np.int)
AA = np.arange(0,70,1).astype(np.int)
BB = np.arange(0,70,1).astype(np.int)

E_G1_Cond_A = AA.astype(np.float)*slope_A + inter_A
E_G2_Cond_B = BB.astype(np.float)*slope_B + inter_B

E_G1_Cond_A = np.where(E_G1_Cond_A >1, 1, E_G1_Cond_A)
E_G1_Cond_A = np.where(E_G1_Cond_A <0, 0, E_G1_Cond_A)

E_G2_Cond_B = np.where(E_G2_Cond_B >1, 1, E_G2_Cond_B)
E_G2_Cond_B = np.where(E_G2_Cond_B <0, 0, E_G2_Cond_B)

#G_A = 0.38
#G_B = 0.38

def der_A_B(A,B):

	#dE_A_dt =  rho_1*(1.0-E_G1_Cond_A[A]) + rho_2*E_G1_Cond_A[A]- delta*A
	#dE_B_dt =  rho_3*(1.0-E_G2_Cond_B[B]) + rho_4*E_G2_Cond_B[B] - delta*B

	dE_A_dt =  rho_1*(1.0-G_A) + rho_2*G_A- delta*A
	dE_B_dt =  rho_3*(1.0-G_B) + rho_4*G_B - delta*B  

	return [dE_A_dt,dE_B_dt]

der = []
Cord = []
for a in AA[::4]:
	for b in BB[::4]:
		Cord.append([a,b])
		der.append(der_A_B(a,b))

der = np.array(der)
Cord = np.array(Cord)

'''
pl.figure(num=1,figsize=(15,4))
pl.suptitle(name)
pl.subplot(1,3,1)
pl.quiver(Cord[:,0],Cord[:,1], der[:,0], der[:,1])
pl.xlim([0,65])
pl.ylim([0,65])
#pl.axis('equal')
pl.subplot(1,3,2)
pl.scatter(s_A,s_B,alpha=0.3)
pl.xlim([0,65])
pl.ylim([0,65])
#pl.axis('equal')
pl.subplot(1,3,3)
pl.plot(E_M_A[::5],E_M_B[::5], '-')
pl.xlim([0,65])
pl.ylim([0,65])
#pl.axis('equal')


pl.figure(num=2,figsize=(20,4))
pl.suptitle(name)
pl.subplot(1,4,1)
pl.plot(E_M_A,E_M_B, 'o-')
pl.axvline(x=E_M_A[-1])
pl.axhline(y=E_M_B[-1])
pl.subplot(1,4,2)
pl.plot(np.divide(E_M_A[1:]-E_M_A[:-1], T[1:]-T[:-1]),'-x')
pl.subplot(1,4,3)
pl.plot(T,E_G_A)
pl.plot(T,E_G_B)
pl.subplot(1,4,4)
pl.plot(u_A, u_G_cond_A,'gx')
pl.plot(AA, E_G1_Cond_A,'g--')
pl.plot(u_B, u_G_cond_B,'rx')
pl.plot(BB, E_G2_Cond_B,'r--')
'''

pl.figure(num=3,figsize=(7,7))
#pl.figure(3)
pl.suptitle(name)
pl.quiver(Cord[:,0],Cord[:,1], der[:,0], der[:,1],alpha=0.2,label='(derE[A(60)],derE[B(60)])')
pl.scatter(s_A,s_B,alpha=0.3, label= '(A(60),B(60))')
sns.kdeplot(sims_A_B[0,:],sims_A_B[1,:], gridsize=50,n_levels=6,alpha=0.9)
pl.plot(E_M_A[::5],E_M_B[::5], '-^', color=	'#df8137', label='(E[A(t)],E[B(t)])', alpha=0.8)
#pl.axis('equal')
pl.legend(loc=4)

pl.xlim([0,70])
pl.ylim([0,70])

pl.show()

pdb.set_trace()



