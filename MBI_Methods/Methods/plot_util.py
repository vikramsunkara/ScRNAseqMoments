import numpy as np
import pickle
import pylab as pl
import pdb
from .Regulatory_Rules import Give_Regulatory_Network

def plot_Fit_and_Data(T, Moms, Fit_Moms, indexes, name = 'None', fig_num=1, show=True, pdf = None):

    ax = pl.figure( figsize=(8,8))
    pl.suptitle(name)
    
    pl.subplot(2,2,1)
    for i in range(2): pl.plot(T,Moms[:,i+1],label=str(indexes[i+1]))
    for i in range(2): pl.plot(T,Fit_Moms[:,i+1],'--')
    pl.xlabel('Time')
    pl.ylabel('Order One Moms')
    pl.legend()

    pl.subplot(2,2,2)
    for i in range(3): pl.plot(T,Moms[:,i+3],label=str(indexes[i+3]))
    for i in range(3): pl.plot(T,Fit_Moms[:,i+3],'--')
    pl.xlabel('Time')
    pl.ylabel('Order Two Moms')
    pl.legend()

    pl.subplot(2,2,3)
    for i in range(4): pl.plot(T,Moms[:,i+6],label=str(indexes[i+6]))
    pl.xlabel('Time')
    pl.ylabel('Order Three Moms')
    pl.legend()

    pl.subplot(2,2,4)
    #pdb.set_trace()
    alpha = (Moms[:,4]-Moms[:,1]*Moms[:,2])/Moms[:,3]
    alpha_Fit = (Fit_Moms[:,4]-Fit_Moms[:,1]*Fit_Moms[:,2])/Fit_Moms[:,3]
    pl.plot(T,alpha,label='alpha_data')
    pl.plot(T,alpha_Fit,'--',label='alpha_fit')
    pl.legend()
    pl.xlabel('Time')
    pl.ylabel(' alpha = COV/Var')

    pl.tight_layout(rect=[0, 0.03, 1, 0.95])

    if show:
        pl.show()
    else:
        pdf.savefig(ax)
    #pl.clf()

def plot_reaction_Firing(K, T, Moms, name, fig_num=1, show=True, pdf = None):

    Mom_inds = [0,0,1,2,4,4,2,1,2,1]

    names = ['*-> A', '*-> B', 'A -> *', 'B -> *', ' B -| A', 'A -| B', ' B ->> A', 'A ->> A', 'B->>B', 'A ->> B']

    react = []
    
    ax = pl.figure(figsize=(10,4))
    pl.suptitle(name)

    pl.subplot(1,2,1)

    for k in range(len(K)):    
        react.append(Moms[:,Mom_inds[k]]*K[k])
        if k in [4,5,6,9]:
            pl.plot(T, react[-1],label=names[k])
        
    pl.ylabel('Reaction Rate per second')
    pl.xlabel('Time')
    pl.legend()
    pl.subplot(1,2,2)
    #pl.plot(T, react[-3] -react[4] + react[6], label ='B regulates A')
    #pl.plot(T, react[-2] -react[5] + react[-1], label= 'A regulates B')

    pl.plot(T,  -react[4] + react[6]  , label     = 'FLUX | B -| A vs  B ->> A')
    pl.plot(T,  -react[5] + react[-1] , label    = 'FLUX | A -| B vs  B ->> A')
    pl.ylabel('Flux Rate per second')
    pl.xlabel('Time')

    pl.legend()

    pl.tight_layout(rect=[0, 0.03, 1, 0.95])

    if show:
        pl.show()
    else:
        pdf.savefig(ax)
    #pl.clf()

    return Give_Regulatory_Network(T[-1]-T[0], np.average(np.array(react),axis=1))

