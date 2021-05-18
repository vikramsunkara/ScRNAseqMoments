#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 13:54:26 2021

@author: araharin
"""

import numpy as np 
import pickle
import pdb
import pylab as pl 
import matplotlib
from scipy import stats


def PreFig():
    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20) 
    fig = pl.figure(figsize=(21,14))
    return fig


def plot_moms(Moms_data, indexes, title = None):
    '''
    @brief Plot moments and 95% confidence interval
    '''
    fig = PreFig()
    
    # grab a random run
    data = Moms_data[np.random.choice(np.arange(0,len(Moms_data),1,dtype=np.int))]
    TT = data['Times']
    moms = data['centers']
    
    Moms = np.zeros((len(TT), moms.shape[1], len(Moms_data)))
    for i in range(len(Moms_data)):
        Moms[:, :, i] = Moms_data[i]['centers']
    
    # draw 95% confidence interval
    alpha = (1-0.95)/2
    df = len(Moms_data) - 1
    t_stats = stats.t.ppf(1-alpha, df)
    
    for j in range(5): # plot the first 5 moments
        ax = fig.add_subplot(2, 3, j+1)
        c_low = np.zeros(len(TT))
        c_up = np.zeros(len(TT))
    
        for t in range(len(TT)):
        
            xbar = np.mean(Moms[t, j+1, :]) # the data includes order 0 moments (0, 0)
            sstd = np.std(Moms[t, j+1, :])
            
            # 95% CI
            #c_low[t] = xbar - t_stats*sstd/np.sqrt(df)
            #c_up[t] = xbar + t_stats*sstd/np.sqrt(df)
            
            # std
            c_low[t] = xbar - sstd
            c_up[t] = xbar + sstd
        
        pl.plot(TT, moms[:, j+1], linewidth = 0.5, color = "green", label = "random run")
        #ax.fill_between(TT, c_low, c_up, color = "red", alpha = 0.75, label = "95% CI")
        ax.fill_between(TT, c_low, c_up, color = "red", alpha = 0.75, label = "STD")
        pl.title("$\mathbb{E}$[mRNA A$(t_d)^%d$ mRNA B$(t_d)^%d$]"%indexes[j])
        if j == 0:
            pl.legend()
    
    pl.suptitle(title)
    return fig   
            

from Compute_Moms import *

"""
Multiple inference for the three models (Double Up, No Up, Single Up) with all simulations 
having initial condition (G1, G2, P1, P2, M1, M2) =(0, 0, 0, 0, 0, 0) (we only used M1 and M2)
Input files, please put your own data file paths
"""

File1 = "/nfs/datanumerik/people/araharin/2mRNA_100000/two_MRNA_Double_Up_data_100000.pck" 
File2 = "/nfs/datanumerik/people/araharin/2mRNA_100000/two_MRNA_No_Up_data_100000.pck" 
File3 = "/nfs/datanumerik/people/araharin/2mRNA_100000/two_MRNA_Single_Up_data_100000.pck" 

"""
Multiple inference for the model Single Up with all simulations 
having different conditions of the form (G1, G2, P1, P2, M1, M2) =(0, 0, 0, 0, M1, M2) (we only change M1 and M2)
Input files, please put your own data file paths
"""

File3a = "/nfs/datanumerik/people/araharin/2mRNA_100000/two_MRNA_Single_Up_data_70_0_100000.pck" 
File3b = "/nfs/datanumerik/people/araharin/2mRNA_100000/two_MRNA_Single_Up_data_0_70_100000.pck" 
File3c = "/nfs/datanumerik/people/araharin/2mRNA_100000/two_MRNA_Single_Up_data_70_70_100000.pck" 


DLab_m =["Double_up", "No_up", "Single_up", "Single_up_70_0 ", "Single_up_0_70", "Single_up_70_70"]


File_list = [File1, File2, File3]#, File3a, File3b, File3c]

indexes = [(1,0),(0,1),(2,0),(1,1),(0,2),(3,0),(2,1),(1,2),(0,3),(4,0),(3,1),(2,2),(1,3),(0,4)]
Moments = [np.array(item) for item in indexes]
Species_To_Store = np.array([False,False,False,False,True,True])

BatchNum = 400 # Number of replicate datasets


import joblib as jb
from functools import partial 
ntasks = 40


for i in range(len(File_list)):
    input_file = File_list[i]
    
    # compute moments for BatchNum replicate datasets (without parallel computing)
    
    """
    Moms_time_data = []
    Moms_time_data_diff = []
    for n in range(BatchNum):
        
        ''' 
        @brief remove temporal correlation
           
        ''' 
        data = Load_moms_time_diff(input_file, Moments, keep_species = Species_To_Store)
        Moms_time_data_diff.append(data)
        
        
        '''
        @brief ignore temporal correlation
        '''
        data = Load_moms_time(input_file, Moments, keep_species = Species_To_Store)

        Moms_time_data.append(data)
    
    """
    # compute moments for BatchNum replicate datasets (with parallel computing)
    
    
    
    ''' 
    @brief remove temporal correlation
    '''
    
    """
    Load_moms_time_p = partial(Load_moms_time_diff, input_filename = input_file, Moments = Moments, keep_species = Species_To_Store)
    
    Moms_time_data_diff = jb.Parallel(n_jobs = ntasks)(jb.delayed(Load_moms_time_p)() for n in range(BatchNum))
    
    ### save runs ####
    f = open("/nfs/datanumerik/people/araharin/2mRNA_100000/Moments/"+DLab_m[i]+"_diff_%d"%BatchNum+".pck", "wb")
    pickle.dump({"moms_runs":Moms_time_data_diff}, f)
    f.close()
    """
    ### load runs ###
    g = open("/nfs/datanumerik/people/araharin/2mRNA_100000/Moments/"+DLab_m[i]+"_diff_%d"%BatchNum+".pck", "rb")
    stuff = pickle.load(g)
    g.close()
    
    Moms_time_data_diff = stuff["moms_runs"]
    
    #plot one run and 95% CI or STD
    fig1 = plot_moms(Moms_time_data_diff, indexes, "Remove temporal correlation")
    fig1.savefig("/nfs/datanumerik/people/araharin/2mRNA_100000/Moments/"+DLab_m[i]+"2_diff.pdf", bbox_inches='tight')
    
    
    '''
    @brief ignore temporal correlation
    '''
    
    """
    Load_moms_time_p = partial(Load_moms_time, input_filename = input_file, Moments = Moments, keep_species = Species_To_Store)
    
    
    Moms_time_data = jb.Parallel(n_jobs = ntasks)(jb.delayed(Load_moms_time_p)() for n in range(BatchNum))
    
    
    ### save runs ####
    f = open("/nfs/datanumerik/people/araharin/2mRNA_100000/Moments/"+DLab_m[i]+"_%d"%BatchNum+".pck", "wb")
    pickle.dump({"moms_runs":Moms_time_data}, f)
    f.close()
    """
    
    ### load runs ###
    g = open("/nfs/datanumerik/people/araharin/2mRNA_100000/Moments/"+DLab_m[i]+"_%d"%BatchNum+".pck", "rb")
    stuff = pickle.load(g)
    g.close()
    
    Moms_time_data = stuff["moms_runs"]
    
    #plot one run and 95% CI or STD
    fig2 = plot_moms(Moms_time_data, indexes, "Ignore temporal correlation")
    fig2.savefig("/nfs/datanumerik/people/araharin/2mRNA_100000/Moments/"+DLab_m[i]+"2.pdf", bbox_inches='tight')
    



        
