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

# Disable print
import sys, os
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
# Restore print
def enablePrint():
    sys.stdout = sys.__stdout__

from Run_Multiple_mRNA_Inference import Batch_Inference


def PreFig(width = 21, height = 14):
    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20) 
    fig = pl.figure(figsize=(width,height))
    return fig


def plot_moms(Moms_data, indexes, title = None):
    '''
    @brief Plot moments and 95% confidence interval
    '''
    fig = PreFig(width = 22, height = 7)
    pl.suptitle(title)
    
    fig_zoom = PreFig(width = 22, height = 7)
    pl.suptitle(title)
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
    
    l = 0
    for j in (0, 3, 6):  #range(5): # plot the first 5 moments [(1,0),(0,1),(2,0),(1,1),(0,2),(3,0),(2,1),(1,2),(0,3),(4,0),(3,1),(2,2),(1,3),(0,4)]
        #ax = fig.add_subplot(2, 3, l+1)
        
        c_low = np.zeros(len(TT))
        c_up = np.zeros(len(TT))
        
        ymin = np.zeros(len(TT))
        ymax = np.zeros(len(TT))
        
        mvar = []
        for t in range(len(TT)):
        
            mbar = np.mean(Moms[t, j+1, :]) # the data includes order 0 moments (0, 0)
            #mvar.append(np.var(Moms[t, j+1, :]))
            sstd = np.std(Moms[t, j+1, :])
            
            # 95% CI
            #c_low[t] = mbar - t_stats*sstd/np.sqrt(df)
            #c_up[t] = mbar + t_stats*sstd/np.sqrt(df)
            
            # std
            c_low[t] = mbar - 2*sstd
            c_up[t] = mbar + 2*sstd
            
            ymin[t] = mbar - 4*sstd
            ymax[t] = mbar + 4*sstd
        
        #pl.plot(TT, xvar, linewidth = 2, color = "green", label = "Variance of Moments")
        ax = fig.add_subplot(1, 3, l+1)
        ax.plot(TT, moms[:, j+1], linewidth = 1., color = "green", label = "random run")
        #ax.fill_between(TT, c_low, c_up, color = "red", alpha = 0.75, label = "95% CI")
        ax.fill_between(TT, c_low, c_up, color = "red", alpha = 0.5, label = "+/- 2STD")
        ax.set_title("$\mathbb{E}$[mRNA A$(t)^%d$ mRNA B$(t)^%d$]"%indexes[j])
        
        
        ax2 = fig_zoom.add_subplot(1, 3, l+1)
        ax2.plot(TT, moms[:, j+1], linewidth = 2.5, color = "green", label = "random run")
        #ax.fill_between(TT, c_low, c_up, color = "red", alpha = 0.75, label = "95% CI")
        ax2.fill_between(TT, c_low, c_up, color = "red", alpha = 0.5, label = "+/- 2STD")
        ax2.set_xlim(40, 60)
        ax2.set_ylim(min(ymin[(TT>40) & (TT<60)]),max(ymax[(TT>40) & (TT<60)]))
        ax2.set_title("$\mathbb{E}$[mRNA A$(t)^%d$ mRNA B$(t)^%d$]"%indexes[j])
        
        l+=1
        
        if j == 0:
            ax.legend()
            ax2.legend()
    
    
    return fig, fig_zoom 
            

def Inference_compare_random(Moms_data, PDFsave, GRNsave, wt_LLS = 40.0, wt_NLLS = 40.0):
    # grab a random run
    data = Moms_data[np.random.choice(np.arange(0,len(Moms_data),1,dtype=np.int))]
    
    print("weight LLS = 40.0, weight NLLS = 40.0")
    LLS, NLLS = Batch_Inference([data], [DLab_m[i]+"(#%d)"%n for n in range(1)], DLab_m[i], shift = 30, sub_sample = 15, PDF_Save_dir = PDFsave, GRN_Save_dir = GRNsave, indexes = indexes)
    
    print("weight LLS = %.1f, weight NLLS = 40.0"%wt_LLS)
    wLSS, NLLS = Batch_Inference([data], ["wLLS_"+DLab_m[i]+"(#%d)"%n for n in range(1)], "wLLS_"+DLab_m[i], shift = 30, sub_sample = 15, PDF_Save_dir = PDFsave, GRN_Save_dir = GRNsave, indexes = indexes, wt_LLS = wt_LLS)
            
    print("weight LLS = 40.0, weight NLLS = %.1f"%wt_NLLS)
    LLS, NLLS = Batch_Inference([data], ["wNLLS_"+DLab_m[i]+"(#%d)"%n for n in range(1)], "wNLLS_"+DLab_m[i], shift = 30, sub_sample = 15, PDF_Save_dir = PDFsave, GRN_Save_dir = GRNsave, indexes = indexes, wt_NLLS = wt_NLLS)

 
def Inference_compare(Moms_data, PDFsave, GRNsave, wt_LLS = 40.0, wt_NLLS = 40.0):
    # grab a random run
    
    data = Moms_data[::4]
    
    """
    blockPrint()
    print("weight LLS = 40.0, weight NLLS = 40.0")
    LLS, NLLS = Batch_Inference(data, [DLab_m[i]+"(#%d)"%n for n in range(len(data))], DLab_m[i], shift = 30, sub_sample = 15, PDF_Save_dir = PDFsave, GRN_Save_dir = GRNsave, indexes = indexes)
    
    print("weight LLS = %.1f, weight NLLS = 40.0"%wt_LLS)
    wLLS, NLLS2 = Batch_Inference(data, ["wLLS_"+DLab_m[i]+"(#%d)"%n for n in range(len(data))], "wLLS_"+DLab_m[i], shift = 30, sub_sample = 15, PDF_Save_dir = PDFsave, GRN_Save_dir = GRNsave, indexes = indexes, wt_LLS = wt_LLS)
            
    print("weight LLS = 40.0, weight NLLS = %.1f"%wt_NLLS)
    LLS2, wNLLS = Batch_Inference(data, ["wNLLS_"+DLab_m[i]+"(#%d)"%n for n in range(len(data))], "wNLLS_"+DLab_m[i], shift = 30, sub_sample = 15, PDF_Save_dir = PDFsave, GRN_Save_dir = GRNsave, indexes = indexes, wt_NLLS = wt_NLLS)
    enablePrint()
    
    
    LLS = np.array(LLS)
    wLLS = np.array(wLLS)
    wNLLS = np.array(wNLLS)
    
    #save
    f = open(GRNsave+"/%s_Mimina.pck"%DLab_m[i], "wb")
    pickle.dump({"LLS":LLS, "wLLS":wLLS, "wNLLS":wNLLS, "w_val_LLS":wt_LLS, "w_val_NLLS":wt_NLLS}, f)
    f.close()
    """
    
    #open
    f = open(GRNsave+"/%s_Mimina.pck"%DLab_m[i], "rb")
    stuff = pickle.load(f)
    f.close()
    
    LLS = stuff["LLS"]
    wLLS = stuff["wLLS"]
    Error = np.mean(np.abs(LLS - wLLS), axis = 0)
    print(DLab_m[i] + " Minma Error: mean(abs(Theta_LLS(40) - Theta_LLS(%.f)))"%wt_LLS, "num_runs = %d"%len(data))    
    print(Error)        
       
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


for i in range(2, len(File_list)):
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
    fig1, fig2 = plot_moms(Moms_time_data_diff, indexes, "Randomised Cell Samples")
    fig1.savefig("/nfs/datanumerik/people/araharin/2mRNA_100000/Moments/"+DLab_m[i]+"2_diff.pdf", bbox_inches='tight')
    fig2.savefig("/nfs/datanumerik/people/araharin/2mRNA_100000/Moments/"+DLab_m[i]+"2_diff_zoom.pdf", bbox_inches='tight')

    
    #plot one random inference LLS or NLLS weight different than the defaults
    print("Remove temporal correlation: Random cell sample at every timepoint")
    #Inference_compare(Moms_time_data_diff, "/nfs/datanumerik/people/araharin/2mRNA_100000/Moments/PDF_diff","/nfs/datanumerik/people/araharin/2mRNA_100000/Moments/GRNs_diff", wt_LLS = 100.0, wt_NLLS = 100.0)
    
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
    fig3, fig4 = plot_moms(Moms_time_data, indexes, "Randomised Trajectories")
    fig3.savefig("/nfs/datanumerik/people/araharin/2mRNA_100000/Moments/"+DLab_m[i]+"2.pdf", bbox_inches='tight')
    fig4.savefig("/nfs/datanumerik/people/araharin/2mRNA_100000/Moments/"+DLab_m[i]+"2_zoom.pdf", bbox_inches='tight')

    
    #plot one random inference LLS or NLLS weight different than the defaults
    print("---------------------------------------------------------")
    print("Ignore temporal correlation: Random Sample trajectories")
    #Inference_compare(Moms_time_data,"/nfs/datanumerik/people/araharin/2mRNA_100000/Moments/PDF","/nfs/datanumerik/people/araharin/2mRNA_100000/Moments/GRNs", wt_LLS = 100.0, wt_NLLS = 100.0)
    print("---------------------------------------------------------")

        
