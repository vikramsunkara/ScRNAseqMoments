import numpy as np 
import pickle
import pdb
import pylab as pl 
from matplotlib.backends.backend_pdf import PdfPages
import time

import sys, os
### Turn print on and off ###
# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

#### Design Matrix Generator

from Methods.two_Dim_mRNA import generate_design_blocks as GDB

### NLLS arguments ###
from Methods.NLLS import NLLS_Fit
### SINDY arguments ###
from Methods.Sindy import SINDY

# Rendering tools.
from Methods.NLLS_Push_Forward import PushForward_Func
from Methods.plot_util import plot_Fit_and_Data, plot_reaction_Firing

#########
## Moment and Design Blocks Configuration
#########

### General moments setups ###
hash_lookUp = {(0,0):0, (1,0):1, (0,1):2, # 1st moments
                (2,0):3, (1,1):4, (0,2):5, # 2nd moments
                (3,0):6, (2,1):7, (1,2):8, (0,3):9, # 3rd moments
                (4,0):10, (3,1):11, (2,2):12, (1,3):13, (0,4):14 # 4th moments
                }

moms_lookUp_list = [(0, 0), (1, 0), (0, 1), # 1st moments
                    (2, 0), (1, 1), (0, 2), # 2nd moments
                    (3, 0), (2, 1), (1, 2), (0, 3), # 3rd moments
                    (4,0), (3,1), (2,2), (1,3), (0,4) # 4th moments
                             ]



MB_LLS_Moments_Fit = np.array([True, True, True, 
                                True, True, True, 
                                True, True, True, True, 
                                False, False, False, False, False
                                ])


NLLS_Moments_Fit = np.array([True, True, True,
                             True, True, True,
                             False, False, False, False,
                             False, False, False, False, False
                             ])


Moments_Spline = np.array([  False, False, False,
                             False, False, False,
                             True, True, True, True,
                             False, False, False, False, False
                             ])

Spline_der_bool = True 


###
# Weights on First Order Moments
###

W_BM_LLS = np.eye(np.sum(MB_LLS_Moments_Fit))
W_BM_LLS[1,1] = 40.0
W_BM_LLS[2,2] = 40.0

W_NLLS = np.eye(np.sum(NLLS_Moments_Fit))
W_NLLS[1,1] = 40.0
W_NLLS[2,2] = 40.0

#######
# Load Design Blocks
#######

# Linear MBI
blockPrint()
inds = np.where(MB_LLS_Moments_Fit)[0]
moms_lookUp_list = [moms_lookUp_list[i] for i in inds]
Design_Blocks_MB_LLS = GDB(hash_lookUp, moms_lookUp_list, Verbose=False)

# Nonlinear MBI
inds = np.where(NLLS_Moments_Fit+Moments_Spline)[0]
moms_lookUp_list = [moms_lookUp_list[i] for i in inds]
Design_Blocks_NLLS = GDB(hash_lookUp, moms_lookUp_list, Verbose=False)
enablePrint()



def Batch_Inference(Data_list, Data_Names, Run_Name, shift = 30, sub_sample = 15,  PDF_Save_dir = 'PDF', GRN_Save_dir = 'GRNs', indexes = None, W_BM_LLS = W_BM_LLS, W_NLLS = W_NLLS):
        '''
        @params Data_list: list of moments data to infere GRN from            list of array      (#num_data) 
        @params Data_Names: corresponding names of data                       list of string     (#num_data)
        @params Run_Names: name of the batch run                              string
        @params shift: shift of moments data                                  integer
        @params sub_sample: sub sampling frequency                            integer
        @params PDF_Save_dir: path to folder to safe figures
        @params GRNs: path to folder to save infered GRN
        
        '''
        MB_LLS_pdf_reaction_firing = PdfPages('%s/%s_MB_LLS_Reactions.pdf'%(PDF_Save_dir,Run_Name))
        NLLS_pdf_push_forward = PdfPages('%s/%s_NLLS_push_fowrads.pdf'%(PDF_Save_dir,Run_Name))
        NLLS_pdf_reaction_firing = PdfPages('%s/%s_NLLS_Reactions.pdf'%(PDF_Save_dir,Run_Name))

        MB_LLS_Reg_Networks = []
        NLLS_Reg_Networks = []

        MB_LLS_Times = []
        NLLS_Times = []

        NLLS_params = []
        MB_LLS_params = []
        
        status_sindy =[]
        status_nlls = []

        for i in range(len(Data_list)):
            print(Data_Names[i], "%d out of %d"%(i, len(Data_list)))
            #### Extract Data and Splice the required data points
            Mom_list = Data_list[i]['centers'][shift:,:]
            TT = Data_list[i]['Times'][shift:]

            E = Mom_list[::sub_sample,:]
            T = TT[::sub_sample]


            ##############
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INFERENCE CODE
            ##############
            

            # Linear MBI 
            blockPrint()
            tic = time.time()
            x_opt, st_sindy = SINDY(E, T, Design_Blocks_MB_LLS, fit = MB_LLS_Moments_Fit, weights = W_BM_LLS)
            toc = time.time()
            MB_LLS_Times.append(toc-tic)
            MB_LLS_params.append(x_opt)
            status_sindy.append(st_sindy)
            
            # Nonlinear MBI

            tic = time.time()
            NLLS_est, st_nlls = NLLS_Fit(np.abs(x_opt), E, T, Design_Blocks_NLLS, NLLS_Moments_Fit, Moments_Spline, Spline_der_bool, weights = W_NLLS)
            toc = time.time()
            NLLS_Times.append(toc-tic)
            NLLS_params.append(NLLS_est)
            status_nlls.append(st_nlls)
            enablePrint()

            ##############
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% RENDERING CODE
            ##############


            ## RENDERING MB-LLS
            name_of_plot = "%s | MB-LLS %s:"%(Run_Name,Data_Names[i]) + "\n"+ "[%.4f, %.4f, %.4f, %.4f, %.4g, %.4g, %.4f, %.4f, %.4f, %.4f]"%tuple(x_opt)
            Reg_Net_MB_LLS = plot_reaction_Firing(np.abs(x_opt), T, E, name_of_plot, fig_num=i+1, show=False, pdf=MB_LLS_pdf_reaction_firing)
            MB_LLS_Reg_Networks.append(Reg_Net_MB_LLS)

            ## RENDERING NLLS
            name_of_plot = "%s | NLLS %s:"%(Run_Name,Data_Names[i]) + "\n"+ "[%.4f, %.4f, %.4f, %.4f, %.4g, %.4g, %.4f, %.4f, %.4f, %.4f]"%tuple(NLLS_est)
            res, pred = PushForward_Func(NLLS_est, E, T, Design_Blocks_NLLS, NLLS_Moments_Fit, Moments_Spline, Spline_der_bool)
            plot_Fit_and_Data(T, E, pred, indexes, name_of_plot, fig_num=i+1, show=False, pdf = NLLS_pdf_push_forward)
            Reg_Net_NLLS = plot_reaction_Firing(NLLS_est, T, E, name_of_plot, fig_num=i+1, show=False, pdf=NLLS_pdf_reaction_firing)
            NLLS_Reg_Networks.append(Reg_Net_NLLS)

            print(Reg_Net_NLLS)

            #plot_Fit_and_Data(T, E, pred, moms_lookUp_list, name = name_of_plot, fig_num=i+1, show=False, pdf=NLLS_pdf_push_forward)


        MB_LLS_pdf_reaction_firing.close()
        NLLS_pdf_push_forward.close()
        NLLS_pdf_reaction_firing.close()


        f = open('%s/%s_MB_Reactions.pck'%(GRN_Save_dir,Run_Name), 'wb')
        pickle.dump({"Species": Data_Names,'NLLS_param': NLLS_params, 'MB-LLS_params':MB_LLS_params, 
                        'NLLS_GRNs': NLLS_Reg_Networks , 'MB-LLS_GRNs': MB_LLS_Reg_Networks, 
                        'NLLS_Comp_Times': NLLS_Times, 'MB-LLS_Comp_Times': MB_LLS_Times, "Status_MB_LLS":status_sindy, "Status_NLLS":status_nlls}
                    ,f)
        f.close()

