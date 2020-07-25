# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 16:26:12 2017

@author: rgrimson
"""

from __future__ import print_function

#WDIR
import my_base_dir
BASE_DIR = my_base_dir.get()
MAPA='Ajo_10km'
wdir=BASE_DIR + 'Mapas/%s/'%MAPA

#Import libraries
import L_ReadObs
import L_Context
import L_Output
import L_Solvers
import L_Syn_Img
import L_ParamEst
from scipy.sparse.linalg import lsqr
import numpy as np
import logging


#%%
#PROCESS PARAMETERS
HDFfilename='GW1AM2_201208110420_224D_L1SGBTBR_2210210'
Band='ka'

Method="Rodgers_IT"
#Method="Weights"
#Method="LSQR"
#Method="Global_GCV_Tichonov"

#%%

def main(): 
    #%%###################
    ## PREPROCESSING    
    #%%
    #Load Context  
    damp=1.0
    Context=L_Context.Load_Context(wdir)

    #Load ellipses and compute synthetic observation
    Observation, tita=L_ReadObs.Load_PKL_Obs(HDFfilename, Context, Band) #Load Observations & Kernels
    #Compute cell in the grid considered as MARGIN-CELLS (for correct error estimation)
    L_Syn_Img.Set_Margins(Context)


    for i in range(10):
        #Create Synthetic Image 
        TbH=L_Syn_Img.Create_Random_Synth_Img(Context,sgm=10)
        TbV=L_Syn_Img.Create_Random_Synth_Img(Context,sgm=100)
        Observation=L_Syn_Img.Simulate_PMW(TbH,TbV,Context,Observation,Obs_error_std=damp) #Simulate SynObs
        #Print Synth Image stats
        tita=L_ParamEst.recompute_param([TbH,TbV],Observation,tita)
        L_ParamEst.pr(tita,'Synthetic Image')
        
        #%###################################################
        ## SOLVE SYSTEM with real parameters
        Sol=L_Solvers.Solve(Context,Observation,Method, tita,damp=damp)
        #%##################################################
        #%    
        #Compare Original and Reconstructed
        tita=L_ParamEst.recompute_param([TbH,TbV],Observation,tita)
        L_ParamEst.pr(tita,'Synthetic Image')
        sH0_o=np.sqrt(tita['H']['sigma2'][0])
        sH1_o=np.sqrt(tita['H']['sigma2'][1])
        sV0_o=np.sqrt(tita['V']['sigma2'][0])
        sV1_o=np.sqrt(tita['V']['sigma2'][1])
        
        tita=L_ParamEst.recompute_param(Sol,Observation,tita)
        L_ParamEst.pr(tita,'Reconstructed Image')
        sH0_c=np.sqrt(tita['H']['sigma2'][0])
        sH1_c=np.sqrt(tita['H']['sigma2'][1])
        sV0_c=np.sqrt(tita['V']['sigma2'][0])
        sV1_c=np.sqrt(tita['V']['sigma2'][1])
    
        print("\nSummary:\n--------")
        print("RMSE H: %.3f"%(L_Syn_Img.RMSE_M(TbH,Sol[0],Context)))
        print("RMSE V: %.3f"%(L_Syn_Img.RMSE_M(TbV,Sol[1],Context)))
        MSG="Orig: %.2f,  \t%.2f,  \t%.2f,  \t%.2f"%(sH0_o, sH1_o,sV0_o, sV1_o)
        print(MSG)
        logging.info(MSG)
        MSG="Comp: %.2f,  \t%.2f,  \t%.2f,  \t%.2f"%(sH0_c, sH1_c,sV0_c, sV1_c)
        print(MSG)
        logging.info(MSG)
        MSG="Quot: %.2f,  \t%.2f,  \t%.2f,  \t%.2f"%(sH0_c/sH0_o, sH1_c/sH1_o,sV0_c/sV0_o, sV1_c/sV1_o)
        print(MSG)
        logging.info(MSG)
    
    #%%


main()

##%%
###····test de bias····###
#Context=L_Context.Load_Context(wdir)
#
##Create Synthetic Image 
#TbH=L_Syn_Img.Create_Random_Synth_Img(Context,sgm=10)
#TbV=L_Syn_Img.Create_Random_Synth_Img(Context,sgm=5)
#
##Load ellipses and compute synthetic observation
#Observation, tita=L_ReadObs.Load_PKL_Obs(HDFfilename, Context, Band) #Load Observations & Kernels
#Observation=L_Syn_Img.Simulate_PMW(TbH,TbV,Context,Observation,Obs_error_std=1.0) #Simulate SynObs
#
##Compute cell in the grid considered as MARGIN-CELLS (for correct error estimation)
#L_Syn_Img.Set_Margins(Context)
#
#tita=L_ParamEst.recompute_param([TbH,TbV],Observation,tita)
#Sol=L_Solvers.Solve(Context,Observation,"Rodgers_IT", tita)
#print("RMSE H: %.3f V: %.3f "%(L_Syn_Img.RMSE_M(TbH,Sol[0],Context),L_Syn_Img.RMSE_M(TbV,Sol[1],Context)))
#
#import copy
#tita_IT=copy.deepcopy(tita)
#
##%%
#import numpy as np
#F= np.arange(1.0,1.5,.05)
#for f in F:
#    tita=copy.deepcopy(tita_IT)
#    for pol in ['H','V']:
#        for lt in [0,1]:
#            tita[pol]['sigma2'][lt]*=f
#    Sol=L_Solvers.Solve(Context,Observation,'Rodgers', tita)
#    print("Factor: %.3f RMSE H: %.3f V: %.3f "%(f,L_Syn_Img.RMSE_M(TbH,Sol[0],Context),L_Syn_Img.RMSE_M(TbV,Sol[1],Context)))
#
#            
#
#            
#
#
#
