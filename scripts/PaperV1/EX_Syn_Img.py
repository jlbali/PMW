# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 16:26:12 2017

@author: rgrimson
"""

from __future__ import print_function

#WDIR
import my_base_dir
BASE_DIR = my_base_dir.get()
MAPA='Ajo_25km'
wdir=BASE_DIR + 'Mapas/%s/'%MAPA

#Import libraries
import L_ReadObs
import L_Context
import L_Output
import L_Solvers
import L_Syn_Img
import L_ParamEst
from scipy.sparse.linalg import lsqr

#%%
#PROCESS PARAMETERS
HDFfilename='GW1AM2_201208110420_224D_L1SGBTBR_2210210'
#HDFfilename='GW1AM2_201406011716_108A_L1SGBTBR_2220220'
#HDFfilename='GW1AM2_201406020358_220D_L1SGBTBR_2220220'
HDFfilename='GW1AM2_201207310439_227D_L1SGBTBR_2210210'

Band='X'

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

    #Create Synthetic Image 
    
    #Export Synthetic Image Map
    #L_Output.Export_Solution([TbH,TbV], Context, Band, "SynthImg")

    #Load ellipses and compute synthetic observation
    Observation, tita=L_ReadObs.Load_Band_Kernels(HDFfilename, Context, Band) #Load Observations & Kernels
    RMSE_H=[]
    RMSE_V=[]

    for i in range(100):
        TbH=L_Syn_Img.Create_Trig_Synth_Img(Context)
        TbV=L_Syn_Img.Create_Random_Synth_Img(Context,sgm=5)
            
        Observation=L_Syn_Img.Simulate_PMW(TbH,TbV,Context,Observation,Obs_error_std=damp) #Simulate SynObs
        #print("LandType approximation error:%.2f, %.2f"%(Observation['LSQRSols']['H']['norm'],Observation['LSQRSols']['V']['norm']))
        #Observation['LSQRSols']={'H':lsqr(M,Observation['Tb']['H']),'V':lsqr(M,Observation['Tb']['V'])}
         
    
        
        #Compute cell in the grid considered as MARGIN-CELLS (for correct error estimation)
        L_Syn_Img.Set_Margins(Context)
    
        #Print Synth Image stats
        tita=L_ParamEst.recompute_param([TbH,TbV],Observation,tita)
        L_ParamEst.pr(tita,'Synthetic Image')
        
        #%###################################################
        ## SOLVE SYSTEM with real parameters
        Sol=L_Solvers.Solve(Context,Observation,Method, tita,damp=damp)
    
        #%##################################################
        ## Export solution
        #L_Output.Export_Solution(Sol, Context, Band, "SolSynthImg_%s"%Method)
        #%    
        #Compare Original and Reconstructed: Evaluate error (with and without margins)
        tita=L_ParamEst.recompute_param([TbH,TbV],Observation,tita)
        L_ParamEst.pr(tita,'Synthetic Image')
        
        tita=L_ParamEst.recompute_param(Sol,Observation,tita)
        L_ParamEst.pr(tita,'Reconstructed Image')
    
        print("\nSummary:\n--------")
        #print("Complete Image")
        #print("  RMSE H: %.3f"%(L_Syn_Img.RMSE(TbH,Sol[0],Context)))
        #print("  RMSE V: %.3f"%(L_Syn_Img.RMSE(TbV,Sol[1],Context)))
        print("Discarding Margin")
        print("  RMSE H: %.3f"%(L_Syn_Img.RMSE_M(TbH,Sol[0],Context)))
        RMSE_H.append(L_Syn_Img.RMSE_M(TbH,Sol[0],Context))
        print("  RMSE V: %.3f"%(L_Syn_Img.RMSE_M(TbV,Sol[1],Context)))
        RMSE_V.append((L_Syn_Img.RMSE_M(TbV,Sol[1],Context)))
        print("For each landtype, discarding Margin")
        for lt in range(2):
            print("       LT: %d"%(lt))
            print("   RMSE H: %.3f"%(L_Syn_Img.RMSE_M_LT(TbH,Sol[0],Context, lt)))
            print("   RMSE V: %.3f"%(L_Syn_Img.RMSE_M_LT(TbV,Sol[1],Context, lt)))            

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
