# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 16:26:12 2017

@author: rgrimson
"""

from __future__ import print_function

#WDIR
import my_base_dir
BASE_DIR = my_base_dir.get()
#Import libraries
import L_ReadObs
import L_Context
import L_Output
import L_Solvers
import L_Syn_Img
import L_ParamEst

#PROCESS PARAMETERS
HDFfilename='GW1AM2_201207310439_227D_L1SGBTBR_2210210'
Bands=['ka','ku','c']
Methods=["Weights","LSQR","Rodgers_IT","Global_GCV_Tichonov"]
MAPAS=['Ajo_25km','Ajo_10km']


#%%
def main(): 
    #%%###################
    damp=1.0 #obs std error
    for MAPA in MAPAS:
        wdir=BASE_DIR + 'Mapas/%s/'%MAPA
        Context=L_Context.Load_Context(wdir)

        #Create Synthetic Image 
        TbH=L_Syn_Img.Create_Trig_Synth_Img(Context)
        TbV=L_Syn_Img.Create_Random_Synth_Img(Context)
    
        #Export Synthetic Image Map
        for Band in Bands:
            #Load ellipses and compute synthetic observation
            Observation, tita=L_ReadObs.Load_PKL_or_Compute_Kernels(HDFfilename, Context, Band, GC=True) #Load Observations & Kernels
            Observation=L_Syn_Img.Simulate_PMW(TbH,TbV,Context,Observation,Obs_error_std=damp) #Simulate SynObs
            L_Output.Export_Solution([TbH,TbV], Context, Band, "SynthImg")

            #Compute cell in the grid considered as MARGIN-CELLS (for correct error estimation)
            L_Syn_Img.Set_Margins(Context)
    
            #Print Synth Image stats
            tita=L_ParamEst.recompute_param([TbH,TbV],Observation,tita)
            L_ParamEst.pr(tita,'Synthetic Image')


            for Method in Methods:
                #%###################################################
                ## SOLVE SYSTEM with real parameters
                Sol=L_Solvers.Solve(Context,Observation,Method, tita,damp=damp)
            
                #%##################################################
                ## Export solution
                L_Output.Export_Solution(Sol, Context, Band, "SolSynthImg_%s"%Method)
                #%    
                #Compare Original and Reconstructed: Evaluate error (with and without margins)
                tita=L_ParamEst.recompute_param([TbH,TbV],Observation,tita)
                L_ParamEst.pr(tita,'Synthetic Image')
                
                tita=L_ParamEst.recompute_param(Sol,Observation,tita)
                L_ParamEst.pr(tita,'Reconstructed Image')
            
                print("\nSummary:\n--------")
                print("RMSE H: %.3f"%(L_Syn_Img.RMSE_M(TbH,Sol[0],Context)))
                print("RMSE V: %.3f"%(L_Syn_Img.RMSE_M(TbV,Sol[1],Context)))
