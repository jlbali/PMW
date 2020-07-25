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
import numpy as np
import L_Syn_Img
import L_ParamEst

#%%
#PROCESS PARAMETERS
HDFfilename='GW1AM2_201208110420_224D_L1SGBTBR_2210210'
Band='ka'
#%%

def main(): 
    #%%###################
    ## PREPROCESSING    
    
    #Load Context    
    Context=L_Context.Load_Context(wdir)

    #Load Observation
    Observation, tita=L_ReadObs.Load_PKL_Obs(HDFfilename, Context, Band) #Lee obs y nucleos

    #Create Synthetic Image 
    TbH=L_Syn_Img.Create_Trig_Synth_Img(Context)
    TbV=L_Syn_Img.Create_Random_Synth_Img(Context)
    
    #Export Synthetic Image Map
    L_Output.Export_Solution([TbH,TbV], Context, Band, "SynthImg")

    #Print Synth Image stats
    tita=L_ParamEst.recompute_param([TbH,TbV],Observation,tita)
    L_ParamEst.pr(tita,'Synthetic Image')
    
    
    Observation=L_Syn_Img.Simulate_PMW(TbH,TbV,Context,Observation,error_var=1.0)
    
    #Compute cell in the grid considered as MARGIN-CELLS
    L_Syn_Img.Set_Margins(Context)

    #%%##################################################
    ## SOLVE SYSTEM with real parameters
        

    #%%##################################################
    ## Export solution
    L_Output.Export_Solution(SolR, Context, Band, "SolSynthImg_%s"%Method)
    
    #Evaluate error (with and without margins)
    print("RMSE H: %.3f,%.3f"%(L_Syn_Img.RMSE_M(TbH,SolR[0],Context), L_Syn_Img.RMSE(TbH,SolR[0])))
    print("RMSE V: %.3f,%.3f"%(L_Syn_Img.RMSE_M(TbV,SolR[1],Context), L_Syn_Img.RMSE(TbV,SolR[1])))

    #%%##################################################
    ## EROR ANALYSIS for Rodgers Method
    
    #Error covariance
    SH=L_Solvers.Rodgers_Compute_Covarince(Context, Observation,tita,'H')
    EH=np.sqrt(SH.diagonal())
    SV=L_Solvers.Rodgers_Compute_Covarince(Context, Observation,tita,'V')
    EV=np.sqrt(SV.diagonal())
    L_Output.Export_Solution([EH,EV], Context, Band, "SolSynthImg_Std")
    
    #Export 5 first error eigenvectors
    EigH=np.linalg.eig(SH)
    EigV=np.linalg.eig(SV)
    for i in range(5):#(EigH[0].shape[0]):
        ViH=EigH[1][:,i]*EigH[0][i]
        ViV=EigV[1][:,i]*EigV[0][i]
        L_Output.Export_Solution([ViH,ViV], Context, Band, "SolSynthImg_Pattern_"+str(i))
    
    #%%#################################################################
    ## SOLVE SYSTEM WITH ITERATIVE METHOD

    #Print Synth Image stats
    tita=L_ParamEst.recompute_param([TbH,TbV],Observation,tita)
    L_ParamEst.pr(tita)

    #Solve system        
    SolR=L_Solvers.Solve(Context,Observation,"Rodgers_IT", tita)

    #Evaluate error  (with and without margins)
    print("RMSE H: %.3f,%.3f"%(L_Syn_Img.RMSE_M(TbH,SolR[0],Context), L_Syn_Img.RMSE(TbH,SolR[0])))
    print("RMSE V: %.3f,%.3f"%(L_Syn_Img.RMSE_M(TbV,SolR[1],Context), L_Syn_Img.RMSE(TbV,SolR[1])))
 
    #Export solution
    L_Output.Export_Solution(SolR, Context, Band, "Sol20120811_SynthImg_Rod_IT_1noise")

   
    #%%    

#main()