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

#%%
#PROCESS PARAMETERS
HDFfilename='GW1AM2_201207310439_227D_L1SGBTBR_2210210'
Bands=['ka','ku','x','c']
Methods=["Weights","LSQR","Rodgers_IT","Global_GCV_Tichonov"]

#%%

def main():
    for MAPA in ['Ajo_25km','Ajo_10km']:
        wdir=BASE_DIR + 'Mapas/%s/'%MAPA
        Context=L_Context.Load_Context(wdir)
    
            
        print('############################################################################') 
        print('## Processing file %s' %HDFfilename)
            
        for Band in Bands:
            Observation, tita=L_ReadObs.Load_PKL_or_Compute_Kernels(HDFfilename,Context,Band,GC=True)
            for Method in Methods:
                print('######################################')
                print('##   SOLVING BAND',Band,'#################')
                print('######################################')
                Sol=L_Solvers.Solve(Context, Observation, Method, tita)
                
                #Write images with solution
                L_Output.Export_Solution(Sol, Context, Band, Observation['FileName']+'_'+Method)
        #%%
