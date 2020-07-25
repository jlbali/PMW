# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 13:47:57 2017

@author: rgrimson
"""
from __future__ import print_function


#WDIR
import my_base_dir
BASE_DIR = my_base_dir.get()
MAPA='Ajo_10km'
wdir=BASE_DIR + 'Mapas/%s/'%MAPA


import sys

import L_ReadObs
import L_Context
import L_Kernels as L_Kernels
import L_Output
import L_Solvers
import L_ParamEst


#%%
def main():    
    global wdir
    #PROCESS PARAMETERS
    HDFfilename='GW1AM2_201208110420_224D_L1SGBTBR_2210210'
    arg_HDF=False #True #process argument file (or fixed example file)?
    if arg_HDF: #infut HDF is given by argument?
        HDFfilename=sys.argv[1]
    
    Context=L_Context.Load_Context(wdir)

        
    print('############################################################################') 
    print('## Processing file %s' %HDFfilename)
        
    Band='ka'
    #Dict_Bands=L_ReadObs.Load_PKL_or_HDF_Ells(HDFfilename, Context,GC=True)
    print('######################################')
    print('##   PRECOMPUTING K for BAND',Band,'#####')
    print('######################################')
    Observation, tita = L_ReadObs.Load_PKL_or_Compute_Kernels(HDFfilename,Context,Band,GC=True)
       
        
    #Solve system
    Method = 'Rodgers' 
    print('######################################')
    print('##   SOLVING BAND',Band,'  Method',Method)
    print('######################################')
    SolR=L_Solvers.Solve(Context, Observation, Method, tita)
    
    tita=L_ParamEst.recompute_param(SolR,Observation,tita)
    tita=L_ParamEst.minSigma2(tita,.99)
    L_ParamEst.pr1(tita,"Post-Rodgers")
    
    Method = 'BGP' 
    print('######################################')
    print('##   SOLVING BAND',Band,'  Method',Method)
    print('######################################')
    Sol=L_Solvers.Solve(Context, Observation, Method, tita)

    #Write images with solution
    L_Output.Export_Solution(Sol, Context, Band, Observation['FileName']+'_'+Method)
    
    print('## Donde processing file')
    print('############################################################################\n') 
    #plt.close()
    

#%%
if __name__ == "__main__":
    main()
    #print("1")
