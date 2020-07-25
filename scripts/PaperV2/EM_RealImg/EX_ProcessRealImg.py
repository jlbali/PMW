# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 13:47:57 2017

@author: rgrimson
"""
from __future__ import print_function


#WDIR
import my_base_dir
BASE_DIR = my_base_dir.get()
MAPA='LCA_25000m'
wdir=BASE_DIR + 'Mapas/%s/'%MAPA


import sys

import L_ReadObs
import L_Context
import L_Kernels as L_Kernels
import L_Output
import L_Solvers


#%%
def main():    
    global wdir
    #PROCESS PARAMETERS
    HDFfilename='GW1AM2_201406020358_220D_L1SGBTBR_2220220'
    Context=L_Context.Load_Context(wdir)

        
    print('############################################################################') 
    print('## Processing file %s' %HDFfilename)
        
    Bands=['KA']#Context['Bands']
    Methods=['Global_GCV_Tichonov','Rodgers_IT','LSQR','Weights']
    Methods=['EM']
    for Band in Bands:
        Observation, tita=L_ReadObs.Load_Band_Kernels(HDFfilename, Context, Band)
            
            
        #%%Solve system
        for Method in Methods:
            print('######################################')
            print('##   SOLVING BAND',Band,'#################')
            print('######################################')
            Sol=L_Solvers.Solve(Context, Observation, Method, tita)
            
            #Write images with solution
            L_Output.Export_Solution(Sol, Context, Band, Observation['FileName']+'_'+Method)
        #%%
    print('## Donde processing file')
    print('############################################################################\n') 
    #plt.close()
    

#%%
if __name__ == "__main__":
    main()


# Out_Grids/Joined es el shapefile final...

