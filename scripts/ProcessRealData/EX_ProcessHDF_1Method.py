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
MAPA='LCA_50000m'
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
    HDFfilename='AMSR_E_L2A_BrightnessTemperatures_V12_200601010409_D'
    HDFfilename='GW1AM2_201408210358_220D_L1SGBTBR_2220220'
    HDFfilename='GW1AM2_201406020358_220D_L1SGBTBR_2220220'
    HDFfilename='GW1AM2_201207310439_227D_L1SGBTBR_2210210'
    arg_HDF=False #True #process argument file (or fixed example file)?
    if arg_HDF: #infut HDF is given by argument?
        HDFfilename=sys.argv[1]
    
    #Method = 'GCV_Tichonov' 
    #Method = 'BGC'    
    #Method = 'Tichonov' 
    #Method = 'SimpleEM'
    #Method = "OHB"
    #Method = "MCEM"
    Method = 'Weights'    
    Method = 'LSQR'
    Method = 'Global_GCV_Tichonov'
    #Method = 'K_Global_GCV_Tichonov'
    Method = 'Rodgers_IT' 
    
    Context=L_Context.Load_Context(wdir)

        
    print('############################################################################') 
    print('## Processing file %s' %HDFfilename)
        
    Bands=['KA','K','KU','C','X']#Context['Bands']
    Methods=['Global_GCV_Tichonov','Rodgers_IT','LSQR','Weights']
    Bands=['KA']#Context['Bands']
    Methods=['Rodgers_IT']
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

