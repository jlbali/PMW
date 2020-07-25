# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 13:47:57 2017

@author: rgrimson
"""
from __future__ import print_function


#WDIR



import sys
import my_base_dir
BASE_DIR = my_base_dir.get()
MAPA='Ajo_25km'
wdir=BASE_DIR + 'Mapas/%s/'%MAPA
import L_ReadObs
import L_Context
import L_Kernels as L_Kernels
import L_Output
import L_Solvers


#%%
def main():    
    global wdir
    #PROCESS PARAMETERS
    #HDFfilename='AMSR_E_L2A_BrightnessTemperatures_V12_200601010409_D'
    #HDFfilename='GW1AM2_201207310439_227D_L1SGBTBR_2210210'
    HDFfilename='GW1AM2_201408210358_220D_L1SGBTBR_2220220'
    HDFfilename='GW1AM2_201207310439_227D_L1SGBTBR_2210210_SH'
    #HDFfilename='GW1AM2_201406011716_108A_L1SGBTBR_2220220'
    #HDFfilename='GW1AM2_201207161744_113A_L1SGRTBR_2220220'
    arg_HDF=False #True #process argument file (or fixed example file)?
    if arg_HDF: #infut HDF is given by argument?
        HDFfilename=sys.argv[1]
    
    Methods=['Weights', 'Rodgers_IT']#, 'LSQR']
    #Methods=['Rodgers_IT']
    #Method = 'K_Global_GCV_Tichonov'
    
    Context=L_Context.Load_Context(wdir)

        
    print('############################################################################') 
    print('## Processing file %s' %HDFfilename)
        
    Bands=['KA','K','KU','X','C']#Context['Bands']
    Bands=['KA']
    for Band in Bands:
        Observation, tita=L_ReadObs.Load_Shifted_Band_Kernels(HDFfilename,Context,Band,-5000,2000)
        L_Output.Export_Observation(Observation, Context, Band, HDFfilename)

            
        #%%Solve system
        for Method in Methods:
            print('######################################')
            print('##   SOLVING BAND',Band,'#################')
            print('##   Method: ',Method,'#################')
            print('######################################')
            Sol=L_Solvers.Solve(Context, Observation, Method, tita)

            L_Output.Export_Solution(Sol, Context, Band, Observation['FileName']+'_'+Method)
        #%%
    print('## Donde processing file')
    print('############################################################################\n') 
    #plt.close()
    

#%%
if __name__ == "__main__":
    main()


# Out_Grids/Joined es el shapefile final...

