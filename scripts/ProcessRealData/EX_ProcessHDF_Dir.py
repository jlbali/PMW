# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 13:47:57 2017

@author: rgrimson
"""
from __future__ import print_function

#logging
import L_log
import logging

#WDIR
import my_base_dir
BASE_DIR = my_base_dir.get()
MAPA='LCA_12500m'
wdir=BASE_DIR + 'Mapas/%s/'%MAPA


import sys

import L_ReadObs
import L_Context
import L_Kernels
import L_Output
import L_Solvers
import L_Files


#%%
def main():    
    global wdir
    #PROCESS PARAMETERS
    #Method = 'Rodgers_IT' 
    #Method = 'Weights'    
    #Method = 'Tichonov' 
    #Method = 'GCV_Tichonov' 
    Method = 'Rodgers_IT' 

    Context=L_Context.Load_Context(wdir)

    FileList=L_Files.listFilesWithExtension(wdir+'HDF/','.h5')
    nF=len(FileList)
    iF=0
    for HDFfilename in FileList:
        print('############################################################################') 
        print('## Processing file %s, method = %s' %(HDFfilename, Method))
        iF+=1
        print('## %d of %d' %(iF,nF))
        Bands=Context['Bands']
            
        for Band in Bands:
            Error=False
            try:
                Observation, tita=L_ReadObs.Load_Band_Kernels(HDFfilename,Context,Band)
            except:
                    Error=True
                    logging.info("Error - Could not load file")
            if not Error:
                try:
                    L_Output.Export_Observation(Observation, Context, Band, HDFfilename)
                except:
                    Error=True
                    logging.info("Error - Could not export observation")
            if not Error:
                try:
                    #Solve system
                    print('######################################')
                    print('##   SOLVING BAND',Band,'#################')
                    print('######################################')
                    Sol=L_Solvers.Solve(Context, Observation, Method, tita)
                    FileName=Method+'_'+HDFfilename+'_'
                except:
                    Error=True
                    logging.info("Error - Could not solve system")
            if not Error:
                try:
                    L_Output.Export_Solution(Sol, Context, Band, FileName)
                except:
                    Error=True
                    logging.info("Error - Could not export solution")
            if Error:
                print('## Could not process file band --- see log file')
                print('############################################################################\n') 
            else:            
                print('## Donde processing band')
                print('############################################################################\n') 
    

#%%
if __name__ == "__main__":
    main()


# Out_Grids/Joined es el shapefile final...
