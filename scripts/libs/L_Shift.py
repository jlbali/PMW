# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 16:53:15 2018

@author: rgrimson
"""
from __future__ import print_function
#uncomment to read AMSRE files
#from pyhdf.SD import SD, SDC
import libs.L_Files as L_Files
import libsL_Kernels as L_Kernels
import logging
import time
import libs.L_ReadObs as L_ReadObs
import copy


SAVE_ELLS=False #file with observations for the study region, in internal format
SAVE_OBS=True  #file with precomputed Kernels for each band

def Shift_Band_Kernels(ObservationOO,Context,Shift_X, Shift_Y):
    Band=ObservationOO['Band']
    MSG="*******************************************************************************\n"
    
    Dict_Bands=copy.deepcopy(ObservationOO['Dict_Bands'])
    Dict_Bands[Band]['GeoCorrection']['X']+=Shift_X
    Dict_Bands[Band]['GeoCorrection']['Y']+=Shift_Y
    
    print('######################################')
    print('##   PRECOMPUTING K for BAND',Band,'#####')
    print('######################################')
    MSG+='Computing Kernels for Band' + Band
    Observation, tita = L_Kernels.Compute_Kernels(Dict_Bands, Band, Context,MinimalProp=0.0)
    Observation['Tb']=ObservationOO['Tb']
    return Observation, tita 
    
def Load_Band_Kernels(HDFfilename,Context,Band):
    path=Context['wdir']
    Band=Band.upper()
    pklz_fn=path+'GC/'+HDFfilename+'_'+Band+'Ker_SHIFT'
    MSG="*******************************************************************************\n"
    if L_Files.exists(pklz_fn+'.pklz'):
        print('##   Using precomputed kernels for band',Band)
        Observation, tita=L_Files.load_obj(pklz_fn)
        MSG+='From PKL'
    else:
        Dict_Bands=L_ReadObs.Load_SingleBand_Ells(HDFfilename, Context,Band)
        print('######################################')
        print('##   PRECOMPUTING K for BAND',Band,'#####')
        print('######################################')
        MSG+='Computing Kernels for Band ' + Band
        Observation, tita = L_Kernels.Compute_Kernels(Dict_Bands, Band, Context,MinimalProp=0.9)
        Observation['Dict_Bands']=Dict_Bands
            
        L_Files.save_obj([Observation, tita ], pklz_fn)
            
    #LOG
    localtime = time.asctime( time.localtime(time.time()) ) + '\n'
    MSG+='\n'
    if Observation=={}:
        MSG+="Error loading (failed to geocorrect)" + HDFfilename + '\n'        
    else:
        MSG+="Loaded " + HDFfilename + '\n' + "Band: " + Band + '\n' 
        MSG+= (not Observation['GeoCorrection']['OK'])*'FAILED GeoCorrecting: ' + Observation['GeoCorrection']['OK']*'GeoCorrection: '
        MSG+='[' + str(Observation['GeoCorrection']['X']) +', ' + str(Observation['GeoCorrection']['Y']) + ']\n'
        MSG+= 'LSQRSols: H ' + str(Observation['LSQRSols']['H']['Sol']) + ' Error:' + str(Observation['LSQRSols']['H']['norm']) + '\n'
        MSG+= '          V ' + str(Observation['LSQRSols']['V']['Sol']) + ' Error:' + str(Observation['LSQRSols']['V']['norm']) + '\n'
    logging.info(localtime + MSG)
    
    return Observation, tita 
