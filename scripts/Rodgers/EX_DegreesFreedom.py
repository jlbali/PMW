# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 13:47:57 2017

@author: rgrimson
"""
from __future__ import print_function


#WDIR
import my_base_dir
BASE_DIR = my_base_dir.get()


import sys
import numpy as np
import L_ReadObs
import L_Context
import L_Kernels as L_Kernels
import L_Output
import L_Solvers
import L_Syn_Img
import L_ParamEst

#%%
def main():    
    global BASE_DIR
    #PROCESS PARAMETERS
    HDFfilename='GW1AM2_201207310439_227D_L1SGBTBR_2210210'
    HDFfilename='GW1AM2_201207310439_227D_L1SGBTBR_2210210'

    Method = 'Rodgers_IT' 
    MAPAS=['Ajo_10km','Ajo_25km']    
    Bands=['KA','K','KU','X','C']
    damp=1
    ds={}
    
    for MAPA in MAPAS:
        wdir=BASE_DIR + 'Mapas/%s/'%MAPA
        Context=L_Context.Load_Context(wdir)
        ds[MAPA] = {}
        
        for Band in Bands:
            Observation, tita=L_ReadObs.Load_Band_Kernels(HDFfilename, Context, Band)#Load_PKL_or_Compute_Kernels(HDFfilename,Context,Band,GC=True)
            print("Degrees of freedom. Grid: %s, \tBand: %s (H,V)"%(MAPA,Band))
            ds[MAPA][Band]=DF(Context, Observation, tita, damp=damp)
            #%%
            K=Observation['Wt']
            W=K.dot(K.transpose())
            np.sqrt(1/W.diagonal().min())
            #%%            
            
    return ds
    

#%%
def DF(Context, Observation, tita, damp=1.0): #compute solution for one pol
        K=Observation['Wt']
        VType=Observation['VType']
        n_Vars=K.shape[1]
        sigma2=np.array([5.0,10.0])#tita[pol]['sigma2']

        Sx=np.zeros([n_Vars])
        for v in range(n_Vars):
            Sx[v]=abs(sigma2[VType[v]])
        
        M=np.linalg.inv(K.transpose().dot(K)/damp/damp+np.diag(1/Sx))
        Mds=M.dot(K.transpose().dot(K))/damp/damp
        ds=np.trace(Mds)
        Mdn=M/Sx
        dn=np.trace(Mdn)
        print("ds:%.2f  \tdn:%.2f  \td=%.2f"%(ds,dn,ds+dn))
        return ds
        
#%%
if __name__ == "__main__":
    ds=main()

#%%
import matplotlib.pyplot as plt
plt.clf()
#%%
MAPAS=['Ajo_10km','Ajo_25km']    
N=[1465,235]
Bands=['KA','KU','K','X','C']
HPDMs={'C':62000,'X':42000,'KU':22000,'K':26000,'KA':12000}   #major diam in m
HPBW=[12000,22000,26000,42000,62000]
for i in range(len(MAPAS)):
    MAPA=MAPAS[i]
    n=N[i]
    plt.plot(HPBW,[ds[MAPA][Band]/n for Band in Bands])
plt.plot(HPBW,[.5,.5,.5,.5,.5])    
#%%
for MAPA in MAPAS:
    for Band in Bands:
            print("Degrees of freedom. Grid: %s, \tBand: %s ds (H,V): \t%.2f"%(MAPA,Band,ds[MAPA][Band]))