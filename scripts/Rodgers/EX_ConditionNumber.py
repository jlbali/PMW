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

from numpy import linalg as LA
#%%
def Kond(K):
    SV=LA.svd(K)[1]
    M=SV.max()
    SV[SV<0.000001]=65535
    return M/SV.min()
    
#%%
#PROCESS PARAMETERS
HDFfilename='GW1AM2_201208110420_224D_L1SGBTBR_2210210'


Method="Rodgers_IT"
Method="Weights"
Method="Rodgers"
#%%


def main(): 
#%%
    for MAPA in ['Ajo_10km','Ajo_25km']:
        wdir=BASE_DIR + 'Mapas/%s/'%MAPA
        Context=L_Context.Load_Context(wdir)
        L_Syn_Img.Set_Margins(Context)
        NoMargin=Context['NoMargin_Vars']
        for Band in ['c','x','ku','K','ka']:
            Observation, tita=L_ReadObs.Load_PKL_or_Compute_Kernels(HDFfilename,Context,Band,GC=True)
            K=Observation['Wt']            
            CN=Kond(K)
            NRM=LA.norm(K)
            Q=CN/NRM
            print("Band: %s\tMap:%s\tCondNumb:%.2f, Norm:%.2f, Q:%.2f"%(Band,MAPA,CN,NRM,Q))
            