# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 09:45:54 2018

@author: mariela
"""

import my_base_dir
BASE_DIR = my_base_dir.get()
csv_dir=BASE_DIR + 'Output/'

#Import libraries
import copy
import L_ReadObs
import L_Context
import L_Output
import L_Solvers
import L_ParamEst
import L_Syn_Img
import numpy as np
import pandas as pd
import time
import pickle


def Coef_correlacion(img_sintetica,img_solucion): # Toma dos imagenes, cada imagen es un array
    matriz_correlacion = np.corrcoef(img_sintetica,img_solucion)
    return matriz_correlacion[0,1]

#%%
# INPUTS
HDFfilename='GW1AM2_201207310439_227D_L1SGBTBR_2210210'
NeighborsBehaviour = L_Context.NO_NEIGHBORS
csv_name ='corrida_noise'

#%%

MAPAS=['LCA_25000m']
Bands=['KA','KU','K','X','C']
Methods = ['EM','LSQR','BGF']

#Bands=['KA']
tot_img_samples=1 #CANTIDAD DE IMAGENES SINTETICAS GENERADAS
tot_obs_samples=1 #CANTIDAD DE OBS SINTETICAS GENERADAS por IMG
#Obs_errors = [0.25, 0.5,1.0, 2.0,4.0]

#Obs_errors = [np.exp(l) for l in np.linspace(np.log(.1),np.log(10),9)]
Obs_errors = np.arange(1.0,20.0, step=1.0)
export_solutions = False
#%%
#np.savetxt(csv_dir + "./Noise_BGF_KA.csv", Res['KA'], delimiter=",")   
#np.savetxt(csv_dir + "./Noise_BGF_KU.csv", Res['KU'], delimiter=",")   
#np.savetxt(csv_dir + "./Noise_BGF_K.csv", Res['K'], delimiter=",")   
#np.savetxt(csv_dir + "./Noise_BGF_X.csv", Res['X'], delimiter=",")   
#np.savetxt(csv_dir + "./Noise_BGF_C.csv", Res['C'], delimiter=",")   

#
#%%
POL=1 #1 o 2 que es H o V, que es T o R
bf=csv_dir + "./Noise_"
df=pd.DataFrame(index=Obs_errors)
for M in Methods:
    for B in Bands:
        fn=bf+M+'_'+B+'.csv'
        D=np.genfromtxt(fn,delimiter=',')[:,POL]
        C=M+'_'+B
        df[C]=D

#%%

import matplotlib.pyplot as plt
plt.clf()
d={'EM':'r','LSQR':'b','BGF':'g'}
for M in Methods:
    for B in ['K']:
        C=M+'_'+B
        plt.plot(df[C],color=d[M],linewidth=2)