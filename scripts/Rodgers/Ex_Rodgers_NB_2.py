# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 10:15:19 2017

@author: rgrimson
"""

# ANTES DE COMENZAR REVISAR MAPA Y EL NOMBRE DEL CSV

import my_base_dir
BASE_DIR = my_base_dir.get()
MAPA='Ajo_25km'
wdir=BASE_DIR + 'Mapas/%s/'%MAPA
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

import matplotlib.pyplot as plt
# INICIALIZO LOS VALORES A EXPERIMENTAR:
#%%
HDFfilename='GW1AM2_201207310439_227D_L1SGBTBR_2210210'
Band='K'

Context=L_Context.Load_Context(wdir)
Observation, tita=L_ReadObs.Load_Band_Kernels(HDFfilename,Context,Band)
L_Syn_Img.Set_Margins(Context)
#%%

RMSEHs=[]
RMSEVs=[]
STDHi0=[]
STDVi0=[]
STDHo0=[]
STDVo0=[]
STDHi1=[]
STDVi1=[]
STDHo1=[]
STDVo1=[]

STDs=[10**i for i in np.arange(-1,1.1,.01)]
for std in STDs:
    TbH = L_Syn_Img.Create_Random_Synth_Img(Context,sgm0=std,sgm1=std)
    TbV = L_Syn_Img.Create_Random_Synth_Img(Context,sgm0=std,sgm1=10)
    Observation=L_Syn_Img.Simulate_PMW(TbH,TbV,Context,Observation,Obs_error_std=1.0)
    L_ParamEst.compute_param([TbH,TbV], Observation)
    
    Sol=L_Solvers.Solve_Rodgers_IT_NB(Context,Observation, tita, damp=1.0,tol=0.00005, max_iter=50)

    RMSEH=L_Syn_Img.RMSE_M(TbH,Sol[0],Context)
    RMSEV=L_Syn_Img.RMSE_M(TbV,Sol[1],Context)    
    print(" -done")
    print("RMSEsHw: %.3f"%RMSEH)
    print("RMSEsVw: %.3f"%RMSEV)
    
    RMSEHs.append(RMSEH)
    RMSEVs.append(RMSEV)
    tita=L_ParamEst.recompute_param([TbH,TbV],Observation,tita)
    STDHi0.append(np.sqrt(tita['H']['sigma2'][0]))
    STDHi1.append(np.sqrt(tita['H']['sigma2'][1]))
    STDVi0.append(np.sqrt(tita['V']['sigma2'][0]))
    STDVi1.append(np.sqrt(tita['V']['sigma2'][1]))
    tita=L_ParamEst.recompute_param(Sol,Observation,tita)
    STDHo0.append(np.sqrt(tita['H']['sigma2'][0]))
    STDHo1.append(np.sqrt(tita['H']['sigma2'][1]))
    STDVo0.append(np.sqrt(tita['V']['sigma2'][0]))
    STDVo1.append(np.sqrt(tita['V']['sigma2'][1]))
#%%
plt.clf()
plt.plot(STDHi1,RMSEHs)
#plt.plot(STDHi1,RMSEVs)
plt.plot(STDs,STDs)
plt.plot(STDHi0,STDHo0)
plt.plot(STDHi1,STDHo1)
plt.plot(STDVi0,STDVo0)
plt.plot(STDVi1,STDVo1)

#%%
import pandas as pd
d={'STDs':STDs,'STDHi0':STDHi0,'STDHi1':STDHi1,'STDHo0':STDHo0,'STDHo1':STDHo1,'RMSEHs':RMSEHs}
df=pd.DataFrame(d)
#%%
aux=df.sort(['STDHi0'])
plt.clf()
plt.plot(aux['STDHi0'],aux['STDHo0'])
plt.plot(aux['STDHi0'],aux['STDHi0'])
    