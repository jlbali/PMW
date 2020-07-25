# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 10:15:19 2017

@author: rgrimson
"""

# ANTES DE COMENZAR REVISAR MAPA Y EL NOMBRE DEL CSV

import my_base_dir
BASE_DIR = my_base_dir.get()
MAPA='Ajo_10km'
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

# INICIALIZO LOS VALORES A EXPERIMENTAR:
#%%
HDFfilename='GW1AM2_201207310439_227D_L1SGBTBR_2210210'
Method = 'BGF'
Band='K'

Context=L_Context.Load_Context(wdir)
Observation, tita=L_ReadObs.Load_Band_Kernels(HDFfilename,Context,Band)
TbH = L_Syn_Img.Create_Random_Synth_Img(Context)
TbV = L_Syn_Img.Create_Trig_Synth_Img(Context)
Observation=L_Syn_Img.Simulate_PMW(TbH,TbV,Context,Observation,Obs_error_std=1.0)
L_Syn_Img.Set_Margins(Context)
L_Output.Export_Solution([TbH,TbV], Context, Band, "SynthImg")

#%%
#FIND Ws
#%%
gamma = np.pi/4
w_min=10**-50
RMSEsVw = 1000
RMSE_gain=1.0
w=10

while ((w>w_min) and (RMSE_gain>0)):
    w_anterior = w
    w = w/10
    RMSEsVw_anterior = RMSEsVw
    
    Sol=L_Solvers.BGF(Context, Observation, gamma=gamma,w=w)
    RMSEsHw=L_Syn_Img.RMSE_M(TbH,Sol[0],Context)
    RMSEsVw=L_Syn_Img.RMSE_M(TbV,Sol[1],Context)    
    print(" -done")
    print("RMSEsHw: %.3f"%RMSEsHw)
    print("RMSEsVw: %.3f"%RMSEsVw)

    RMSE_gain = RMSEsVw_anterior - RMSEsVw


w = w_anterior
print("Best w: %.3f"%w)

#%%
#GAMMAS
#%%
gammas =np.linspace(0,np.pi/2,10)
N=len(gammas)
RMSEsH=np.zeros(N)
RMSEsV=np.zeros(N)
#%%
for i in range(N):
    gamma=gammas[i]
    Sol=L_Solvers.BGF(Context, Observation, gamma=gamma,w=w)
    RMSEsH[i]=L_Syn_Img.RMSE_M(TbH,Sol[0],Context)
    RMSEsV[i]=L_Syn_Img.RMSE_M(TbV,Sol[1],Context)
    print("- done")
    print("RMSEsH: %.3f"%RMSEsH[i])
    print("RMSEsV: %.3f"%RMSEsV[i])
    L_Output.Export_Solution(Sol, Context, Band, 'Sol20120731_BGF_g_'+str(i)+'_'+str(int(gamma*100)))
#%%
#Find best gamma
gammai=np.argmin(RMSEsV)
gamma=gammas[gammai]
print("Best gamma: %.3f"%gamma)
#%%
#REFINE
gammam=gammas[gammai-1]
gammaM=gammas[gammai+1]
Q=10
delta=(gammaM-gammam)/Q
gammaR =np.linspace(gammam+delta,gammaM-delta,Q)
M=len(gammaR)
RMSEsHr=np.zeros(M)
RMSEsVr=np.zeros(M)
for i in range(M):
    gamma=gammaR[i]
    Sol=L_Solvers.BGF(Context, Observation, gamma=gamma,w=w)
    RMSEsHr[i]=L_Syn_Img.RMSE_M(TbH,Sol[0],Context)
    RMSEsVr[i]=L_Syn_Img.RMSE_M(TbV,Sol[1],Context)
    print("- done")
    print("RMSEsH: %.3f"%RMSEsH[i])
    print("RMSEsV: %.3f"%RMSEsV[i])
    L_Output.Export_Solution(Sol, Context, Band, 'Sol20120731_BGF_R_'+str(i)+'_'+str(int(gamma*100)))
#%%
gammai=np.argmin(RMSEsVr)
gamma=gammaR[gammai]
print("Best gamma: %.3f"%gamma)
#%%
import matplotlib.pyplot as plt
plt.figure()
plt.plot(gammas, np.log(RMSEsV))
plt.figure()
plt.plot(gammaR, np.log(RMSEsVr))
