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

TbH = L_Syn_Img.Create_Random_Synth_Img(Context,sgm0=1.0,sgm1=1.0)
TbV = L_Syn_Img.Create_Trig_Synth_Img(Context)
Observation=L_Syn_Img.Simulate_PMW(TbH,TbV,Context,Observation,Obs_error_std=1.0)
L_ParamEst.compute_param([TbH,TbV], Observation)

#L_Output.Export_Solution([TbH,TbV], Context, Band, "SynthImg")

#%%
#Rodgers no bound
#%%

Sol=L_Solvers.Solve_Rodgers_IT_NB(Context,Observation, tita, damp=1.0,tol=0.00005, max_iter=50)

RMSEsHw=L_Syn_Img.RMSE_M(TbH,Sol[0],Context)
RMSEsVw=L_Syn_Img.RMSE_M(TbV,Sol[1],Context)    
print(" -done")
print("RMSEsHw: %.3f"%RMSEsHw)
print("RMSEsVw: %.3f"%RMSEsVw)


