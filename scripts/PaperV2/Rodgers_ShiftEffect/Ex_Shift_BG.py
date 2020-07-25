# -*- coding: utf-8 -*-

#WDIR
import my_base_dir
BASE_DIR = my_base_dir.get()
MAPA='LCA_25000m'
#MAPA='Ajo_25km'
wdir=BASE_DIR + 'Mapas/%s/'%MAPA
csv_dir=BASE_DIR + 'Output/'

import sys

import L_ReadObs
import L_Context
import L_Kernels as L_Kernels
import L_Output
import L_Solvers

import copy
import pickle

import L_ParamEst
import L_Syn_Img
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import L_Shift


def Coef_correlacion(img_sintetica,img_solucion): # Toma dos imagenes, cada imagen es un array
    matriz_correlacion = np.corrcoef(img_sintetica,img_solucion)
    return matriz_correlacion[0,1]

#%%

# INPUTS
HDFfilename='GW1AM2_201207310439_227D_L1SGBTBR_2210210'
#Bands=['ku','ka','c','x','K']
Bands=['KA','K','KU','C','X']
#Bands =  ['KA']
#Bands = ['KU']
Bands = ['C']

NeighborsBehaviour = L_Context.NO_NEIGHBORS
Methods = ['LSQR','Weights','Rodgers_IT',"Global_GCV_Tichonov"] #,'BGP','BGF']
#Method = 'Rodgers_ITH'
Methods = ['EM']
Img_rep=1 #CANTIDAD DE IMAGENES SINTETICAS GENERADAS

#COORDENADAS A SHIFTEAR, OBSERVAR 0,0 ES LA ORIGINAL
x_shift = [0,67,125,250,500,1000,2000,4000,8000,-67,-125,-250,-500,-1000,-2000,-4000,-8000]
x_shift = [-4000,-3000,-2000,-1500,-1000,-500,-250,0,250,500,1000,1500,2000,3000,4000]
y_shift = [0]

csv_name = 'shift_soloX_10km_conRMSE_border'
#csv_name = 'exp_bands_methods_shifts_25km'
#%%
Band='C'
columnas = ['shift_x','shift_y','Method','Band','Pol','Img_num','Mu_real_0','Sigma2_real_0','Mu_real_1','Sigma2_real_1','Mu_0','Sigma2_0','Mu_1','Sigma2_1','RSME','RMSE_border','CoefCorr','time'] #el 0 y el 1 correspondonen al LT
df = pd.DataFrame(columns=columnas)



Context=L_Context.Load_Context(wdir, NeighborsBehaviour)
L_Syn_Img.Set_Margins(Context)

#%%
#Bands={'KA','KU','K','X','C'}
Res={}
for Band in Bands:
    OrigObser, tita= L_Shift.Load_Band_Kernels(HDFfilename,Context,Band)
    TbH = L_Syn_Img.Create_Random_Synth_Img(Context)
    TbV = L_Syn_Img.Create_Trig_Synth_Img(Context)
    
    OrigObser=L_Syn_Img.Simulate_PMW(TbH,TbV,Context,OrigObser,Obs_error_std=1.0)
    
    L_Output.Export_Solution([TbH,TbV], Context, "Synth", "SynthImg_sh")
    
    RMSE=[]
    for Shift_X in x_shift:
        ShiftObser, tita = L_Shift .Shift_Band_Kernels(OrigObser,Context,Shift_X, Shift_Y)
        Sol=L_Solvers.Solve(Context,ShiftObser,Method, tita)
        RMSE_H=L_Syn_Img.COMP_COND(TbH,Sol[0],Context,Margin=False)[0]
        RMSE_V=L_Syn_Img.COMP_COND(TbV,Sol[1],Context,Margin=False)[0]
        RMSE.append([Shift_X,RMSE_H,RMSE_V])
        print("\nShift {0}, RMSE_H: {1}, RMSE_V: {2}".format(Shift_X,RMSE_H,RMSE_V))
        #L_Output.Export_Solution(Sol, Context, "Synth", "SynthImg_Sol_sh"+str(Shift_X).replace('-','m'))
    Res[Band]=np.array(RMSE)


#np.savetxt("/home/lbali/ShiftKA.csv", Res['KA'], delimiter=",")
#np.savetxt("/home/lbali/ShiftKU.csv", Res['KU'], delimiter=",")
np.savetxt("/home/lbali/ShiftC.csv", Res['C'], delimiter=",")    
