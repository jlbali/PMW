# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 09:45:54 2018

@author: mariela
"""

import my_base_dir
#BASE_DIR = '../../'   
BASE_DIR = my_base_dir.get()
csv_dir=BASE_DIR + 'Output/'

print(csv_dir)




#Import libraries
import copy

import L_ReadObs
import L_Context
import L_Output
import L_Solvers
import L_ParamEst
import L_Syn_Img
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import pickle


def Coef_correlacion(img_sintetica,img_solucion): # Toma dos imagenes, cada imagen es un array
    matriz_correlacion = np.corrcoef(img_sintetica,img_solucion)
    return matriz_correlacion[0,1]

#%%
# INPUTS
HDFfilename='GW1AM2_201207310439_227D_L1SGBTBR_2210210'
#NeighborsBehaviour = L_Context.NO_NEIGHBORS
NeighborsBehaviour = L_Context.COMPUTE_IF_NECESSARY

#COMPUTE_IF_NECESSARY
csv_name ='corrida'

#%%
MAPAS=['Reg25']
#MAPAS=['LCA_25000m']

Methods = ['LSQR','Weights','Rodgers_IT','Global_GCV_Tichonov','BGF']
#Methods = ['Weights','Rodgers_IT']
Methods = ['Rodgers_IT']
Methods = ['Rodgers_IT', 'Global_GCV_Tichonov', 'GCV_Tichonov']
Methods = ['LSQR', 'Rodgers_IT', 'GCV_Tichonov']

#Methods = ['LSQR','Weights','Rodgers_IT','Global_GCV_Tichonov']


Bands=['KA','KU','K','X','C']
#Bands=['KU','X']
#Bands=['KA']
#Bands=['X']
tot_img_samples=1 #CANTIDAD DE IMAGENES SINTETICAS GENERADAS
tot_obs_samples=1 #CANTIDAD DE OBS SINTETICAS GENERADAS por IMG
#Obs_errors = [0.25, 0.5,1.0, 2.0,4.0]
gammas = np.array([0, 0.5, 1])
#gammas = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
export_solutions = True
#%%

# Levantamos el diccionario de coeficientes de Backus-Gilbert..
#dic_BG_coef = {}
#for MAPA in MAPAS:
#    dic_mapa = {}
#    dic_BG_coef[MAPA] = dic_mapa
#    for Band in Bands:
#        filepath                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               = csv_dir + "BG_" + MAPA + "_" + Band + ".pickle"
#        with open(filepath, 'rb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
#            data = pickle.load(f)
#            dic_mapa[Band] = data



np.random.seed(1)
columnas = ['CellSize','Method','Band','Pol','Img_num','Obs_num','gamma','RMSE','CoefCorr','time','Mu_real_0','Mu_rec_0','Sigma_real_0','Sigma_rec_0'] #el 0 y el 1 correspondonen al LT
df = pd.DataFrame(columns=columnas)

for MAPA in MAPAS:
    wdir=BASE_DIR + 'Mapas/%s/'%MAPA
    Context=L_Context.Load_Context(wdir, NeighborsBehaviour)
    L_Syn_Img.Set_Margins(Context)
    #TbTrig=L_Syn_Img.Create_Trig_Synth_Img0(Context)
    TbBiModal = L_Syn_Img.Create_Bimodal_Synth_Img(Context)
                                                                                                                                        
    for gamma in gammas:

      for n_img in range(tot_img_samples):
        TbHa=L_Syn_Img.Create_Random_Synth_Img(Context)
        TbHb=L_Syn_Img.Create_Random_Synth_Img(Context)
        
        TbH=gamma*TbHa + (1-gamma)*TbBiModal
        TbV=gamma*TbHb + (1-gamma)*TbBiModal

        titaReal=copy.deepcopy(L_ParamEst.Compute_param([TbH,TbV],Context))
        
    
        for Band in Bands:
              Observation, tita=L_ReadObs.Load_PKL_or_Compute_Kernels(HDFfilename,Context,Band,GC=False)
              
              print("Observation", Observation)
    
    
              #for Obs_std in Obs_errors:
              Obs_std=1
                
              for n_obs in range(tot_obs_samples):
                #Observation=L_Syn_Img.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         (TbH,TbV,Context,Observation,Obs_error_std=Obs_std)
    
    
                for Method in Methods:
                        Observation=L_Syn_Img.Simulate_PMW(TbH,TbV,Context,Observation,Obs_error_std=Obs_std)
                        L_Output.Export_Solution([TbH,TbV], Context, 'P', 'LB')
    
                        t_inicial = time.time()                
                        Sol=L_Solvers.Solve(Context,Observation,Method, tita, obsstd=Obs_std)
            
                        t_final = time.time()
                        t = t_final - t_inicial
                        
                        #L_ParamEst.pr(titaReal,'Synthetic Image')
                        #tita=L_ParamEst.recompute_param(Sol,Observation,tita)
                        L_ParamEst.pr(tita, 'Reconstructed Image')
                        km = 25
                        dic = {'CellSize':km,'Method': Method,'Band':Band,'Pol': 'H','Img_num': n_img,'Obs_num': n_obs,'gamma':gamma,'RMSE':float(str(L_Syn_Img.COMP_COND(TbH,Sol[0],Context,Margin=False)[0])),'CoefCorr':float(str(L_Syn_Img.COMP_COND(TbH,Sol[0],Context,Margin=False)[1])),'time':t,'Mu_real_0':titaReal['H']['mu'][0],'Sigma_real_0':np.sqrt(titaReal['H']['sigma2'][0]),'Mu_rec_0':tita['H']['mu'][0],'Sigma_rec_0':np.sqrt(tita['H']['sigma2'][0])}
                        df = df.append(dic, ignore_index=True)
                        
                        dic = {'CellSize':km,'Method': Method,'Band':Band,'Pol': 'V','Img_num': n_img,'Obs_num': n_obs,'gamma':gamma,'RMSE':float(str(L_Syn_Img.COMP_COND(TbV,Sol[1],Context,Margin=False)[0])),'CoefCorr':float(str(L_Syn_Img.COMP_COND(TbV,Sol[1],Context,Margin=False)[1])),'time':t,'Mu_real_0':titaReal['V']['mu'][0],'Sigma_real_0':np.sqrt(titaReal['V']['sigma2'][0]),'Mu_rec_0':tita['V']['mu'][0],'Sigma_rec_0':np.sqrt(tita['V']['sigma2'][0])}
                        df = df.append(dic, ignore_index=True)

                        print('termino:',Band,Method,Obs_std,n_img,n_obs)
                        if export_solutions:
                            L_Output.Export_Solution(Sol, Context, Band, 'S_R25_'+Band+'_'+Method+str(n_img)+'_'+str(n_obs)+'_'+str(gamma).replace('.','_'))

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
                        
#%%

print("Grabando en ", csv_dir + csv_name + '.csv' )
df.to_csv(csv_dir + csv_name + '.csv')


df = pd.read_csv(csv_dir + csv_name + '.csv')
#%%
dfX=df[df.Band == "X"]
dfKU=df[df.Band == "KU"]

#X
dfX_Tich = dfX[dfX.Method == "GCV_Tichonov"]
dfX_Rodgers = dfX[dfX.Method== "Rodgers_IT"]
dfX_LSQR = dfX[dfX.Method== "LSQR"]

dfX_Rodgers=dfX_Rodgers[['gamma', 'RMSE']].groupby('gamma').mean()
dfX_Tich=dfX_Tich[['gamma', 'RMSE']].groupby('gamma').mean()
dfX_LSQR=dfX_LSQR[['gamma', 'RMSE']].groupby('gamma').mean()

dgX=pd.DataFrame({'Tich':dfX_Tich.RMSE.values.astype(float),'Rodg':dfX_Rodgers.RMSE.values.astype(float), 'LSQR':dfX_LSQR.RMSE.values.astype(float)},index=dfX_Tich.index.values)
dgX.plot()


#KU
dfKU_Tich = dfKU[dfKU.Method == "GCV_Tichonov"]
dfKU_Rodgers = dfKU[dfKU.Method== "Rodgers_IT"]
dfKU_LSQR = dfKU[dfKU.Method== "LSQR"]

dfKU_Rodgers=dfKU_Rodgers[['gamma', 'RMSE']].groupby('gamma').mean()
dfKU_Tich=dfKU_Tich[['gamma', 'RMSE']].groupby('gamma').mean()
dfKU_LSQR=dfKU_LSQR[['gamma', 'RMSE']].groupby('gamma').mean()

dgKU=pd.DataFrame({'Tich':dfKU_Tich.RMSE.values.astype(float),'Rodg':dfKU_Rodgers.RMSE.values.astype(float), 'LSQR':dfKU_LSQR.RMSE.values.astype(float)},index=dfKU_Tich.index.values)
dgKU.plot()



#            
#
#Band='KU' #no hay otra
#Pol='V'
#Plot segun 'Method'
#

#df_Tich = df[df.Method == "GCV_Tichonov"].copy()
#df_Rodgers = df[df.Method== "Rodgers_IT"].copy()
#
#df_Rodgers=df_Rodgers[['Noise_std_error', 'RMSE']].groupby('Noise_std_error').mean()
#df_Tich=df_Tich[['Noise_std_error', 'RMSE']].groupby('Noise_std_error').mean()
##
#dg=pd.DataFrame({'Tich':df_Tich.RMSE.values.astype(float),'Rodg':df_Rodgers.RMSE.values.astype(float)},index=df_Tich.index.values)
##dg
#dg.plot()
