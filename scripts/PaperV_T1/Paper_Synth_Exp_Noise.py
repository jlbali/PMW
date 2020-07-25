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
MAPAS=['LCA_12500m','LCA_25000m','LCA_50000m']
#MAPAS=['LCA_25000m']


Methods = ['LSQR','Weights','Rodgers_IT','Global_GCV_Tichonov','BGF']
#Methods = ['Weights','Rodgers_IT']
Methods = ['Rodgers_IT']
Methods = ['Rodgers_IT', 'Global_GCV_Tichonov', 'GCV_Tichonov']
Methods = ['Rodgers_IT', 'GCV_Tichonov']

#Methods = ['LSQR','Weights','Rodgers_IT','Global_GCV_Tichonov']


Bands=['KA','KU','K','X','C']
Bands=['KU','X']
#Bands=['KU']
tot_img_samples=1 #CANTIDAD DE IMAGENES SINTETICAS GENERADAS
tot_obs_samples=1 #CANTIDAD DE OBS SINTETICAS GENERADAS por IMG
#Obs_errors = [0.25, 0.5,1.0, 2.0,4.0]
Noise_STDs = np.array([0,.5,1])
export_solutions = True
#%%

# Levantamos el diccionario de coeficientes de Backus-Gilbert..
#dic_BG_coef = {}
#for MAPA in MAPAS:
#    dic_mapa = {}
#    dic_BG_coef[MAPA] = dic_mapa
#    for Band in Bands:
#        filepath = csv_dir + "BG_" + MAPA + "_" + Band + ".pickle"
#        with open(filepath, 'rb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
#            data = pickle.load(f)
#            dic_mapa[Band] = data



np.random.seed(1)
columnas = ['CellSize','Method','Band','Pol','Img_num','Obs_num','Noise_std_error','RMSE','CoefCorr','time','Mu_real_0','Mu_rec_0','Sigma_real_0','Sigma_rec_0','Mu_real_1','Mu_rec_1','Sigma_real_1','Sigma_rec_1'] #el 0 y el 1 correspondonen al LT
df = pd.DataFrame(columns=columnas)

for MAPA in MAPAS:
    wdir=BASE_DIR + 'Mapas/%s/'%MAPA
    Context=L_Context.Load_Context(wdir, NeighborsBehaviour)
    L_Syn_Img.Set_Margins(Context)
    km=int(MAPA[4:6])
    TbTrig = L_Syn_Img.Create_Trig_Synth_Img(Context)

    
    for Noise_std in Noise_STDs:

      for n_img in range(tot_img_samples):
        #TbH = L_Syn_Img.Create_RandTrig_Synth_Img(Context,sgmR0=Noise_std,sgmR1=Noise_std)
        #TbV = L_Syn_Img.Create_RandTrig_Synth_Img(Context,sgmR0=Noise_std,sgmR1=Noise_std)
        TbHa=L_Syn_Img.Create_Random_Synth_Img(Context)
        TbHb=L_Syn_Img.Create_Random_Synth_Img(Context)
        
        TbH=Noise_std*TbHa + (1-Noise_std)*TbTrig
        TbV=Noise_std*TbHb + (1-Noise_std)*TbTrig

        titaReal=copy.deepcopy(L_ParamEst.Compute_param([TbH,TbV],Context))
        
    
        for Band in Bands:
              Observation, tita=L_ReadObs.Load_PKL_or_Compute_Kernels(HDFfilename,Context,Band,GC=True)
    
    
              #for Obs_std in Obs_errors:
              Obs_std=1
                
              for n_obs in range(tot_obs_samples):
                Observation=L_Syn_Img.Simulate_PMW(TbH,TbV,Context,Observation,Obs_error_std=Obs_std)
    
    
                for Method in Methods:
    
    
                        t_inicial = time.time()                
                        #corrergir esto!!!
                        if Method == 'BGF':
                            BG_Coef= dic_BG_coef[MAPA][Band]
                            Sol=[BG_Coef.dot(Observation['Tb']['H']), BG_Coef.dot(Observation['Tb']['V'])]
                        else:
                            Sol=L_Solvers.Solve(Context,Observation,Method, tita, obsstd=Obs_std)
            
                        t_final = time.time()
                        t = t_final - t_inicial
                        
                        L_ParamEst.pr(titaReal,'Synthetic Image')
                        tita=L_ParamEst.recompute_param(Sol,Observation,tita)
                        L_ParamEst.pr(tita, 'Reconstructed Image')
                
                        dic = {'CellSize':km,'Method': Method,'Band':Band,'Pol': 'H','Img_num': n_img,'Obs_num': n_obs,'Noise_std_error':Noise_std,'RMSE':float(str(L_Syn_Img.COMP_COND(TbH,Sol[0],Context,Margin=False)[0])),'CoefCorr':float(str(L_Syn_Img.COMP_COND(TbH,Sol[0],Context,Margin=False)[1])),'time':t,'Mu_real_0':titaReal['H']['mu'][0],'Sigma_real_0':np.sqrt(titaReal['H']['sigma2'][0]),'Mu_real_1':titaReal['H']['mu'][1],'Sigma_real_1':np.sqrt(titaReal['H']['sigma2'][1]),'Mu_rec_0':tita['H']['mu'][0],'Sigma_rec_0':np.sqrt(tita['H']['sigma2'][0]),'Mu_rec_1':tita['H']['mu'][1],'Sigma_rec_1':np.sqrt(tita['H']['sigma2'][1])}
                        df = df.append(dic, ignore_index=True)
                        
                        dic = {'CellSize':km,'Method': Method,'Band':Band,'Pol': 'V','Img_num': n_img,'Obs_num': n_obs,'Noise_std_error':Noise_std,'RMSE':float(str(L_Syn_Img.COMP_COND(TbV,Sol[1],Context,Margin=False)[0])),'CoefCorr':float(str(L_Syn_Img.COMP_COND(TbV,Sol[1],Context,Margin=False)[1])),'time':t,'Mu_real_0':titaReal['V']['mu'][0],'Sigma_real_0':np.sqrt(titaReal['V']['sigma2'][0]),'Mu_real_1':titaReal['V']['mu'][1],'Sigma_real_1':np.sqrt(titaReal['V']['sigma2'][1]),'Mu_rec_0':tita['V']['mu'][0],'Sigma_rec_0':np.sqrt(tita['V']['sigma2'][0]),'Mu_rec_1':tita['V']['mu'][1],'Sigma_rec_1':np.sqrt(tita['V']['sigma2'][1])}
                        df = df.append(dic, ignore_index=True)

                        print('termino:',km,Band,Method,Obs_std,n_img,n_obs)
                        if export_solutions:
                            L_Output.Export_Solution(Sol, Context, Band, 'S_'+str(km)+'_'+Band+'_'+Method+str(n_img)+'_'+str(n_obs)+'_'+str(Obs_std).replace('.','_'))

                        
                        
   

print("Grabando en ", csv_dir + csv_name + '.csv' )
df.to_csv(csv_dir + csv_name + '.csv')
            
#%%
dfX=df[df.Band == "X"]
dfKU=df[df.Band == "KU"]

#X
dfX_Tich = dfX[dfX.Method == "GCV_Tichonov"]
dfX_Rodgers = dfX[dfX.Method== "Rodgers_IT"]

dfX_Rodgers=dfX_Rodgers[['Noise_std_error', 'RMSE']].groupby('Noise_std_error').mean()
dfX_Tich=dfX_Tich[['Noise_std_error', 'RMSE']].groupby('Noise_std_error').mean()

dgX=pd.DataFrame({'Tich':dfX_Tich.RMSE.values.astype(float),'Rodg':dfX_Rodgers.RMSE.values.astype(float)},index=dfX_Tich.index.values)
dgX.plot()


#KU
dfKU_Tich = dfKU[dfKU.Method == "GCV_Tichonov"]
dfKU_Rodgers = dfKU[dfKU.Method== "Rodgers_IT"]

dfKU_Rodgers=dfKU_Rodgers[['Noise_std_error', 'RMSE']].groupby('Noise_std_error').mean()
dfKU_Tich=dfKU_Tich[['Noise_std_error', 'RMSE']].groupby('Noise_std_error').mean()

dgKU=pd.DataFrame({'Tich':dfKU_Tich.RMSE.values.astype(float),'Rodg':dfKU_Rodgers.RMSE.values.astype(float)},index=dfKU_Tich.index.values)
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
