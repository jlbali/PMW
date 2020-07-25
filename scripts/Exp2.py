


# EXPERIMENTO 2

Band='KU'
tot_img_samples=1 #CANTIDAD DE IMAGENES SINTETICAS GENERADAS
tot_obs_samples=1 #CANTIDAD DE OBS SINTETICAS GENERADAS por IMG




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 17:54:47 2019

@author: rgrimson
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 09:45:54 2018

@author: mariela
"""

# ADAPTAR ESTO PARA EL NUEVO ESQUEMA SIN POLARIZACION.
"""
Problemas:
    - GCV Tichonov aparente devolver el lambda mas bajo posible...
    - Siempre hace el geocorrecting, y no guarda nada en el GC.

Para que librería hdf5 ande en el Anaconda es preciso tener la última versión.

ESTE ESTA ANDANDO MEDIO RARO
"""

#import os

#os.environ['LD_LIBRARY_PATH'] = "/home/rgrimson/anaconda3/pkgs/hdf5-1.10.1-h9caa474_1/lib:/usr/local/lib:/home/rgrimson/anaconda3/lib"

#print("Environment: ", os.environ['LD_LIBRARY_PATH'])

#import my_base_dir
#BASE_DIR = my_base_dir.get()
BASE_DIR="../"
csv_dir=BASE_DIR + 'outputs/'

#Import libraries
import copy
import libs.L_ReadObs as L_ReadObs
import libs.L_Context as L_Context
import libs.L_Output as L_Output
import libs.L_NewSolvers as L_NewSolvers
import libs.L_ParamEst as L_ParamEst
import libs.L_Syn_Img as L_Syn_Img
import numpy as np
import pandas as pd
import time
import pickle
import sys
import matplotlib.pyplot as plt


# Tichonov no requiere saber calibracion instrumental...

#%%
# INPUTS
HDFfilename='GW1AM2_201207310439_227D_L1SGBTBR_2210210'
NeighborsBehaviour = L_Context.COMPUTE_IF_NECESSARY
#csv_name ='simple_Trigonometric_2'
csv_name ='exp2_' + Band

#%%
MAPAS=['LCA_12500m','LCA_25000m','LCA_50000m']
MAPAS=['LCA_50000m','LCA_25000m','LCA_12500m']
MAPAS=['LCA_25000m'] # Solo para trigonometric.
#MAPAS=['Reg25'] # Solo para bimodal.


Methods = ['LSQR','Weights','EM','Rodgers_IT','Global_GCV_Tichonov','BGF']
#Methods = ['EM','BGF','LSQR']
Methods = ['EM','EM_Adapt','GCV_Tichonov','LSQR']


#Bands=['KA','KU','K','X','C']
#Bands=['KA']
#Bands=[ 'X'] # Probar cambiando la banda.
#Bands=[ 'KU']
#Bands=['KA','KU','K','X','C']
#Bands=['K']
Bands=[Band]
#base_type = "TRIGONOMETRIC_ONE_TYPE"
base_type = "TRIGONOMETRIC_TWO_TYPES" # Usar mapas LCA para esto.
#base_type = "BIMODAL"  # Usar mapas Reg para esto.
#tipo ="RANDOM"
#tipo = "MIXED"
#mix_ratios = [1.0]
mix_ratios = [0.5]
#mix_ratios = np.linspace(0.0, 1.0, 10)
#mix_ratios = [0.0, 1.0]



#Obs_errors = [0.25, 0.5,1.0, 2.0,4.0]
#Obs_errors = [1.0]
#Obs_errors = [2.0]
#Obs_errors = [0.25]
#Obs_errors = [1e-3, 1e-2, 1e-1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
#Obs_errors = [1e-3, 1e-2, 1e-1, 0.25, 0.3, 0.4, 0.5, 0.6,0.7, 0.8, 0.9, 1.0, 1.1, 1.2,1.3,1.4,1.5,1.7,1.85, 2.0, 5.0, 10.0, 20.0]
#Obs_errors = np.linspace(0.1, 2.0, 20)
Obs_errors = np.linspace(0.1, 2.0, 10)
#Obs_errors = [1.0]
# Para probar con distintos obs_errors... por màs y por menos.

export_solutions = False
export_real = False
#%%

# Levantamos el diccionario de coeficientes de Backus-Gilbert..
dic_BG_coef = {}
if 'BGF' in Methods:
    for MAPA in MAPAS:
        dic_mapa = {}
        dic_BG_coef[MAPA] = dic_mapa
        for Band in Bands:
            filepath = csv_dir + "BG_" + MAPA + "_" + Band + ".pickle"
            with open(filepath, 'rb') as f:
                # The protocol version used is detected automatically, so we do not
                # have to specify it.
                data = pickle.load(f)
                dic_mapa[Band] = data



np.random.seed(1)
columnas = ['CellSize','Method','Band','mix_ratio','Img_num','Obs_num','Obs_std_error','RMSE','CoefCorr','time'] #el 0 y el 1 correspondonen al LT
df = pd.DataFrame(columns=columnas)



for MAPA in MAPAS:
    wdir=BASE_DIR + 'maps/%s/'%MAPA
    Context=L_Context.Load_Context(wdir, NeighborsBehaviour)
    print("Contexto cargado en", wdir)
    #print("Context", Context)
    #print("Keys: ", Context.keys())
    #print("sdfsdf: ", Context['Dict_Vars'].keys())
    VType = Context['Dict_Vars']["VType"]
    neighbors = Context["Neighbors"]

    L_Syn_Img.Set_Margins(Context)
    print("Margenes setados para mapa", MAPA)
    km=int(MAPA[4:6])
    Observations = {}
    titas = {}
    v_errs = {}
    extras ={ 
        "neighbors": neighbors,
        "obsstd": 1.0,        
    }                  
    print("Precomputo de bandas")
    for Band in Bands:
        print("Banda ", Band)
        Observation, tita=L_ReadObs.Load_PKL_or_Compute_Kernels(HDFfilename,Context,Band,GC=True)
        Observations[Band] = Observation
        titas[Band] = tita
        K=Observation['Wt']
        n_cells = K.shape[1] # No cambia con la banda.
        n_ell = K.shape[0]
        v_errs[Band] = {}
        for Obs_std in Obs_errors:
            v_errs[Band][Obs_std] = []
            for n_obs in range(tot_obs_samples):
                v_err=np.random.normal(0,Obs_std,n_ell)
                v_errs[Band][Obs_std].append(v_err)
    print("Fin precomputo de bandas")
    for n_img in range(tot_img_samples):
        print("n_img", n_img)
        if base_type == "BIMODAL":
            #Tb_base = L_Syn_Img.Create_Bimodal_Synth_Img(Context, mu=180, sdev=0.3)
            Tb_base = L_Syn_Img.Create_True_Bimodal_Synth_Img(Context)
            Tb_random = 230 + np.random.normal(0,10,n_cells)
            L_Output.Export_Solution([Tb_base,Tb_base], Context, "RealBase", 'RealBase' )                                 
        elif base_type =="TRIGONOMETRIC_ONE_TYPE":
            Tb_base = L_Syn_Img.Create_Trig_Synth_Img0(Context)
            Tb_random = L_Syn_Img.Create_Random_Synth_Img(Context)
        elif base_type == "TRIGONOMETRIC_TWO_TYPES":
            Tb_base = L_Syn_Img.Create_Trig_Synth_Img(Context)
            Tb_random = L_Syn_Img.Create_Random_Synth_Img(Context)
        else:
            sys.exit(0)
        
        
        for mix_ratio in mix_ratios:
            print("Mix ratio: ", mix_ratio)
            """            
            if base_type == "BIMODAL":
                 Tb = Tb_base + (1-mix_ratio)*Tb_random
            else:
                 Tb = mix_ratio*Tb_base + (1-mix_ratio)*Tb_random
            """
            Tb = mix_ratio*Tb_base + (1-mix_ratio)*Tb_random
            if export_real:
                 Sol = [Tb, Tb]
                 L_Output.Export_Solution(Sol, Context, "Real", 'Real_'+str(km)+'_' +str(mix_ratio).replace('.','_') )                     
        

            for Band in Bands:
                print("Band", Band)
                Observation = Observations[Band]
                tita = titas[Band]
                print("Banda", Band, "cargada")
                K=Observation['Wt']
                n_ell = K.shape[0]
                n_cells = K.shape[1]
                for Obs_std in Obs_errors:
                   print("Obs_std", Obs_std)
                   titaReal=copy.deepcopy(L_ParamEst.Compute_NoPol_param(Tb,VType))
                   for n_obs in range(tot_obs_samples):
                        print("n_obs", n_obs)
                        v_err=v_errs[Band][Obs_std][n_obs]
                        Tb_sim=L_Syn_Img.Simulate_NoPol_PMW(K,Tb,Obs_error_std=0)
                        Tb_sim+=v_err
                        for Method in Methods:        
                            print("Method", Method)
                            t_inicial = time.time()                
                            #corrergir esto!!!
                            extras_adapt ={ 
                                "neighbors": neighbors,
                                "obsstd": Obs_std        
                            }                  

                            if Method == "EM_Adapt":
                                Sol=L_NewSolvers.Solve(VType, K, Tb_sim, "EM", extras_adapt)
                            else:
                                Sol=L_NewSolvers.Solve(VType, K, Tb_sim, Method, extras)
                
                            t_final = time.time()
                            t = t_final - t_inicial
                            
                            L_ParamEst.pr_NoPol(titaReal,'Synthetic Image')
                            tita=L_ParamEst.Compute_NoPol_param(Sol, VType)
                            L_ParamEst.pr_NoPol(tita, 'Reconstructed Image')
                    
                            dic = {'CellSize':km,'Method': Method,'Band':Band,'Img_num': n_img,'Obs_num': n_obs,'Obs_std_error':Obs_std, 'mix_ratio': mix_ratio,
                              'RMSE':(L_Syn_Img.COMP_COND(Tb,Sol,Context,Margin=False)[0]),
                              'CoefCorr':(L_Syn_Img.COMP_COND(Tb,Sol,Context,Margin=False)[1]),
                              'time':t}
                            df = df.append(dic, ignore_index=True)
            
                            #print('termino:',km,Band,Method,Obs_std,n_img,n_obs)
                            if export_solutions:
                                Sol = [Sol, Sol]
                                L_Output.Export_Solution(Sol, Context, Band, 'S_'+str(km)+'_'+Band+'_'+Method+str(n_img)+'_'+str(n_obs)+'_'+str(Obs_std).replace('.','_') + "_"  +str(mix_ratio).replace('.','_') )

                        
                        

df.to_csv(csv_dir + csv_name + '.csv')
            

