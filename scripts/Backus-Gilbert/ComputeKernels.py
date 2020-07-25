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
import L_Files
import L_ParamEst
import L_Syn_Img
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import pickle

#%%   BG init
dic_maxObs = {'KA12':6,'KU12':6,'K12':6,'X12':4,'C12':4,'KA25':16,'KU25':16,'K25':16,'X25':10,'C25':10,'KA50':32,'KU50':30,'K50':30,'X50':16,'C50':16}
dic_omega = {50: 0.001, 25: .001, 12: 0.001}
dict_gamma = {'X12': 0.61236513052394792, 'KU12': 0.0001, #computed from 10 iteration with Trigonometric synth img
              'C12': 0.7682029405253775, 'KA12': 0.0001, 
              'X25': 0.18304225751445416, 'K12': 0.0001,
              'KA25': 0.096957173174296807, 'KU25': 0.10875923786109196, 
              'K25': 0.2027251484990118, 'C25': 0.096180595400429697, 
              'KA50': 0.029414081607707171, 'C50': 0.046249520754752083, 
              'KU50': 0.043028161100192239, 'K50': 0.05660389107298016, 
              'X50': 0.059595153609357167}


#%%

#%%
# INPUTS
HDFfilename='GW1AM2_201207310439_227D_L1SGBTBR_2210210'
NeighborsBehaviour = L_Context.NO_NEIGHBORS 
#L_Context.LOAD_NEIGHBORS
csv_name ='corrida_TODOS'

#%%
Obs_std  = 1.0
MAPAS=['LCA_12500m','LCA_25000m','LCA_50000m']
MAPAS=['LCA_50000m','LCA_25000m','LCA_12500m']
#MAPAS=['LCA_12500m']
Bands=['KA','KU','K','X','C']
#Bands=['KA']
tot_img_samples=1 #CANTIDAD DE IMAGENES SINTETICAS GENERADAS

#%%

np.random.seed(1)
#cols = ['km','Band','MaxObs','w','Gamma','RMSE']
i=0

for MAPA in MAPAS:
    wdir=BASE_DIR + 'Mapas/%s/'%MAPA
    Context=L_Context.Load_Context(wdir, NeighborsBehaviour)
    L_Syn_Img.Set_Margins(Context)
    km=int(MAPA[4:6])
    omega=dic_omega[km]
    distances_path = wdir + 'distance'
    if not L_Files.exists(distances_path):
       distances = L_Solvers.precompute_all_optimized(Context)
       L_Solvers.save_distances(distances_path, distances)
    distances = L_Solvers.load_distances(distances_path)


    for Band in Bands:
        Observation, tita=L_ReadObs.Load_Band_Kernels(HDFfilename,Context,Band)
    
        key = Band + str(km)
        max_obs = dic_maxObs[key]
        omega = dic_omega[km]
        gamma = dict_gamma[key]
        
        #usar diccionarios con KEY para max_obs + omega + gamma para calcular kernel
        print("Computando coeficientes para banda ", Band, " y Mapa ", MAPA)
        COEF=L_Solvers.BG_Precomp_MaxObs_Cell(distances, Context, Observation, gamma=gamma,w=omega,error_std=1,MaxObs=max_obs)
        print("Guardando los coeficientes...")
        filepath = csv_dir + "BG_" + MAPA + "_" + Band + ".pickle"        
        with open(filepath, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(COEF, f, pickle.HIGHEST_PROTOCOL)

        #GUARDAR COEF
