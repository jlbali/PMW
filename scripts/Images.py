
# EXPERIMENTO 1

tot_img_samples=1 #CANTIDAD DE IMAGENES SINTETICAS GENERADAS




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

#import my_base_dir
#BASE_DIR = my_base_dir.get()
BASE_DIR="/home/rgrimson/Projects/PMW_Tychonov/"
csv_dir=BASE_DIR + 'Output/'

#Import libraries
import copy
import L_ReadObs
import L_Context
import L_Output
import L_NewSolvers
import L_ParamEst
import L_Syn_Img
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


#%%

MAPAS=['LCA_12500m','LCA_25000m','LCA_50000m']
MAPAS=['LCA_50000m','LCA_25000m','LCA_12500m']
MAPAS=['LCA_25000m'] # Solo para trigonometric.
#MAPAS=['Reg25'] # Solo para bimodal.




base_type = "TRIGONOMETRIC_TWO_TYPES" # Usar mapas LCA para esto.
mix_ratios = np.linspace(0.0, 1.0, 11)



export_solutions = False
export_real = True
#%%

np.random.seed(1)


for MAPA in MAPAS:
    wdir=BASE_DIR + 'Mapas/%s/'%MAPA
    Context=L_Context.Load_Context(wdir, NeighborsBehaviour)
    print("Context", Context.keys())
    VType = Context['Dict_Vars']["VType"]
    print("VType", VType)
    #sys.exit(0)
    n_cells = len(VType)
    print("Contexto cargado en", wdir)
    #print("Context", Context)
    #print("Keys: ", Context.keys())
    #print("sdfsdf: ", Context['Dict_Vars'].keys())
    VType = Context['Dict_Vars']["VType"]
    neighbors = Context["Neighbors"]
    extras ={ 
        "neighbors": neighbors,
        "obsstd": 1.0,        
    }

    L_Syn_Img.Set_Margins(Context)
    print("Margenes setados para mapa", MAPA)
    km=int(MAPA[4:6])
    Observations = {}
    titas = {}
    v_errs = {}

    for n_img in range(tot_img_samples):
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
        


                        
