#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 14:04:03 2019

@author: lbali
"""

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
import scipy as sp


BASE_DIR="/home/lbali/PMW-Tychonov/"
csv_dir=BASE_DIR + 'Output/'


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
BASE_DIR="/home/lbali/PMW-Tychonov/"
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
csv_name ='simple_BiModal_3'

#%%
MAPAS=['LCA_12500m','LCA_25000m','LCA_50000m']
MAPAS=['LCA_50000m','LCA_25000m','LCA_12500m']
MAPAS=['LCA_25000m']
MAPAS=['Reg25']


Methods = ['LSQR','Weights','EM','Rodgers_IT','Global_GCV_Tichonov','BGF']
#Methods = ['EM','BGF','LSQR']
Methods = ['EM','GCV_Tichonov','LSQR']


Bands=['KA','KU','K','X','C']
Bands=['KA']
Bands=['KA', 'C']
Bands=['KA','KU','K','X','C']
tot_img_samples=1 #CANTIDAD DE IMAGENES SINTETICAS GENERADAS
tot_obs_samples=1 #CANTIDAD DE OBS SINTETICAS GENERADAS por IMG
#base_type = "TRIGONOMETRIC"
base_type = "BIMODAL"
#tipo ="RANDOM"
#tipo = "MIXED"
#mix_ratios = [1.0]
mix_ratios = np.linspace(0.0, 1.0, 10)
#mix_ratios = [0.0, 1.0]



#Obs_errors = [0.25, 0.5,1.0, 2.0,4.0]
Obs_errors = [1.0]
export_solutions = False
export_real = True
#%%

# Levantamos el diccionario de coeficientes de Backus-Gilbert..

np.random.seed(1)


for MAPA in MAPAS:
    wdir=BASE_DIR + 'Mapas/%s/'%MAPA
    Context=L_Context.Load_Context(wdir, NeighborsBehaviour)
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
    
    n_Vars=Context['Dict_Vars']['n_Vars']    
    VType=Context['Dict_Vars']['VType']
    nlt=VType.max()+1

    nlt=VType.max()+1
        
    Tb=np.zeros(n_Vars)
    k=.2*Context['Param']['DX']/12500.0
    #sh=Context['Param']['DX']/12500.0/2
    x_length = 18
    y_length = 20
    #%%        
    for v in range(n_Vars):
        x_coord = Context['Dict_Vars']['Var'][v]['Coords'][0][1]+.5
        y_coord = Context['Dict_Vars']['Var'][v]['Coords'][0][0]+.5
        print(x_coord,y_coord)
        x_arg = (x_coord / x_length)-.5
        y1_arg = (y_coord / y_length)-.25
        y2_arg = (y_coord / y_length)-.75
        
        Tb[v]= (sp.stats.norm.pdf(np.sqrt(x_arg**2+y1_arg**2))-sp.stats.norm.pdf(np.sqrt(x_arg**2+y2_arg**2)))*100+180
        
    L_Output.Export_Solution([Tb,Tb], Context, "Real", 'TestImage' )
    #np.histogram(Tb)
    a = np.hstack(Tb)
    _ = plt.hist(a, bins=20)  # arguments are passed to np.histogram
    plt.title("Histogram with 'auto' bins")
    plt.show()
    
    #%%        
    for v in range(n_Vars):
        x_coord = Context['Dict_Vars']['Var'][v]['Coords'][0][1]+.5
        y_coord = Context['Dict_Vars']['Var'][v]['Coords'][0][0]+.5
        x_arg = (x_coord / x_length)*1.5*np.pi+np.pi/4
        y_arg = ((y_coord - y_length/2)/ (y_length/2))*2.0
        Tb[v]= (np.sin(x_arg)*(sp.stats.norm.pdf(y_arg)+.7)+0.5)*100+ 180
    L_Output.Export_Solution([Tb,Tb], Context, "Real", 'TestImage' )
    #np.histogram(Tb)
    a = np.hstack(Tb)
    _ = plt.hist(a, bins=30)  # arguments are passed to np.histogram
    plt.title("Histogram with 'auto' bins")
    plt.show()
    
    """
    # Normal multivariada con correlacion espacial a vecinos cercanos.
    
    n_Vars=Context['Dict_Vars']['n_Vars']    
    VType=Context['Dict_Vars']['VType']
    nlt=VType.max()+1

    nlt=VType.max()+1
        
    Tb=np.zeros(n_Vars)
    k=.2*Context['Param']['DX']/12500.0
    #sh=Context['Param']['DX']/12500.0/2
    x_length = 18
    y_length = 20
    """

                     