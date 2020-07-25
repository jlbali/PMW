# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 13:47:57 2017

@author: rgrimson
"""
from __future__ import print_function


#WDIR
import my_base_dir
BASE_DIR = my_base_dir.get()
MAPA='LCA_25000m'
csv_dir=BASE_DIR + 'Output/'
mapa_dir = BASE_DIR + "Mapas/"


import sys

import L_ReadObs
import L_Context
import L_Kernels as L_Kernels
import L_Output
import L_Solvers
import L_Syn_Img
import pickle

#%%

HDFfilename='GW1AM2_201406020358_220D_L1SGBTBR_2220220'


MAPAS=['LCA_25000m']
Bands=['KA','KU', 'K', 'X', 'C']


# Levantamos el diccionario de coeficientes de Backus-Gilbert..
dic_BG_coef = {}
for MAPA in MAPAS:
    dic_mapa = {}
    dic_BG_coef[MAPA] = dic_mapa
    for Band in Bands:
        filepath = csv_dir + "RI_" + MAPA + "_" + Band + ".pickle"
        with open(filepath, 'rb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            data = pickle.load(f)
            dic_mapa[Band] = data



# Out_Grids/Joined es el shapefile final...

#BG_Coef= dic_BG_coef[MAPA][Band]
#Sol=[BG_Coef.dot(Observation['Tb']['H']), BG_Coef.dot(Observation['Tb']['V'])]



dic_maxObs={'KA12':6,'KU12':6,'K12':6,'X12':4,'C12':4,'KA25':16,'KU25':16,'K25':16,'X25':10,'C25':10,'KA50':32,'KU50':30,'K50':30,'X50':16,'C50':16}

# INPUTS
NeighborsBehaviour = L_Context.COMPUTE_IF_NECESSARY



MAPA = MAPAS[0]

def compute_and_save_distances(MAPA):
    wdir=BASE_DIR + 'Mapas/%s/'%MAPA
    Context=L_Context.Load_Context(wdir, NeighborsBehaviour)
    L_Syn_Img.Set_Margins(Context)    
    distances = L_Solvers.precompute_all_optimized(Context)
    L_Solvers.save_distances(csv_dir + "/"+MAPA + "_distances.pickle", distances)

def compute_and_save_coefficients(MAPA, Band):
    #grilla_gammas = np.linspace(start = 0.0001, stop= np.pi/2.0, num=CANT_GAMMAS )
    grilla_gammas = L_Solvers.grillaLogaritmica(0.0001, np.pi/2, CANT_GAMMAS) 
    wdir=BASE_DIR + 'Mapas/%s/'%MAPA
    Context=L_Context.Load_Context(wdir, NeighborsBehaviour)
    L_Syn_Img.Set_Margins(Context)
    print("Directorio: ", csv_dir + "" + MAPA + "_distances.pickle")    
    distances = L_Solvers.load_distances(csv_dir + "" + MAPA + "_distances.pickle")
    Observation, tita=L_ReadObs.Load_PKL_or_Compute_Kernels(HDFfilename,Context,Band,GC=True)
    km = MAPA[4:6]
    L_Solvers.preComp_BG(output_dir + "/BGC", MAPA, Band, distances, Context, Observation, grilla_gammas,w=.001,error_std=1.0,MaxObs=dic_maxObs[Band+km])         

def preComp_BG(path, mapStr, bandStr, distances,  Context, Observation, gammas,w=.001,error_std=1.0,MaxObs=10):
    for i in range(len(gammas)):
        print("Precomputando para ", i)
        gamma = gammas[i]
        A = L_Solvers.preComp_BG_Step(distances, Context, Observation, gamma,w,error_std,MaxObs)
        filePath = path + "RI_" + mapStr + "_" + bandStr + ".pickle"
        with open(filePath, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
           pickle.dump(A, f, pickle.HIGHEST_PROTOCOL)


def compute_and_save_coefficients_give_parameters(MAPA, Band,gamma=0.1, w=0.001):
    wdir=BASE_DIR + 'Mapas/%s/'%MAPA
    Context=L_Context.Load_Context(wdir, NeighborsBehaviour)
    L_Syn_Img.Set_Margins(Context)    
    distances = L_Solvers.load_distances(wdir + "distance")
    Observation, tita=L_ReadObs.Load_PKL_or_Compute_Kernels(HDFfilename,Context,Band,GC=True)
    km = MAPA[4:6]
    preComp_BG(csv_dir, MAPA, Band, distances, Context, Observation, [gamma],w=.001,error_std=1.0,MaxObs=dic_maxObs[Band+km])         
#compute_and_save_distances('LCA_50000m')
#compute_and_save_distances('LCA_25000m')
#compute_and_save_distances('LCA_12500m')

#compute_and_save_distances("LCA_25000m")
#compute_and_save_coefficients_give_parameters('LCA_25000m', 'C')
#compute_and_save_coefficients_give_parameters('LCA_25000m', 'X')
#compute_and_save_coefficients_give_parameters('LCA_25000m', 'K')
#compute_and_save_coefficients_give_parameters('LCA_25000m', 'KA')
#compute_and_save_coefficients_give_parameters('LCA_25000m', 'KU')




