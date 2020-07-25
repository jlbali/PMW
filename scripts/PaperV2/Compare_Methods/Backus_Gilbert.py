# -*- coding: utf-8 -*-



import my_base_dir
BASE_DIR = my_base_dir.get()
output_dir=BASE_DIR + 'Output/'

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

CANT_GAMMAS = 100

dic_maxObs={'KA12':6,'KU12':6,'K12':6,'X12':4,'C12':4,'KA25':16,'KU25':16,'K25':16,'X25':10,'C25':10,'KA50':32,'KU50':30,'K50':30,'X50':16,'C50':16}

# INPUTS
HDFfilename='GW1AM2_201207310439_227D_L1SGBTBR_2210210'
NeighborsBehaviour = L_Context.COMPUTE_IF_NECESSARY


MAPAS=['LCA_50000m','LCA_25000m','LCA_12500m']
Bands=['KA','KU', 'K', 'X', 'C']

def compute_and_save_distances(MAPA):
    wdir=BASE_DIR + 'Mapas/%s/'%MAPA
    Context=L_Context.Load_Context(wdir, NeighborsBehaviour)
    L_Syn_Img.Set_Margins(Context)    
    distances = L_Solvers.precompute_all_optimized(Context)
    L_Solvers.save_distances(output_dir + "/"+MAPA + "_distances.pickle", distances)

def compute_and_save_coefficients(MAPA, Band):
    #grilla_gammas = np.linspace(start = 0.0001, stop= np.pi/2.0, num=CANT_GAMMAS )
    grilla_gammas = L_Solvers.grillaLogaritmica(0.0001, np.pi/2, CANT_GAMMAS) 
    wdir=BASE_DIR + 'Mapas/%s/'%MAPA
    Context=L_Context.Load_Context(wdir, NeighborsBehaviour)
    L_Syn_Img.Set_Margins(Context)    
    distances = L_Solvers.load_distances(output_dir + "/"+MAPA + "_distances.pickle")
    Observation, tita=L_ReadObs.Load_PKL_or_Compute_Kernels(HDFfilename,Context,Band,GC=True)
    km = MAPA[4:6]
    L_Solvers.preComp_BG(output_dir + "/BGC", MAPA, Band, distances, Context, Observation, grilla_gammas,w=.001,error_std=1.0,MaxObs=dic_maxObs[Band+km])         

#compute_and_save_distances('LCA_50000m')
#compute_and_save_distances('LCA_25000m')
#compute_and_save_distances('LCA_12500m')

compute_and_save_coefficients('LCA_50000m', 'C')

