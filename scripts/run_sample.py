#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 12:04:28 2020

@author: lbali
"""

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


# Band configuration.
Band = 'KU'


BASE_DIR="/home/lbali/proyectos/Gits/UNSAM/PMW/"
csv_dir=BASE_DIR + 'outputs/'


# INPUTS
HDFfilename='GW1AM2_201207310439_227D_L1SGBTBR_2210210'
NeighborsBehaviour = L_Context.COMPUTE_IF_NECESSARY
Method = "GCV_Tichonov"
MAP = "LCA_25000m"


if __name__ == "__main__":
    
    wdir=BASE_DIR + 'maps/%s/'%MAP
    Context=L_Context.Load_Context(wdir, NeighborsBehaviour)
    
    VType = Context['Dict_Vars']["VType"]
    neighbors = Context["Neighbors"]
    extras ={ 
        "neighbors": neighbors,
        "obsstd": 1.0,        
    }
    
    L_Syn_Img.Set_Margins(Context)
    
    Observations = {}
    titas = {}
    v_errs = {}
    
    Observation, tita=L_ReadObs.Load_PKL_or_Compute_Kernels(HDFfilename,Context,Band,GC=True)
    Observations[Band] = Observation
    titas[Band] = tita
    K=Observation['Wt']
    n_cells = K.shape[1] # No cambia con la banda.
    n_ell = K.shape[0]
    v_errs[Band] = {}
    
    #print("Tb", Observation["Tb"])
    TbH = Observation["Tb"]["H"]
    TbV = Observation["Tb"]["V"]
    SolH=L_NewSolvers.Solve(VType, K, TbH, Method, extras)
    print(SolH)
    SolV=L_NewSolvers.Solve(VType, K, TbV, Method, extras)
    Sol = [SolH, SolV]
    L_Output.Export_Solution(Sol, Context, Band, 'Solution' )

    

    