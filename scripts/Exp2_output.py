#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 14:40:00 2019

@author: rgrimson
"""

Band = "K"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



BASE_DIR="/home/lbali/proyectos/Gits/PMW-Tychonov/"
csv_dir=BASE_DIR + 'Output/'


#csv_name ='simple_Trigonometric_sigmas'
#csv_name = 'simple_Trigonometric_sigmas_mixRatio05'
#csv_name ='simple_Trigonometric_sigmas3'
#csv_name ='simple_Trigonometric_sigmas3_MR05'
#csv_name ='simple_Trigonometric_sigmas3_MR05_2x2'

csv_name ='exp2_' + Band

#Bands=['KA','KU','K','X','C']
#Bands=['K']

#mix_ratio = 1.0


#Obs_errors = [1e-3, 1e-2, 1e-1, 0.25, 0.3, 0.4, 0.5, 0.6,0.7, 0.8, 0.9, 1.0, 1.1, 1.2,1.3,1.4,1.5,1.7,1.85, 2.0, 5.0, 10.0, 20.0]
#Obs_errors = np.linspace(0.1, 2.0, 20)


MAPAS=['LCA_25000m'] # Solo para trigonometric.

base_type = "EXP2" # Usar mapas LCA para esto.

df = pd.read_csv(csv_dir + csv_name + ".csv")
#dg.groupby(['Method','Band','Pol']).mean()['RMSE']
#dg = dg.groupby(['CellSize', 'Band', 'Method', 'mix_ratio']).mean()[['RMSE','RMSE_C0','RMSE_L0','RMSE_C1','RMSE_L1']]
dg = df.groupby(['CellSize', 'Band', 'Method', 'mix_ratio', 'Obs_std_error'])['RMSE'].mean()
Obs_errors = list(set(df["Obs_std_error"]))
Obs_errors.sort()
mix_ratio = list(set(df["mix_ratio"]))[0]

Bands = list(set(df["Band"]))

print(Obs_errors)

linestyles = ['-', '--', '-.', ':']
for MAPA in MAPAS:
    km=int(MAPA[4:6])
    print("Mapa: " + MAPA)
    for Band in Bands:
        print("Banda: " + Band)
        fig, ax = plt.subplots()
        #plt.ylim((0,10))
        serie1 = []
        serie2 = []
        serie3 = []
        serie_base = []
        sigmas = []
        for Obs_std in Obs_errors:
            #if Obs_std < 0.1 or Obs_std > 2.0:
                #continue
            #Obs_std = 1.0
            err_LSQR = dg[km, Band, "LSQR", mix_ratio, Obs_std]
            err_EM = dg[km, Band, "EM", mix_ratio, Obs_std]
            err_EM_Adj = dg[km, Band, "EM_Adapt", mix_ratio, Obs_std]
            err_Tych = dg[km, Band, "GCV_Tichonov", mix_ratio, Obs_std]
            serie1.append(err_EM)
            #serie2.append(err_EM_Adj)
            serie3.append(err_LSQR )
            serie_base.append(err_Tych)
            sigmas.append(Obs_std)
        if Band == "C":
            plt.ylim(0,6)
        ax.set_ylabel("RMSE", fontsize=14)
        ax.set_xlabel("$\\sigma_{obs}$", fontsize=20)
        ax.plot(Obs_errors, serie_base, label="Tychonov", color="black", linestyle=linestyles[0])
        ax.plot(Obs_errors, serie1, label="EM", color="black", linestyle=linestyles[1])
        ax.plot(Obs_errors, serie3, label="LSQR", color="black", linestyle=linestyles[2])
        legend = ax.legend(loc='best', shadow=True, fontsize='x-large') 
        plt.title("Banda: " + Band)
        plt.savefig(csv_dir + "Imgs/" + Band + "_" + base_type + ".eps")
        plt.show()
