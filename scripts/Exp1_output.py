
#%%
Band = 'K'

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 14:40:00 2019

@author: rgrimson
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



BASE_DIR="../"
csv_dir=BASE_DIR + 'outputs/'


#csv_name ='simple_Trigonometric_sigmas'
#csv_name = 'simple_Trigonometric_sigmas_mixRatio05'
#csv_name ='simple_Trigonometric_sigmas3'
#csv_name ='simple_Trigonometric_sigmas3_MR05'
#csv_name ='simple_Trigonometric_sigmas3_MR05_2x2'

#csv_name ='simple_Trigonometric_sigmas3_MR05_KU'
csv_name ='exp1_' + Band




#Bands=['KA','KU','K','X','C']
#Bands=['K']

#mix_ratio = 1.0


#Obs_errors = [1e-3, 1e-2, 1e-1, 0.25, 0.3, 0.4, 0.5, 0.6,0.7, 0.8, 0.9, 1.0, 1.1, 1.2,1.3,1.4,1.5,1.7,1.85, 2.0, 5.0, 10.0, 20.0]
#Obs_errors = np.linspace(0.1, 2.0, 20)


MAPAS=['LCA_25000m'] # Solo para trigonometric.

base_type = "EXP1" # Usar mapas LCA para esto.

df = pd.read_csv(csv_dir + csv_name + ".csv")
#dg.groupby(['Method','Band','Pol']).mean()['RMSE']
#dg = dg.groupby(['CellSize', 'Band', 'Method', 'mix_ratio']).mean()[['RMSE','RMSE_C0','RMSE_L0','RMSE_C1','RMSE_L1']]
dg = df.groupby(['CellSize', 'Band', 'Method', 'mix_ratio', 'Obs_std_error'])['RMSE'].mean()
Obs_errors = list(set(df["Obs_std_error"]))
Obs_errors.sort()
mix_ratios = list(set(df["mix_ratio"]))
mix_ratios.sort()

Bands = list(set(df["Band"]))
Methods = list(set(df["Method"]))

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
        serie_base = []
        sigmas = []
        for Obs_std in Obs_errors:
            #if Obs_std < 0.1 or Obs_std > 2.0:
                #continue
            #Obs_std = 1.0
            i = 0
            for Method in Methods:
#                if Method == "LSQR" and Band == "C":
#                    continue
                serie = []
                for mix_ratio in mix_ratios:
                    serie.append(dg[km, Band, Method, mix_ratio, Obs_std])
                if Method == "GCV_Tichonov":
                    Method = "Tychonov"
                if Band == "C":
                    plt.ylim(0,6)
                ax.set_ylabel("RMSE", fontsize=14)
                ax.set_xlabel("$\\alpha$", fontsize=20)
                ax.plot(mix_ratios, serie, label=Method, color="black", linestyle=linestyles[i])
                i = i + 1
        legend = ax.legend(loc='best', shadow=True, fontsize='x-large') 
        plt.title("Band: " + Band)
        plt.savefig(csv_dir + "Imgs/" + Band + "_" + base_type + ".eps")
        #plt.show()
