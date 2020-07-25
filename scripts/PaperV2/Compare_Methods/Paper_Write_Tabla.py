# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 09:45:54 2018

@author: rgrimson
"""

import my_base_dir
BASE_DIR = my_base_dir.get()
csv_dir=BASE_DIR + 'Output/'

#Import libraries
import numpy as np
import pandas as pd
#%%
# INPUTS
csv_name ='corrida_todos_tablaX10x10'
Bands=['KA','KU','K','X','C']
Methods = ['EM','BGF','LSQR']

                      
df=pd.read_csv(csv_dir + csv_name + '.csv')

dg=pd.DataFrame({'RMSE-mu':df.groupby(['CellSize', 'Method', 'Pol','Band']).mean()['RMSE'],'RMSE-std':df.groupby(['CellSize', 'Method', 'Pol','Band']).std()['RMSE'],'Rho-m':df.groupby(['CellSize', 'Method', 'Pol', 'Band']).mean()['CoefCorr']})
            

#%%            

for Pol in ['H','V']:
  print("Polarization: ", Pol)
  for Band in Bands:
    #print("Band: ", Band)
      CellSize=12
      #print("[",CellSize,"]",end='\n')
      for Method in Methods:
        #print("(",Method[0],")",end='')
        RMSE_m = dg.loc[CellSize, Method, Pol, Band]['RMSE-mu']
        RMSE_s = dg.loc[CellSize, Method, Pol, Band]['RMSE-std']
        Rho_m  = dg.loc[CellSize, Method, Pol, Band]['Rho-m']
        print("%.2f (%.2f) & %.2f &" %(RMSE_m,RMSE_s, Rho_m),end='')
      print("\\%s & "%Band.capitalize(), end='')

      CellSize=25
      for Method in Methods:
        #print("(",Method[0],")",end='')
        RMSE_m = dg.loc[CellSize, Method, Pol, Band]['RMSE-mu']
        RMSE_s = dg.loc[CellSize, Method, Pol, Band]['RMSE-std']
        Rho_m  = dg.loc[CellSize, Method, Pol, Band]['Rho-m']
        print("%.2f (%.2f) & %.2f &" %(RMSE_m,RMSE_s, Rho_m),end='')
      print("\\\\")

#%%
for Pol in ['H','V']:
  print("Polarization: ", Pol)
  for Band in Bands:
      CellSize=50
      print("Band %s: "%Band.capitalize(),end=' ')
      for Method in Methods:
        RMSE_m = dg.loc[CellSize, Method, Pol, Band]['RMSE-mu']
        RMSE_s = dg.loc[CellSize, Method, Pol, Band]['RMSE-std']
        Rho_m  = dg.loc[CellSize, Method, Pol, Band]['Rho-m']
        print("%.2f (%.2f) & %.2f & " %(RMSE_m,RMSE_s, Rho_m),end='')
      print("")

  