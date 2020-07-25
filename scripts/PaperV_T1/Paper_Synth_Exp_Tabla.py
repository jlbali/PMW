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
import L_ParamEst
import L_Syn_Img
import numpy as np
import pandas as pd
import time
import pickle
#%%
# INPUTS
HDFfilename='GW1AM2_201207310439_227D_L1SGBTBR_2210210'
NeighborsBehaviour = L_Context.NO_NEIGHBORS
csv_name ='Metricas_corrida_todos_tablaX10x10'

#%%
MAPAS=['LCA_12500m','LCA_25000m','LCA_50000m']
MAPAS=['LCA_50000m','LCA_25000m','LCA_12500m']
#MAPAS=['LCA_25000m']

Methods = ['LSQR','Weights','EM','Rodgers_IT','Global_GCV_Tichonov','BGF']
Methods = ['EM','BGF','LSQR']


Bands=['KA','KU','K','X','C']
#Bands=['KA']
tot_img_samples=10 #CANTIDAD DE IMAGENES SINTETICAS GENERADAS
tot_img_samples=1
tot_obs_samples=10 #CANTIDAD DE OBS SINTETICAS GENERADAS por IMG
#Obs_errors = [0.25, 0.5,1.0, 2.0,4.0]
Obs_errors = [1.0]
export_solutions = False
#%%

# Levantamos el diccionario de coeficientes de Backus-Gilbert..
dic_BG_coef = {}
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
columnas = ['CellSize','Method','Band','Pol','Img_num','Obs_num','Obs_std_error','RMSE','RMSE_L0','RMSE_C0','RMSE_L1','RMSE_C1','CoefCorr','time','Mu_real_0','Mu_rec_0','Sigma_real_0','Sigma_rec_0','Mu_real_1','Mu_rec_1','Sigma_real_1','Sigma_rec_1'] #el 0 y el 1 correspondonen al LT
df = pd.DataFrame(columns=columnas)

for MAPA in MAPAS:
    wdir=BASE_DIR + 'Mapas/%s/'%MAPA
    Context=L_Context.Load_Context(wdir, NeighborsBehaviour)
    L_Syn_Img.Set_Margins(Context)
    km=int(MAPA[4:6])

    for n_img in range(tot_img_samples):
        TbH=L_Syn_Img.Create_Random_Synth_Img(Context)
        TbV = L_Syn_Img.Create_Trig_Synth_Img(Context)
        titaReal=copy.deepcopy(L_ParamEst.Compute_param([TbH,TbV],Context))
        
    
        for Band in Bands:
            Observation, tita=L_ReadObs.Load_PKL_or_Compute_Kernels(HDFfilename,Context,Band,GC=True)
    
    
            for Obs_std in Obs_errors:
                
              for n_obs in range(tot_obs_samples):
                Observation=L_Syn_Img.Simulate_PMW(TbH,TbV,Context,Observation,Obs_error_std=Obs_std)
    
    
                for Method in Methods:
    
    
                        t_inicial = time.time()                
                        #corrergir esto!!!
                        if Method == 'BGF':I RMSE
                            BG_Coef= dic_BG_coef[MAPA][Band]
                            Sol=[BG_Coef.dot(Observation['Tb']['H']), BG_Coef.dot(Observation['Tb']['V'])]
                        else:
                            Sol=L_Solvers.Solve(Context,Observation,Method, tita, obsstd=Obs_std)
            
                        t_final = time.time()
                        t = t_final - t_inicial
                        
                        L_ParamEst.pr(titaReal,'Synthetic Image')
                        tita=L_ParamEst.recompute_param(Sol,Observation,tita)
                        L_ParamEst.pr(tita, 'Reconstructed Image')
                
                        dic = {'CellSize':km,'Method': Method,'Band':Band,'Pol': 'H','Img_num': n_img,'Obs_num': n_obs,'Obs_std_error':Obs_std,
                        'RMSE_L0':(L_Syn_Img.COMP_COND(TbH,Sol[0],Context,Margin=False,Regular=True,LT=0)[0]),
                        'RMSE_C0':(L_Syn_Img.COMP_COND(TbH,Sol[0],Context,Margin=False,Regular=False,LT=0)[0]),
                        'RMSE_L1':(L_Syn_Img.COMP_COND(TbH,Sol[0],Context,Margin=False,Regular=True,LT=1)[0]),
                        'RMSE_C1':(L_Syn_Img.COMP_COND(TbH,Sol[0],Context,Margin=False,Regular=False,LT=1)[0]),
                        'RMSE':(L_Syn_Img.COMP_COND(TbH,Sol[0],Context,Margin=False)[0]),
                        'CoefCorr':(L_Syn_Img.COMP_COND(TbH,Sol[0],Context,Margin=False)[1]),
                        'time':t,'Mu_real_0':titaReal['H']['mu'][0],'Sigma_real_0':np.sqrt(titaReal['H']['sigma2'][0]),'Mu_real_1':titaReal['H']['mu'][1],'Sigma_real_1':np.sqrt(titaReal['H']['sigma2'][1]),'Mu_rec_0':tita['H']['mu'][0],'Sigma_rec_0':np.sqrt(tita['H']['sigma2'][0]),'Mu_rec_1':tita['H']['mu'][1],'Sigma_rec_1':np.sqrt(tita['H']['sigma2'][1])}
                        df = df.append(dic, ignore_index=True)
                        
                        dic = {'CellSize':km,'Method': Method,'Band':Band,'Pol': 'V','Img_num': n_img,'Obs_num': n_obs,'Obs_std_error':Obs_std,
                        'RMSE_L0':(L_Syn_Img.COMP_COND(TbV,Sol[1],Context,Margin=False,Regular=True,LT=0)[0]),
                        'RMSE_C0':(L_Syn_Img.COMP_COND(TbV,Sol[1],Context,Margin=False,Regular=False,LT=0)[0]),
                        'RMSE_L1':(L_Syn_Img.COMP_COND(TbV,Sol[1],Context,Margin=False,Regular=True,LT=1)[0]),
                        'RMSE_C1':(L_Syn_Img.COMP_COND(TbV,Sol[1],Context,Margin=False,Regular=False,LT=1)[0]),
                        'RMSE':(L_Syn_Img.COMP_COND(TbV,Sol[1],Context,Margin=False)[0]),
                        'CoefCorr':(L_Syn_Img.COMP_COND(TbV,Sol[1],Context,Margin=False)[1]),'time':t,'Mu_real_0':titaReal['V']['mu'][0],'Sigma_real_0':np.sqrt(titaReal['V']['sigma2'][0]),'Mu_real_1':titaReal['V']['mu'][1],'Sigma_real_1':np.sqrt(titaReal['V']['sigma2'][1]),'Mu_rec_0':tita['V']['mu'][0],'Sigma_rec_0':np.sqrt(tita['V']['sigma2'][0]),'Mu_rec_1':tita['V']['mu'][1],'Sigma_rec_1':np.sqrt(tita['V']['sigma2'][1])}
                        df = df.append(dic, ignore_index=True)
        
                        print('termino:',km,Band,Method,Obs_std,n_img,n_obs)
                        if export_solutions:
                            L_Output.Export_Solution(Sol, Context, Band, 'S_'+str(km)+'_'+Band+'_'+Method+str(n_img)+'_'+str(n_obs)+'_'+str(Obs_std).replace('.','_'))

                        
                        
   


df.to_csv(csv_dir + csv_name + '.csv')
            

dg=df.copy()
#dg.groupby(['Method','Band','Pol']).mean()['RMSE']
dg.groupby(['CellSize', 'Band','Pol', 'Method']).mean()[['RMSE','RMSE_C0','RMSE_L0','RMSE_C1','RMSE_L1']]

#%%
L_Output.Export_Solution(Sol, Context, Band, 'TestRafa')
SolH=Sol[0]
SolV=Sol[1]

Reg=np.array(Context['Dict_Vars']['SqCell'])
IReg=np.array([not r for r in Context['Dict_Vars']['SqCell']]) #irregular cell, frontier of LTs.
Mar=np.array(Context['Dict_Vars']['Margin']) #a marginal cell, border of the image
NMar=np.array([not m for m in Context['Dict_Vars']['Margin']]) #interior cell

SolN=SolH.copy()
SolN[Reg]=0
L_Output.Export_Solution([SolN,SolN], Context, Band, 'TestRafa_Costa')

SolN=SolH.copy()
SolN[IReg]=0
L_Output.Export_Solution([SolN,SolN], Context, Band, 'TestRafa_Reg')

SolN=SolH.copy()
SolN[Mar]=0
L_Output.Export_Solution([SolN,SolN], Context, Band, 'TestRafa_NMar')

SolN=SolH.copy()
SolN[NMar]=0
L_Output.Export_Solution([SolN,SolN], Context, Band, 'TestRafa_Mar')

SolN=SolH.copy()
SolN[Mar]=0
SolN[Reg]=0
L_Output.Export_Solution([SolN,SolN], Context, Band, 'TestRafa_CostaInterior')

SolN=SolH.copy()
SolN[Mar]=0
SolN[IReg]=0
L_Output.Export_Solution([SolN,SolN], Context, Band, 'TestRafa_LandInterior')

#%%


Reg=np.array(Context['Dict_Vars']['SqCell'])
IReg=np.array([not r for r in Context['Dict_Vars']['SqCell']]) #irregular cell, frontier of LTs.
Mar=np.array(Context['Dict_Vars']['Margin']) #a marginal cell, border of the image
NMar=np.array([not m for m in Context['Dict_Vars']['Margin']]) #interior cell
VType=Context['Dict_Vars']['VType']
L0=np.array(VType==0)
L1=np.array(VType==1)

B0=(NMar*IReg*L0)
B1=(NMar*IReg*L1)
I0=(NMar*Reg*L0)
I1=(NMar*Reg*L1)

print(B0.sum())
print(B1.sum())
print(I0.sum())
print(I1.sum())

#%%
def Metricas(Orig, Sol, Context, Margin=None, Regular=None, LT=None):
    n_Vars=Context['Dict_Vars']['n_Vars']    
    T=np.ones(n_Vars,dtype=bool)
    if (Margin==None):
        M=T
    elif Margin:
        M=np.array(Context['Dict_Vars']['Margin']) #a marginal cell, border of the image
    else:
        M=np.array([not m for m in Context['Dict_Vars']['Margin']]) #interior cell
    if (Regular==None):
        R=T
    elif Regular:
        R=np.array(Context['Dict_Vars']['SqCell']) #a square, regular cell, typically not land type frontier.
    else:
        R=np.array([not r for r in Context['Dict_Vars']['SqCell']]) #irregular cell, frontier of LTs.
    if (LT==None):
        L=T
    else:
        VType=Context['Dict_Vars']['VType']
        L=np.array(VType==LT)
    Cells=np.where(M*R*L)[0]
    
    return RMSE(Orig[Cells], Sol[Cells]),CC(Orig[Cells], Sol[Cells])
