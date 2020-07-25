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
csv_name ='corrida_todos_tablaX10x10'

#%%
MAPAS=['LCA_12500m','LCA_25000m','LCA_50000m']
MAPAS=['LCA_50000m','LCA_25000m','LCA_12500m']
#MAPAS=['LCA_50000m']

Methods = ['LSQR','Weights','EM','Rodgers_IT','Global_GCV_Tichonov','BGF']
#Methods = ['Rodgers_IT']


Bands=['KA','KU','K','X','C']
#Bands=['KA']
tot_img_samples=10 #CANTIDAD DE IMAGENES SINTETICAS GENERADAS
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
columnas = ['CellSize','Method','Band','Pol','Img_num','Obs_num','Obs_std_error','RMSE','CoefCorr','time','Mu_real_0','Mu_rec_0','Sigma_real_0','Sigma_rec_0','Mu_real_1','Mu_rec_1','Sigma_real_1','Sigma_rec_1'] #el 0 y el 1 correspondonen al LT
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
                        if Method == 'BGF':
                            BG_Coef= dic_BG_coef[MAPA][Band]
                            Sol=[BG_Coef.dot(Observation['Tb']['H']), BG_Coef.dot(Observation['Tb']['V'])]
                        else:
                            Sol=L_Solvers.Solve(Context,Observation,Method, tita, obsstd=Obs_std)
            
                        t_final = time.time()
                        t = t_final - t_inicial
                        
                        L_ParamEst.pr(titaReal,'Synthetic Image')
                        tita=L_ParamEst.recompute_param(Sol,Observation,tita)
                        L_ParamEst.pr(tita, 'Reconstructed Image')
                
                        dic = {'CellSize':km,'Method': Method,'Band':Band,'Pol': 'H','Img_num': n_img,'Obs_num': n_obs,'Obs_std_error':Obs_std,'RMSE':str(L_Syn_Img.COMP_COND(TbH,Sol[0],Context,Margin=False)[0]),'CoefCorr':str(L_Syn_Img.COMP_COND(TbH,Sol[0],Context,Margin=False)[1]),'time':t,'Mu_real_0':titaReal['H']['mu'][0],'Sigma_real_0':np.sqrt(titaReal['H']['sigma2'][0]),'Mu_real_1':titaReal['H']['mu'][1],'Sigma_real_1':np.sqrt(titaReal['H']['sigma2'][1]),'Mu_rec_0':tita['H']['mu'][0],'Sigma_rec_0':np.sqrt(tita['H']['sigma2'][0]),'Mu_rec_1':tita['H']['mu'][1],'Sigma_rec_1':np.sqrt(tita['H']['sigma2'][1])}
                        df = df.append(dic, ignore_index=True)
                        
                        dic = {'CellSize':km,'Method': Method,'Band':Band,'Pol': 'V','Img_num': n_img,'Obs_num': n_obs,'Obs_std_error':Obs_std,'RMSE':str(L_Syn_Img.COMP_COND(TbV,Sol[1],Context,Margin=False)[0]),'CoefCorr':str(L_Syn_Img.COMP_COND(TbV,Sol[1],Context,Margin=False)[1]),'time':t,'Mu_real_0':titaReal['V']['mu'][0],'Sigma_real_0':np.sqrt(titaReal['V']['sigma2'][0]),'Mu_real_1':titaReal['V']['mu'][1],'Sigma_real_1':np.sqrt(titaReal['V']['sigma2'][1]),'Mu_rec_0':tita['V']['mu'][0],'Sigma_rec_0':np.sqrt(tita['V']['sigma2'][0]),'Mu_rec_1':tita['V']['mu'][1],'Sigma_rec_1':np.sqrt(tita['V']['sigma2'][1])}
                        df = df.append(dic, ignore_index=True)
        
                        print('termino:',km,Band,Method,Obs_std,n_img,n_obs)
                        if export_solutions:
                            L_Output.Export_Solution(Sol, Context, Band, 'S_'+str(km)+'_'+Band+'_'+Method+str(n_img)+'_'+str(n_obs)+'_'+str(Obs_std).replace('.','_'))

                        
                        
   


df.to_csv(csv_dir + csv_name + '.csv')
            

#%%            
df50=df[df['CellSize']==50]
df25=df[df['CellSize']==25]
df12=df[df['CellSize']==12]

df50H=df50[df50['Pol']=='H'][['Method','RMSE']]
df50V=df50[df50['Pol']=='V'][['Method','RMSE']]
df25H=df25[df25['Pol']=='H'][['Method','RMSE']]
df25V=df25[df25['Pol']=='V'][['Method','RMSE']]
df12H=df12[df12['Pol']=='H'][['Method','RMSE']]
df12V=df12[df12['Pol']=='V'][['Method','RMSE']]

Methods= set(df['Method'])

#%%
df=pd.read_csv(csv_dir + csv_name + '.csv')
M=np.array(df['RMSE'].values,dtype=float).reshape([3,3,5,10,5,2])

#MM=M.mean(axis=3).mean(axis=1) ##img and obs means
RAND=M[:,:,:,:,:,0] # me quedo solo con la pol H
TRIG=M[:,:,:,:,:,1] # me quedo con la Pol V

RAND_mean=RAND.mean(axis=3).mean(axis=1) #promedio las 10 observaciones y las 3 imgs
TRIG_mean=TRIG.mean(axis=3).mean(axis=1) #promedio las 30 observaciones

RAND_mean_50KA=RAND_mean[0,0,:] #selecciono mapa y banda
RAND_mean_50KU=RAND_mean[0,1,:] #selecciono mapa y banda
RAND_mean_50K =RAND_mean[0,2,:] #selecciono mapa y banda
RAND_mean_50X =RAND_mean[0,3,:] #selecciono mapa y banda
RAND_mean_50C =RAND_mean[0,4,:] #selecciono mapa y banda

TRIG_mean_50KA=TRIG_mean[0,0,:] #selecciono mapa y banda
TRIG_mean_50KU=TRIG_mean[0,1,:] #selecciono mapa y banda
TRIG_mean_50K =TRIG_mean[0,2,:] #selecciono mapa y banda
TRIG_mean_50X =TRIG_mean[0,3,:] #selecciono mapa y banda
TRIG_mean_50C =TRIG_mean[0,4,:] #selecciono mapa y banda


RAND_mean_25KA=RAND_mean[1,0,:] #selecciono mapa y banda
RAND_mean_25KU=RAND_mean[1,1,:] #selecciono mapa y banda
RAND_mean_25K =RAND_mean[1,2,:] #selecciono mapa y banda
RAND_mean_25X =RAND_mean[1,3,:] #selecciono mapa y banda
RAND_mean_25C =RAND_mean[1,4,:] #selecciono mapa y banda

TRIG_mean_25KA=TRIG_mean[1,0,:] #selecciono mapa y banda
TRIG_mean_25KU=TRIG_mean[1,1,:] #selecciono mapa y banda
TRIG_mean_25K =TRIG_mean[1,2,:] #selecciono mapa y banda
TRIG_mean_25X =TRIG_mean[1,3,:] #selecciono mapa y banda
TRIG_mean_25C =TRIG_mean[1,4,:] #selecciono mapa y banda


RAND_mean_12KA=RAND_mean[2,0,:] #selecciono mapa y banda
RAND_mean_12KU=RAND_mean[2,1,:] #selecciono mapa y banda
RAND_mean_12K =RAND_mean[2,2,:] #selecciono mapa y banda
RAND_mean_21X =RAND_mean[2,3,:] #selecciono mapa y banda
RAND_mean_12C =RAND_mean[2,4,:] #selecciono mapa y banda

TRIG_mean_12KA=TRIG_mean[2,0,:] #selecciono mapa y banda
TRIG_mean_12KU=TRIG_mean[2,1,:] #selecciono mapa y banda
TRIG_mean_12K =TRIG_mean[2,2,:] #selecciono mapa y banda
TRIG_mean_12X =TRIG_mean[2,3,:] #selecciono mapa y banda
TRIG_mean_12C =TRIG_mean[2,4,:] #selecciono mapa y banda


#%%
import matplotlib.pyplot as plt

plt.figure(1)
#%%
plt.clf()
plt.plot(RAND_mean[2,:,:])
plt.ylim(ymax=20)
plt.legend(Methods,loc='upper left', shadow=True)
plt.title("12.5km RAND.")
#%%
plt.figure()
plt.plot(RAND_mean[1,:,:])
plt.ylim(ymax=20)
plt.legend(Methods,loc='upper left', shadow=True)
plt.title("25km RAND.")
#%%
plt.figure()
plt.plot(RAND_mean[0,:,:])
plt.ylim(ymax=5)
plt.legend(Methods,loc='upper left', shadow=True)
plt.title("50km RAND.")

#%%
plt.figure()
plt.plot(TRIG_mean[2,:,:])
plt.ylim(ymax=20)
plt.legend(Methods,loc='upper left', shadow=True)
plt.title("12.5km TRIG.")
#%%
plt.figure()
plt.plot(TRIG_mean[1,:,:])
plt.ylim(ymax=20)
plt.legend(Methods,loc='upper left', shadow=True)
plt.title("25km TRIG.")
#%%
plt.figure()
plt.plot(TRIG_mean[0,:,:])
plt.ylim(ymax=5)
plt.legend(Methods,loc='upper left', shadow=True)
plt.title("50km TRIG.")

