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


def Coef_correlacion(img_sintetica,img_solucion): # Toma dos imagenes, cada imagen es un array
    matriz_correlacion = np.corrcoef(img_sintetica,img_solucion)
    return matriz_correlacion[0,1]

#%%
# INPUTS
HDFfilename='GW1AM2_201207310439_227D_L1SGBTBR_2210210'
NeighborsBehaviour = L_Context.NO_NEIGHBORS
csv_name ='corrida_noise'

#%%
MAPAS=['LCA_12500m','LCA_25000m','LCA_50000m']
MAPAS=['LCA_50000m','LCA_25000m','LCA_12500m']
MAPAS=['LCA_25000m']

#Methods = ['EM']
#Methods = ['LSQR']
Methods = ['BGF']
#Methods = ['Rodgers_IT']


Bands=['KA','KU','K','X','C']
#Bands=['KA']
tot_img_samples=1 #CANTIDAD DE IMAGENES SINTETICAS GENERADAS
tot_obs_samples=1 #CANTIDAD DE OBS SINTETICAS GENERADAS por IMG
#Obs_errors = [0.25, 0.5,1.0, 2.0,4.0]

#Obs_errors = [np.exp(l) for l in np.linspace(np.log(.1),np.log(10),9)]
Obs_errors = np.arange(1.0,20.0, step=1.0)
export_solutions = False
#%%

# Levantamos el diccionario de coeficientes de Backus-Gilbert..
dic_BG_distances = {}
for MAPA in MAPAS:
    filepath = csv_dir  + MAPA + "_distances.pickle"
    with open(filepath, 'rb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            data = pickle.load(f)
            dic_BG_distances[MAPA] = data


np.random.seed(1)
columnas = ['CellSize','Method','Band','Pol','Img_num','Obs_num','Obs_std_error','RMSE','CoefCorr','time','Mu_real_0','Mu_rec_0','Sigma_real_0','Sigma_rec_0','Mu_real_1','Mu_rec_1','Sigma_real_1','Sigma_rec_1'] #el 0 y el 1 correspondonen al LT
df = pd.DataFrame(columns=columnas)

Res={}

# Para Backus-Gilbert.
gamma = 0.1
w = 0.001
dic_maxObs={'KA12':6,'KU12':6,'K12':6,'X12':4,'C12':4,'KA25':16,'KU25':16,'K25':16,'X25':10,'C25':10,'KA50':32,'KU50':30,'K50':30,'X50':16,'C50':16}



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


            for Method in Methods:

                RMSE=[]
                for Obs_std in Obs_errors:
                    
                      if Method == 'BGF':
                            km = MAPA[4:6]
                            MaxObs = dic_maxObs[Band + km]
                            A = L_Solvers.preComp_BG_Step(dic_BG_distances[MAPA], Context, Observation, gamma,w,Obs_std,MaxObs)
                      for n_obs in range(tot_obs_samples):
                            Observation=L_Syn_Img.Simulate_PMW(TbH,TbV,Context,Observation,Obs_error_std=Obs_std)
                            t_inicial = time.time()                
                            #corrergir esto!!!
                            if Method == 'BGF':
                                Sol=[A.dot(Observation['Tb']['H']), A.dot(Observation['Tb']['V'])]
                            else:
                                Sol=L_Solvers.Solve(Context,Observation,Method, tita, obsstd=Obs_std)
                
                            t_final = time.time()
                            t = t_final - t_inicial
                            
                            L_ParamEst.pr(titaReal,'Synthetic Image')
                            tita=L_ParamEst.recompute_param(Sol,Observation,tita)
                            L_ParamEst.pr(tita, 'Reconstructed Image')
                    
                            dic = {'CellSize':km,'Method': Method,'Band':Band,'Pol': 'H','Img_num': n_img,'Obs_num': n_obs,'Obs_std_error':Obs_std,'RMSE':str(L_Syn_Img.COMP_COND(TbH,Sol[0],Context,Margin=False)[0]),'CoefCorr':str(L_Syn_Img.COMP_COND(TbH,Sol[0],Context,Margin=False)[1]),'time':t,'Mu_real_0':titaReal['H']['mu'][0],'Sigma_real_0':np.sqrt(titaReal['H']['sigma2'][0]),'Mu_real_1':titaReal['H']['mu'][1],'Sigma_real_1':np.sqrt(titaReal['H']['sigma2'][1]),'Mu_rec_0':tita['H']['mu'][0],'Sigma_rec_0':np.sqrt(tita['H']['sigma2'][0]),'Mu_rec_1':tita['H']['mu'][1],'Sigma_rec_1':np.sqrt(tita['H']['sigma2'][1])}
                            df = df.append(dic, ignore_index=True)
                            #errores.append(L_Syn_Img.COMP_COND(TbH,Sol[0],Context,Margin=False)[0])
                            RMSE_H=L_Syn_Img.COMP_COND(TbH,Sol[0],Context,Margin=False)[0]
                            RMSE_V=L_Syn_Img.COMP_COND(TbV,Sol[1],Context,Margin=False)[0]
                            RMSE.append([Obs_std,RMSE_H,RMSE_V])
                            
                            dic = {'CellSize':km,'Method': Method,'Band':Band,'Pol': 'V','Img_num': n_img,'Obs_num': n_obs,'Obs_std_error':Obs_std,'RMSE':str(L_Syn_Img.COMP_COND(TbV,Sol[1],Context,Margin=False)[0]),'CoefCorr':str(L_Syn_Img.COMP_COND(TbV,Sol[1],Context,Margin=False)[1]),'time':t,'Mu_real_0':titaReal['V']['mu'][0],'Sigma_real_0':np.sqrt(titaReal['V']['sigma2'][0]),'Mu_real_1':titaReal['V']['mu'][1],'Sigma_real_1':np.sqrt(titaReal['V']['sigma2'][1]),'Mu_rec_0':tita['V']['mu'][0],'Sigma_rec_0':np.sqrt(tita['V']['sigma2'][0]),'Mu_rec_1':tita['V']['mu'][1],'Sigma_rec_1':np.sqrt(tita['V']['sigma2'][1])}
                            df = df.append(dic, ignore_index=True)
            
                            print('termino:',km,Band,Method,Obs_std,n_img,n_obs)
                            #if export_solutions:
                                #L_Output.Export_Solution(Sol, Context, Band, 'S_'+str(km)+'_'+Band+'_'+Method+str(n_img)+'_'+str(n_obs)+'_'+str(Obs_std).replace('.','_'))
                Res[Band] = np.array(RMSE)
                        
                        
#np.savetxt(csv_dir + "./Noise_EM_KA.csv", Res['KA'], delimiter=",")   
#np.savetxt(csv_dir + "./Noise_EM_KU.csv", Res['KU'], delimiter=",")   
#np.savetxt(csv_dir + "./Noise_EM_K.csv", Res['K'], delimiter=",")   
#np.savetxt(csv_dir + "./Noise_EM_X.csv", Res['X'], delimiter=",")   
#np.savetxt(csv_dir + "./Noise_EM_C.csv", Res['C'], delimiter=",")   

#np.savetxt(csv_dir + "./Noise_LSQR_KA.csv", Res['KA'], delimiter=",")   
#np.savetxt(csv_dir + "./Noise_LSQR_KU.csv", Res['KU'], delimiter=",")   
#np.savetxt(csv_dir + "./Noise_LSQR_K.csv", Res['K'], delimiter=",")   
#np.savetxt(csv_dir + "./Noise_LSQR_X.csv", Res['X'], delimiter=",")   
#np.savetxt(csv_dir + "./Noise_LSQR_C.csv", Res['C'], delimiter=",")   

#np.savetxt(csv_dir + "./Noise_BGF_KA.csv", Res['KA'], delimiter=",")   
#np.savetxt(csv_dir + "./Noise_BGF_KU.csv", Res['KU'], delimiter=",")   
#np.savetxt(csv_dir + "./Noise_BGF_K.csv", Res['K'], delimiter=",")   
#np.savetxt(csv_dir + "./Noise_BGF_X.csv", Res['X'], delimiter=",")   
#np.savetxt(csv_dir + "./Noise_BGF_C.csv", Res['C'], delimiter=",")   



#df.to_csv(csv_dir + csv_name + '.csv')
            
#%%            

#%%
M=np.array(df['RMSE'].values,dtype=float).reshape([5,9,4,2])
#Band, Noise, Meth, Pol

#MM=M.mean(axis=3).mean(axis=1) ##img and obs means
RAND=M[:,:,:,0] # me quedo solo con la pol H
TRIG=M[:,:,:,1] # me quedo con la Pol V


RAND_25KA=RAND[0,:,:] #selecciono mapa y banda
RAND_25KU=RAND[1,:,:] #selecciono mapa y banda
RAND_25K =RAND[2,:,:] #selecciono mapa y banda
RAND_25X =RAND[3,:,:] #selecciono mapa y banda
RAND_25C =RAND[4,:,:] #selecciono mapa y banda

TRIG_25KA=TRIG[0,:,:] #selecciono mapa y banda
TRIG_25KU=TRIG[1,:,:] #selecciono mapa y banda
TRIG_25K =TRIG[2,:,:] #selecciono mapa y banda
TRIG_25X =TRIG[3,:,:] #selecciono mapa y banda
TRIG_25C =TRIG[4,:,:] #selecciono mapa y banda
#%%
import matplotlib.pyplot as plt

#%%
plt.clf()
plt.plot(TRIG_25KA)
plt.ylim(ymax=5)
plt.legend(Methods,loc='upper left', shadow=True)
plt.title("25km KA TRIG.")
#%%
plt.figure()
plt.plot(RAND_25KA)
plt.ylim(ymax=5)
plt.legend(Methods,loc='upper left', shadow=True)
plt.title("25km KA RAND.")
#%%
plt.figure()
plt.plot(TRIG_25K)
plt.ylim(ymax=7)
plt.legend(Methods,loc='upper left', shadow=True)
plt.title("25km K TRIG.")
#%%
plt.figure()
plt.plot(RAND_25K)
plt.ylim(ymax=7)
plt.legend(Methods,loc='upper left', shadow=True)
plt.title("25km K RAND.")
#%%
plt.figure()
plt.plot(TRIG_25C)
plt.ylim(ymax=25)
plt.legend(Methods,loc='upper left', shadow=True)
plt.title("25km C TRIG.")
#%%
plt.figure()
plt.plot(RAND_25C)
plt.ylim(ymax=25)
plt.legend(Methods,loc='upper left', shadow=True)
plt.title("25km C RAND.")


#%%
#%%
#%%

#%%
plt.clf()
plt.plot(TRIG_50KA)
plt.ylim(ymax=5)
plt.legend(Methods,loc='upper left', shadow=True)
plt.title("50km KA TRIG.")
#%%
plt.figure()
plt.plot(RAND_50KA)
plt.ylim(ymax=5)
plt.legend(Methods,loc='upper left', shadow=True)
plt.title("50km KA RAND.")
#%%
plt.figure()
plt.plot(TRIG_50K)
plt.ylim(ymax=7)
plt.legend(Methods,loc='upper left', shadow=True)
plt.title("50km K TRIG.")
#%%
plt.figure()
plt.plot(RAND_50K)
plt.ylim(ymax=7)
plt.legend(Methods,loc='upper left', shadow=True)
plt.title("50km K RAND.")
#%%
plt.figure()
plt.plot(TRIG_50C)
plt.ylim(ymax=50)
plt.legend(Methods,loc='upper left', shadow=True)
plt.title("50km C TRIG.")
#%%
plt.figure()
plt.plot(RAND_50C)
plt.ylim(ymax=50)
plt.legend(Methods,loc='upper left', shadow=True)
plt.title("50km C RAND.")
