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
import L_Files
import L_ParamEst
import L_Syn_Img
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

#%%
def Coef_correlacion(img_sintetica,img_solucion): # Toma dos imagenes, cada imagen es un array
    matriz_correlacion = np.corrcoef(img_sintetica,img_solucion)
    return matriz_correlacion[0,1]


#%%   BG init
dic_maxObs={'KA12':6,'KU12':6,'K12':6,'X12':4,'C12':4,'KA25':16,'KU25':16,'K25':16,'X25':10,'C25':10,'KA50':32,'KU50':30,'K50':30,'X50':16,'C50':16}
dic_omega={50:0.001,25:.001,12:0.000001}

#%%

def BG(Context, Observation, omega = 0.001, gamma = np.pi/4, error_std = 1, max_obs =  4):
    BG_Coef = L_Solvers.BG_Precomp_MaxObs_Cell(distances, Context, Observation, gamma=gamma,w=omega,error_std=error_std,MaxObs=max_obs)
    Sol = [BG_Coef.dot(Observation['Tb']['H']), BG_Coef.dot(Observation['Tb']['V'])]
    return Sol
    
def RMSE_BG(Tb,Context, Observation, omega = 0.001, gamma = np.pi/4, error_std = 1, max_obs =  4):
    Sol=BG(Context, Observation, omega = omega, gamma = gamma, error_std = error_std, max_obs =  max_obs)
    RMSE=L_Syn_Img.RMSE_M(Tb,Sol[1],Context)
    print("RMSE: %.5f\n*****************************************"%RMSE)        
    return RMSE
    
#%%
# INPUTS
HDFfilename='GW1AM2_201207310439_227D_L1SGBTBR_2210210'
NeighborsBehaviour = L_Context.NO_NEIGHBORS 
#L_Context.LOAD_NEIGHBORS
csv_name ='corrida'

#%%
Obs_std  = 1.0
MAPAS=['LCA_12500m','LCA_25000m','LCA_50000m']
MAPAS=['LCA_50000m','LCA_25000m','LCA_12500m']
#MAPAS=['LCA_12500m']
Bands=['KA','KU','K','X','C']
#Bands=['KA']
tot_img_samples=10 #CANTIDAD DE IMAGENES SINTETICAS GENERADAS

MAPA='LCA_50000m'
Band='X'

#%%
   
def calc_gamma(Tb,Context, Observation, XI,XD,XC,YI,YD,YC,MO):
    End=False
    Error=False
    while not End:
        if (YC>YI)and(YC>YD):
            print("Error")
            Error=True
            End=True
            if YI<YD:
                XF=XI
                YF=YI 
            else:
                XF=XD
                YF=YD 
        elif (YC<YI)and(YC<YD):
            if (np.random.randint(2)==0):
                print("Elijo der")
                Xn=(XC+XD)/2
                Yn = RMSE_BG(Tb,Context, Observation, gamma=Xn,max_obs = MO,omega=omega)
                if Yn<YC:
                    XI=XC
                    XC=Xn
                    YI=YC
                    YC=Yn
                elif (Yn>YD):
                    print("Error")
                    Error=True
                    End=True
                    XF=XC
                    YF=YC
                else:
                    XD=Xn
                    YD=Yn
            else:
                print("Elijo izq")
                Xn=(XC+XI)/2
                Yn=RMSE_BG(Tb,Context, Observation, gamma=Xn,max_obs = MO,omega=omega)
                if Yn<YC:
                    XD=XC
                    XC=Xn
                    YD=YC
                    YC=Yn
                elif (Yn>YI):
                    print("Error")
                    Error=True
                    End=True
                    XF=XC
                    YF=YC
                else:
                    XI=Xn
                    YI=Yn
        else:
            print("Refinar", end=" ")
            if (YD<YI):
                print("a derecha")
                XI=XC
                XC=(XD+XC)/2
                YI=YC
                YC=RMSE_BG(Tb,Context, Observation, gamma=XC,max_obs = MO,omega=omega)
            else:
                print("a izquierda")
                XD=XC
                XC=(XI+XC)/2
                YD=YC
                YC=RMSE_BG(Tb,Context, Observation, gamma=XC,max_obs = MO,omega=omega)
        XF=XC
        YF=YC
                
        D=np.max([np.abs(YI-YC),np.abs(YD-YC),np.abs(XI-XC),np.abs(XD-XC)])
        if D<0.001: 
            End=True
            
    return XF,YF,Error
#%%

np.random.seed(1)
#cols = ['km','Band','MaxObs','w','Gamma','RMSE']
cols = ['km','Band','MaxObs','w','Gamma','RMSE','Error']

df = pd.DataFrame(columns = cols)
i=0

for MAPA in MAPAS:
    wdir=BASE_DIR + 'Mapas/%s/'%MAPA
    Context=L_Context.Load_Context(wdir, NeighborsBehaviour)
    L_Syn_Img.Set_Margins(Context)
    km=int(MAPA[4:6])
    omega=dic_omega[km]
    distances_path = wdir + 'distance'
    if not L_Files.exists(distances_path):
       distances = L_Solvers.precompute_all_optimized(Context)
       L_Solvers.save_distances(distances_path, distances)
    distances = L_Solvers.load_distances(distances_path)

    for rep in range(tot_img_samples):
        TbH = L_Syn_Img.Create_Random_Synth_Img(Context)
        TbV = L_Syn_Img.Create_Trig_Synth_Img(Context)
        titaReal=copy.deepcopy(L_ParamEst.Compute_param([TbH,TbV],Context))
        
    
        for Band in Bands:
            Observation, tita=L_ReadObs.Load_Band_Kernels(HDFfilename,Context,Band)
            Observation=L_Syn_Img.Simulate_PMW(TbH,TbV,Context,Observation,Obs_error_std=Obs_std)
            Tb=TbV
        
            key = Band + str(km)
            max_obs = dic_maxObs[key]
            XI=0
            XD=np.pi/2
            XC=np.pi/4
            
            YI=RMSE_BG(Tb,Context, Observation, gamma=XI,max_obs = max_obs,omega=omega)
            YD=RMSE_BG(Tb,Context, Observation, gamma=XD,max_obs = max_obs,omega=omega)
            YC=RMSE_BG(Tb,Context, Observation, gamma=XC,max_obs = max_obs,omega=omega)
            
            best_gamma,lowest_RMSE, finished_stat = calc_gamma(Tb, Context, Observation, XI,XD,XC,YI,YD,YC,max_obs)
            df.loc[i] = [km,Band,max_obs,omega,best_gamma,lowest_RMSE, finished_stat]
            if finished_stat:
                for gamma in np.linspace(0,np.pi/2,11):
                    RMSE=RMSE_BG(Tb,Context, Observation, gamma=gamma,max_obs = max_obs,omega=omega)
                    df.loc[i] = [km,Band,max_obs,omega,gamma,RMSE,True]
                    i+=1

                

            i+=1

df.to_csv(csv_dir+'Gammas_BG_CELL_VPol_Gammas_10rep_.csv')

#%%
G={}
Gammas={}
for km in [12,25,50]:
    G[km]={}
    for Band in Bands:
        G[km][Band]={}
        W1=(df['km']==km)
        W2=(df[W1]['Band']==Band)
        Gammas[str(km)+Band]=df[W1][W2]['Gamma'].mean()
        G[km][Band]['Gamma']=df[W1][W2]['Gamma'].mean()
        G[km][Band]['GammaStd']=df[W1][W2]['Gamma'].std()
        G[km][Band]['RMSE']=df[W1][W2]['RMSE'].mean()
        G[km][Band]['RMSEStd']=df[W1][W2]['RMSE'].std()
dg=pd.DataFrame(G)
dgam=pd.DataFrame(Gammas)
dg.to_csv(csv_dir+'Gammas_Avg_VPol_Gammas_10rep_.csv')
dgam.to_csv(csv_dir+'Gammas_Avg_BG_.csv')

