# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 10:15:19 2017

@author: rgrimson
"""

# ANTES DE COMENZAR REVISAR MAPA Y EL NOMBRE DEL CSV

import my_base_dir
BASE_DIR = my_base_dir.get()
MAPA='Ajo_10km'
wdir=BASE_DIR + 'Mapas/%s/'%MAPA
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

import matplotlib.pyplot as plt
# INICIALIZO LOS VALORES A EXPERIMENTAR:
#%%
HDFfilename='GW1AM2_201207310439_227D_L1SGBTBR_2210210'
Bands=['X','K','KU', 'KA','C']
Bands=['KA','KU','K', 'X','C']

Context=L_Context.Load_Context(wdir)
L_Syn_Img.Set_Margins(Context)

np.random.seed(1)
#%%
#FIND Ws
#%%
BestW={}
BestG={}
BestRH={}
BestRV={}
MaxObsS=[3,4,6,9,12,16,20,25]
OUT_FN=wdir+"Best_GammaW_ManyObs.txt"

for MaxObs in MaxObsS:
    BestW[MaxObs]={}
    BestG[MaxObs]={}
    BestRH[MaxObs]={}
    BestRV[MaxObs]={}
    with open(OUT_FN, "a") as myfile:
        myfile.write("\n\n###############################################\n")
        myfile.write("### %d Obs, %s ##.\n"%(MaxObs,MAPA))
        myfile.write("###############################################\n")

    for Band in Bands:
        try:
            print("\n\n********************************************")
            print("Starting Band: %s with %d observations in %s."%(Band,MaxObs,MAPA))
            print("********************************************")
            
            with open(OUT_FN, "a") as myfile:
                myfile.write("\n\n********************************************\n")
                myfile.write("Starting Band: %s with %d observations in %s.\n"%(Band,MaxObs,MAPA))
                myfile.write("********************************************\n")
            
            Observation, tita=L_ReadObs.Load_Band_Kernels(HDFfilename,Context,Band)
            np.random.seed(1)
            TbH = L_Syn_Img.Create_Random_Synth_Img(Context)
            np.random.seed(1)
            TbV = L_Syn_Img.Create_Trig_Synth_Img(Context)
            Observation=L_Syn_Img.Simulate_PMW(TbH,TbV,Context,Observation,Obs_error_std=1.0)
            #L_Output.Export_Solution([TbH,TbV], Context, Band, "SynthImg")
    
            gamma = np.pi/4
            w_min=10**-50
            w_max=10**50
            RMSEsVw = 1000
            RMSE_gain=1.0
            
            w1=1.0
            w0=0.1
    
                
            BG_Coef=L_Solvers.BG_Precomp_MaxObs(Context, Observation, gamma=gamma,w=w0,MaxObs=MaxObs)
            Sol=[BG_Coef.dot(Observation['Tb']['H']), BG_Coef.dot(Observation['Tb']['V'])]
            RMSEsHw0=L_Syn_Img.RMSE_M(TbH,Sol[0],Context)
            RMSEsVw0=L_Syn_Img.RMSE_M(TbV,Sol[1],Context)    
            print("RMSEsHw: %.10f"%RMSEsHw0)
            print("RMSEsVw: %.10f"%RMSEsVw0)
            
    
            BG_Coef=L_Solvers.BG_Precomp_MaxObs(Context, Observation, gamma=gamma,w=w1,MaxObs=MaxObs)
            Sol=[BG_Coef.dot(Observation['Tb']['H']), BG_Coef.dot(Observation['Tb']['V'])]
    
            RMSEsHw1=L_Syn_Img.RMSE_M(TbH,Sol[0],Context)
            RMSEsVw1=L_Syn_Img.RMSE_M(TbV,Sol[1],Context)    
            print("RMSEsHw: %.10f"%RMSEsHw1)
            print("RMSEsVw: %.10f"%RMSEsVw1)
            
        
            if (RMSEsVw0>RMSEsVw1):
                factor=10
                w=w1
                print("Going up!")
            else:
                factor=.1
                w=w0
                print("Going down!")
            
            
            while ((w>w_min) and (w<w_max) and (RMSE_gain>0)):
                w_anterior = w
                w = w*factor
                RMSEsVw_anterior = RMSEsVw
                
                BG_Coef=L_Solvers.BG_Precomp_MaxObs(Context, Observation, gamma=gamma,w=w,MaxObs=MaxObs)
                Sol=[BG_Coef.dot(Observation['Tb']['H']), BG_Coef.dot(Observation['Tb']['V'])]
    
                RMSEsHw=L_Syn_Img.RMSE_M(TbH,Sol[0],Context)
                RMSEsVw=L_Syn_Img.RMSE_M(TbV,Sol[1],Context)    
                print(" -done")
                print("RMSEsHw: %.10f"%RMSEsHw)
                print("RMSEsVw: %.10f"%RMSEsVw)
            
                RMSE_gain = RMSEsVw_anterior - RMSEsVw
            
            
            w = w_anterior
            print("\n\nBest w: %.10f"%w)
            
            #%%
            #GAMMAS
            #%%
            gammas =np.linspace(0,np.pi/2,10)
            N=len(gammas)
            RMSEsH=np.zeros(N)
            RMSEsV=np.zeros(N)
            #%%
            for i in range(N):
                gamma=gammas[i]
                BG_Coef=L_Solvers.BG_Precomp_MaxObs(Context, Observation, gamma=gamma,w=w,MaxObs=MaxObs)
                Sol=[BG_Coef.dot(Observation['Tb']['H']), BG_Coef.dot(Observation['Tb']['V'])]
    
                RMSEsH[i]=L_Syn_Img.RMSE_M(TbH,Sol[0],Context)
                RMSEsV[i]=L_Syn_Img.RMSE_M(TbV,Sol[1],Context)
                print("- done")
                print("RMSEsH: %.10f"%RMSEsH[i])
                print("RMSEsV: %.10f"%RMSEsV[i])
                #L_Output.Export_Solution(Sol, Context, Band, 'Sol20120731_BGF_g_'+Band+'_'+str(i)+'_'+str(int(gamma*100)))
            #%%
            #Find best gamma
            gammai=np.argmin(RMSEsV)
            gamma=gammas[gammai]
            print("********************\nBest gamma (coarse): %.10f"%gamma)
            #%%
            #REFINE
            gammam=gammas[gammai-1]
            gammaM=gammas[gammai+1]
            Q=10
            delta=(gammaM-gammam)/Q
            gammaR =np.linspace(gammam+delta,gammaM-delta,Q)
            M=len(gammaR)
            RMSEsHr=np.zeros(M)
            RMSEsVr=np.zeros(M)
            for i in range(M):
                gamma=gammaR[i]
                BG_Coef=L_Solvers.BG_Precomp_MaxObs(Context, Observation, gamma=gamma,w=w,MaxObs=MaxObs)
                Sol=[BG_Coef.dot(Observation['Tb']['H']), BG_Coef.dot(Observation['Tb']['V'])]
    
                RMSEsHr[i]=L_Syn_Img.RMSE_M(TbH,Sol[0],Context)
                RMSEsVr[i]=L_Syn_Img.RMSE_M(TbV,Sol[1],Context)
                print("- done")
                print("RMSEsH: %.10f"%RMSEsHr[i])
                print("RMSEsV: %.10f"%RMSEsVr[i])
                #L_Output.Export_Solution(Sol, Context, Band, 'Sol20120731_BGF_R_'+str(i)+'_'+str(int(gamma*100)))
            #%%
            gammai=np.argmin(RMSEsVr)
            print("********************\nBest gamma (fine) for Band %s: %.10f"%(Band,gamma))
            print("RMSE H:%.10f V:%.10f"%(RMSEsHr[i],RMSEsVr[i]))
            with open(OUT_FN, "a") as myfile:
                myfile.write("Band: %s, w: %.10f, g: %.10f\nRSME: H:%.10f V:%.10f\n"%(Band,w,gamma,RMSEsHr[i],RMSEsVr[i]))
    
            BestW[MaxObs][Band]=w
            BestG[MaxObs][Band]=gamma
            
            BestRH[MaxObs][Band]=RMSEsHr[i]
            BestRV[MaxObs][Band]=RMSEsVr[i]
        except:
            print("Unexpected error, Band %s"%Band)
    
        
        
        
#%%
#import matplotlib.pyplot as plt
#plt.figure()
#plt.plot(gammas, np.log(RMSEsV))
#plt.figure()
#plt.plot(gammaR, np.log(RMSEsVr))
