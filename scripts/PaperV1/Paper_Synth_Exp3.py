# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 17:03:26 2017

@author: mariela
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 16:26:12 2017

@author: rgrimson
"""

#from __future__ import print_function

#WDIR
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
import matplotlib.pyplot as plt
import pandas as pd
import time


def Coef_correlacion(img_sintetica,img_solucion): # Toma dos imagenes, cada imagen es un array
    matriz_correlacion = np.corrcoef(img_sintetica,img_solucion)
    return matriz_correlacion[0,1]


#%%
##PROCESS PARAMETERS
#HDFfilename='GW1AM2_201208110420_224D_L1SGBTBR_2210210'
##Bands=['ka','K','ku','x','c']
#Band='ka'
#Obs_error_std=0.5
#NeighborsBehaviour = L_Context.LOAD_NEIGHBORS
#Methods = ["Weights", "Global_GCV_Tichonov", "K_Global_GCV_Tichonov","Rodgers_IT",'LSTSQR','BGF']
#N_rep=3#3



#%%
HDFfilename='GW1AM2_201207310439_227D_L1SGBTBR_2210210'

Bands=['KA','KU','K','X','C']
#Bands=['X']

Methods = ['Rodgers_IT','Weights','LSQR']
#Methods = ['Rodgers_IT']

N_rep=100
Obs_error_std=1.0


#%%

def main(): 
    #%%###################
    ## PREPROCESSING    
    cols = ['Method','Band','rep','RMSE_H','RMSE_V','CC_H','CC_V','RMuH0','RMuH1','RstdH0','RstdH1','RMuV0','RMuV1','RstdV0','RstdV1','SMuH0','SMuH1','SstdH0','SstdH1','SMuV0','SMuV1','SstdV0','SstdV1']
    df=pd.DataFrame(columns=cols)
    Context=L_Context.Load_Context(wdir)
    L_Syn_Img.Set_Margins(Context)
    N=0
    for Band in Bands:
        Observation, tita=L_ReadObs.Load_Band_Kernels(HDFfilename, Context, Band)#Load_PKL_or_Compute_Kernels(HDFfilename,Context,Band,GC=True)
    
        for rep in range(N_rep): # En cada repeticion arma una imagen sintetica distinta para cada banda, esta sería l nueva verdad del terreno.
            print("*************")
            print("Iteration: %d"%rep)       
            print("*************")
            print("Band: %s"%Band)
            
            TbH = L_Syn_Img.Create_Random_Synth_Img(Context,sgm0=10,sgm1=5,mu0=180,mu1=270)
            TbV = L_Syn_Img.Create_Trig_Synth_Img(Context)
            titaOrig=L_ParamEst.recompute_param([TbH,TbV],Observation,tita)
            L_ParamEst.pr(titaOrig,'Synthetic Image')
            RMuH0=titaOrig['H']['mu'][0]
            RMuH1=titaOrig['H']['mu'][1]
            RstdH0=np.sqrt(titaOrig['H']['sigma2'][0])
            RstdH1=np.sqrt(titaOrig['H']['sigma2'][1])
            RMuV0=titaOrig['V']['mu'][0]
            RMuV1=titaOrig['V']['mu'][1]
            RstdV0=np.sqrt(titaOrig['V']['sigma2'][0])
            RstdV1=np.sqrt(titaOrig['V']['sigma2'][1])

            Observation=L_Syn_Img.Simulate_PMW(TbH,TbV,Context,Observation,Obs_error_std=Obs_error_std)

            for Method in Methods:
                Sol=L_Solvers.Solve(Context,Observation,Method, tita)
                tita=L_ParamEst.recompute_param(Sol,Observation,tita)
                print("\nErrors:\n--------")
                print("RMSE H: %.3f"%(L_Syn_Img.RMSE_M(TbH,Sol[0],Context))) # ¿Tiene sentido calcularlo en cada iteracion?
                print("RMSE V: %.3f"%(L_Syn_Img.RMSE_M(TbV,Sol[1],Context)))
                print("RMSE H: %.3f"%(L_Syn_Img.RMSE(TbH,Sol[0]))) # ¿Tiene sentido calcularlo en cada iteracion?
                print("RMSE V: %.3f"%(L_Syn_Img.RMSE(TbV,Sol[1])))
                RMSE_H      = L_Syn_Img.RMSE_M(TbH,Sol[0],Context)
                RMSE_V      = L_Syn_Img.RMSE_M(TbV,Sol[1],Context)
                CC_H = Coef_correlacion(TbH,Sol[0])
                CC_V = Coef_correlacion(TbH,Sol[1])
                SMuH0=tita['H']['mu'][0]
                SMuH1=tita['H']['mu'][1]
                SstdH0=np.sqrt(tita['H']['sigma2'][0])
                SstdH1=np.sqrt(tita['H']['sigma2'][1])
                SMuV0=tita['V']['mu'][0]
                SMuV1=tita['V']['mu'][1]
                SstdV0=np.sqrt(tita['V']['sigma2'][0])
                SstdV1=np.sqrt(tita['V']['sigma2'][1])
                COL=[Method, Band, rep, RMSE_H, RMSE_V,CC_H,CC_V,RMuH0,RMuH1,RstdH0,RstdH1,RMuV0,RMuV1,RstdV0,RstdV1,SMuH0,SMuH1,SstdH0,SstdH1,SMuV0,SMuV1,SstdV0,SstdV1]
                df.loc[N]=COL
                N=N+1
    df.to_csv(wdir + 'SALIDA_RAFA.csv')
#%%            

        

if __name__ == "__main__":
    main()

def summary():
    df=pd.read_csv(wdir + 'SALIDA_RAFA.csv')
    for Method in Methods:
        print("%s "%Method,end='')
    print("\n---------------------------------")
    for Band in Bands:
        for Method in Methods:
            dfBM=df.loc[(df['Band'] == Band)&(df['Method'] == Method)]
            RMSE_H=dfBM['RMSE_H'].mean()
            RMSE_Hstd=dfBM['RMSE_H'].std()
            CC_H=dfBM['CC_H'].mean()
            print("%.2f (%.2f) & %.2f &"%(RMSE_H,RMSE_Hstd,CC_H),end=' ')
        print("\\%s"%Band,end=' \n')
    print("---------------------------------")
    for Band in Bands:
        for Method in Methods:
            dfBM=df.loc[(df['Band'] == Band)&(df['Method'] == Method)]
            RMSE_V=dfBM['RMSE_V'].mean()
            RMSE_Vstd=dfBM['RMSE_V'].std()
            CC_V=dfBM['CC_V'].mean()
            print("%.2f (%.2f) & %.2f &"%(RMSE_V,RMSE_Vstd,CC_V),end=' ')
        print("\\%s"%Band,end=' \n')
            