# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 10:15:19 2017

@author: rgrimson
"""
import my_base_dir
BASE_DIR = my_base_dir.get()
MAPA='Ajo_25km'
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
Bands=['KA','KU','K', 'X','C']
#Bands=['KA','K','C']
Observations={}
Context=L_Context.Load_Context(wdir)
L_Syn_Img.Set_Margins(Context)

D={}
for Band in Bands:
    Observations[Band], tita=L_ReadObs.Load_Band_Kernels(HDFfilename,Context,Band)
    D[Band]={}
    D[Band]['RMSEHs']=[]
    D[Band]['RMSEVs']=[]
    D[Band]['STDHo0']=[]
    D[Band]['STDHo1']=[]
    D[Band]['STDVo0']=[]
    D[Band]['STDVo1']=[]
    D['STDHi0']=[]
    D['STDHi1']=[]
    D['STDVi0']=[]
    D['STDVi1']=[]
    
    

#%%

STDs=[10**i for i in np.arange(-1,1.1,.05)]
for std in STDs:
    TbH = L_Syn_Img.Create_Random_Synth_Img(Context,sgm0=std,sgm1=std)
    TbV = L_Syn_Img.Create_Random_Synth_Img(Context,sgm0=std,sgm1=10)
    tita=L_ParamEst.recompute_param([TbH,TbV],Observations[Band],tita)
    D['STDHi0'].append(np.sqrt(tita['H']['sigma2'][0]))
    D['STDHi1'].append(np.sqrt(tita['H']['sigma2'][1]))
    D['STDVi0'].append(np.sqrt(tita['V']['sigma2'][0]))
    D['STDVi1'].append(np.sqrt(tita['V']['sigma2'][1]))

    for Band in Bands:
        Observations[Band]=L_Syn_Img.Simulate_PMW(TbH,TbV,Context,Observations[Band],Obs_error_std=1.0)
        L_ParamEst.compute_param([TbH,TbV], Observations[Band])
        Sol=L_Solvers.Solve_Rodgers_IT_NB(Context,Observations[Band], tita, damp=1.0,tol=0.00005, max_iter=50)

        RMSEH=L_Syn_Img.RMSE_M(TbH,Sol[0],Context)
        RMSEV=L_Syn_Img.RMSE_M(TbV,Sol[1],Context)    
        print(" -done")
        print("RMSEsHw: %.3f"%RMSEH)
        print("RMSEsVw: %.3f"%RMSEV)
        tita=L_ParamEst.recompute_param(Sol,Observations[Band],tita)        
        
        D[Band]['RMSEHs'].append(RMSEH)
        D[Band]['RMSEVs'].append(RMSEV)
        D[Band]['STDHo0'].append(np.sqrt(tita['H']['sigma2'][0]))
        D[Band]['STDHo1'].append(np.sqrt(tita['H']['sigma2'][1]))
        D[Band]['STDVo0'].append(np.sqrt(tita['V']['sigma2'][0]))
        D[Band]['STDVo1'].append(np.sqrt(tita['V']['sigma2'][1]))

#%%
d={'STDs':STDs,'STDHi0':D['STDHi0'],'STDHi1':D['STDHi1'],'STDVi0':D['STDVi0'],'STDVi1':D['STDVi1']}
for Band in Bands:
    d['STDHo0_'+Band]=D[Band]['STDHo0']
    d['STDHo1_'+Band]=D[Band]['STDHo1']
    d['RMSEHs_'+Band]=D[Band]['RMSEHs']
    d['STDVo0_'+Band]=D[Band]['STDVo0']
    d['STDVo1_'+Band]=D[Band]['STDVo1']
    d['RMSEVs_'+Band]=D[Band]['RMSEVs']
df=pd.DataFrame(d)
df.to_csv('Eval_Rodgers_NB_3_10km.csv')

#%%
#aux=df.sort(['STDHi0'])
plt.figure(1)
plt.clf()
plt.title("STAT H RMSE. Different std and bands. "+MAPA)
plt.xlabel('std for each land type')
plt.ylabel('RMSE')
plt.plot(df['STDHi0'],df['STDHi0'])
for Band in Bands:
    plt.plot(df['STDs'],df['RMSEHs_'+Band],label='RMSE '+Band)
plt.legend(loc='upper left')
plt.show()
plt.savefig(MAPA+"H_RMSE")
#%%
aux=df.sort(['STDHi0'])
plt.clf()
plt.title("STD in vs STD out (STAT) H. Different std and bands. "+MAPA)
plt.xlabel('std in')
plt.ylabel('std out')

plt.plot(aux['STDHi0'],aux['STDHi0'])
for Band in Bands:
    plt.plot(aux['STDHi0'],aux['STDHo0_'+Band],label=Band)

plt.legend(loc='upper left')
plt.show()
plt.savefig(MAPA+"H_STD")

#%%
for Band in Bands:
    plt.clf()
    plt.title("STD in vs STD out (both LT) and RMSE. "+MAPA+". Band: "+Band)
    plt.xlabel('std in')
    plt.ylabel('std out/RMSE')
    plt.plot(df['STDHi0'],df['STDHi0'])
    plt.plot(df['STDHi0'],df['STDHo0_'+Band],label='LT=0')
    plt.plot(df['STDHi1'],df['STDHo1_'+Band],label='LT=1')
    #plt.plot(df['STDVi0'],df['STDVo0_'+Band])
    #plt.plot(df['STDVi1'],df['STDVo1_'+Band])
    plt.plot(df['STDs'],df['RMSEHs_'+Band],label='RMSE')
    #plt.plot(df['STDs'],df['RMSEVs_'+Band])
    plt.legend(loc='upper left')
    plt.show()
    plt.savefig(MAPA+"H_STD_"+Band)
    
#%%
plt.clf()
plt.title("STAT V RMSE. Different std and bands. "+MAPA)
plt.xlabel('std for each land type')
plt.ylabel('RMSE')
plt.plot(df['STDVi0'],df['STDVi0'])
for Band in Bands:
    plt.plot(df['STDs'],df['RMSEVs_'+Band],label='RMSE '+Band)
plt.legend(loc='upper left')
plt.show()
plt.savefig(MAPA+"V_RMSE")
#%%
aux=df.sort(['STDVi0'])
plt.clf()
plt.title("STD in vs STD out (STAT) V. Different std and bands. "+MAPA)
plt.xlabel('std in')
plt.ylabel('std out')

plt.plot(aux['STDVi0'],aux['STDVi0'])
for Band in Bands:
    plt.plot(aux['STDVi0'],aux['STDVo0_'+Band],label=Band)

plt.legend(loc='upper left')
plt.show()
plt.savefig(MAPA+"V_STD")

#%%
for Band in Bands:
    plt.clf()
    plt.title("STD in vs STD out (both LT) and RMSE. "+MAPA+". Band: "+Band)
    plt.xlabel('std in')
    plt.ylabel('std out/RMSE')
    plt.plot(df['STDVi0'],df['STDVi0'])
    plt.plot(df['STDVi0'],df['STDVo0_'+Band],label='LT=0')
    plt.plot(df['STDVi1'],df['STDVo1_'+Band],label='LT=1')
    #plt.plot(df['STDVi0'],df['STDVo0_'+Band])
    #plt.plot(df['STDVi1'],df['STDVo1_'+Band])
    plt.plot(df['STDs'],df['RMSEVs_'+Band],label='RMSE')
    #plt.plot(df['STDs'],df['RMSEVs_'+Band])
    plt.legend(loc='upper left')
    plt.show()
    plt.savefig(MAPA+"V_STD_"+Band)
    
