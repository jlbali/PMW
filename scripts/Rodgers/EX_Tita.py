# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 16:26:12 2017

@author: rgrimson
"""
from __future__ import print_function

#Import libraries
import L_ReadObs
import L_Context
import L_Solvers
import numpy as np


#WDIR
import my_base_dir
BASE_DIR = my_base_dir.get()
MAPA='Ajo_10km'
wdir=BASE_DIR + 'Mapas/%s/'%MAPA

#%%
#PROCESS PARAMETERS
HDFfilename='GW1AM2_201208110420_224D_L1SGBTBR_2210210'
Band='ka'
#%%
def mu_grid(tita_pol,K,VType):
    n_Vars=K.shape[1]
    mu=tita_pol['mu']
    #sigma2=tita_pol['sigma2']

    #Sx=np.zeros([n_Vars])
    x0=np.zeros(n_Vars)
    for v in range(n_Vars):
        #Sx[v]=abs(sigma2[VType[v]])
        x0[v]=mu[VType[v]]
        
    return x0
    
def sigma_grid(tita_pol,K,VType):
    n_Vars=K.shape[1]
    sigma2=tita_pol['sigma2']

    Sx=np.zeros([n_Vars])
    for v in range(n_Vars):
        Sx[v]=np.sqrt(abs(sigma2[VType[v]]))
        
    return Sx
    
def compute_LP(x,pol,tita,Observation):
    y=Observation['Tb'][pol]
    #P(y|x) · P(x|θ)
    K=Observation['Wt']
    Dy=y-K.dot(x)
    LPyx=-(Dy**2).sum() #assuming std_error=1
    VType=Observation['VType']
    
    Dx = x-mu_grid(tita[pol],K,VType )
    Sx   = sigma_grid(tita[pol],K,VType)
    LPxt = -(((Dx/Sx)**2).sum()+np.log(Sx**2).sum())
    LP   = LPyx + LPxt
    print("LP = LPyx + LPxt  ==> %.2f = %.2f + %.2f"%(LP,LPyx,LPxt))
    print("E(|Dx|)=%-2f"%abs(Dx).mean())
    print("E(|Dy|)=%-2f"%abs(Dy).mean())
    return LP, LPyx, LPxt, abs(Dx).mean(), abs(Dy).mean()
#%%
def main(): 
    Method='Rodgers'
    #Load Context    
    Context=L_Context.Load_Context(wdir)

    print('######################################')
    print('##   PRECOMPUTING K for BAND',Band,'#####')
    print('######################################')
    Observation, tita = L_ReadObs.Load_PKL_or_Compute_Kernels(HDFfilename,Context,Band,GC=True)
    print('######################################')
    print('##   SOLVING BAND',Band,'  Method',Method)
    print('######################################')
       
    #%%    
    ls=[]
    lp=[]    
    lpyx=[]    
    lpxt=[]    
    ex=[]
    ey=[]
    lso=[]
    lsi=[]
    pol='H'
    for std in np.arange(.01,20,1):
        print ("std:", end=' ')
        print(std, end='\t\t')
        #Solve system
        
        tita[pol]['sigma2']=np.array([std*std,std*std])
        Sol=L_Solvers.Solve(Context, Observation, Method, tita)
        lsi.append(std)
        sstd=Sol[0][VType==0].std()
        lso.append(sstd)
        print(std,sstd)
        LP, LPyx, LPxt , Ex, Ey = compute_LP(Sol[0],pol,tita,Observation)
        lp.append(LP)
        lpyx.append(LPyx)
        lpxt.append(LPxt)
        ex.append(Ex)
        ey.append(Ey)
        
        
        #print ("P:%.10f"%p)
    
    
    import matplotlib.pyplot as plt
    plt.clf()
    plt.plot(ls,lp)
    plt.plot(ls,lpyx)
    plt.plot(ls,lpxt)
    plt.plot(ls,ex)
    plt.plot(ls,ey)
    
    
    SolIt=L_Solvers.Solve(Context, Observation, "Rodgers_IT", tita)