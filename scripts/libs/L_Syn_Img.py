# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 16:15:12 2017

@author: rgrimson
"""

from __future__ import print_function
from scipy.sparse.linalg import lsqr

import numpy as np
import gdal
import os
import libs.L_Output as L_Output
from statistics import median
import scipy as sp
import matplotlib.pyplot as plt
import sys

def Create_Random_Synth_Img(Context,sgm0=5,sgm1=10,mu0=180,mu1=270):
    n_Vars=Context['Dict_Vars']['n_Vars']    
    VType=Context['Dict_Vars']['VType']
    nlt=VType.max()+1

    Vars_of_type=[]
    for lt in range(nlt):
        Vars_of_type.append(np.where(VType==lt)[0])

    nvt=list(map(len,Vars_of_type))        
    mu=[mu0,mu1]
    sigma=[sgm0,sgm1]
    
    
    Tb=np.zeros(n_Vars)
    for lt in range(nlt):
        Tb[Vars_of_type[lt]]=np.random.normal(mu[lt],sigma[lt],nvt[lt])
    return Tb

def Create_Trig_Synth_Img(Context,sgm0=5,sgm1=10,mu0=180,mu1=270):
    n_Vars=Context['Dict_Vars']['n_Vars']    
    VType=Context['Dict_Vars']['VType']
    nlt=VType.max()+1

    Vars_of_type=[]
    for lt in range(nlt):
        Vars_of_type.append(np.where(VType==lt)[0])
        
    Tb=np.zeros(n_Vars)
    k=.2*Context['Param']['DX']/12500.0
    #sh=Context['Param']['DX']/12500.0/2
    
    for v in Vars_of_type[0]:
        Tb[v]=mu0+sgm0*np.sqrt(2)*np.sin((Context['Dict_Vars']['Var'][v]['Coords'][0][1]+.5)*k - (Context['Dict_Vars']['Var'][v]['Coords'][0][0]+.5)*k)
    for v in Vars_of_type[1]:
        Tb[v]=mu1+sgm1*np.sqrt(2)*np.sin((Context['Dict_Vars']['Var'][v]['Coords'][0][1]+.5)*k + (Context['Dict_Vars']['Var'][v]['Coords'][0][0]+.5)*k)

  
    return Tb

def Create_Trig_Synth_Img0(Context,sgm0=5,mu0=180):
    n_Vars=Context['Dict_Vars']['n_Vars']    
    VType=Context['Dict_Vars']['VType']
    nlt=VType.max()+1

    Vars_of_type=[]
    for lt in range(nlt):
        Vars_of_type.append(np.where(VType==lt)[0])
        
    Tb=np.zeros(n_Vars)
    k=.2*Context['Param']['DX']/12500.0
    #sh=Context['Param']['DX']/12500.0/2
    
    for v in Vars_of_type[0]:
        Tb[v]=mu0+sgm0*np.sqrt(2)*np.sin((Context['Dict_Vars']['Var'][v]['Coords'][0][1]+.5)*k - (Context['Dict_Vars']['Var'][v]['Coords'][0][0]+.5)*k)
 
    return Tb



def Create_Bimodal_Synth_Img_OLD(Context, mu=180):
    n_Vars=Context['Dict_Vars']['n_Vars']    
    VType=Context['Dict_Vars']['VType']
    nlt=VType.max()+1
        
    Tb=np.zeros(n_Vars)
    k=.2*Context['Param']['DX']/12500.0
    #sh=Context['Param']['DX']/12500.0/2
    x_length = 18
    y_length = 20
#%%        
    for v in range(n_Vars):
        x_coord = Context['Dict_Vars']['Var'][v]['Coords'][0][1]+.5
        y_coord = Context['Dict_Vars']['Var'][v]['Coords'][0][0]+.5
        x_arg = (x_coord / x_length)*1.5*np.pi+np.pi/4
        y_arg = ((y_coord - y_length/2)/ (y_length/2))*2.0
        Tb[v]= (np.sin(x_arg)*(sp.stats.norm.pdf(y_arg)+.7)+0.5)*100+ mu

    return Tb

def Create_True_Bimodal_Synth_Img(Context, mu1=180, mu2=270, sdev1=10, sdev2=10):
    n_Vars=Context['Dict_Vars']['n_Vars']    

    Tb=np.zeros(n_Vars)
#%%        
    for v in range(n_Vars):
        if np.random.random() < 0.5:
            Tb[v] = np.random.normal(mu1, sdev1)
        else:
            Tb[v] = np.random.normal(mu2, sdev2)
        #print(x_arg,y1_arg, y2_arg, Tb[v])
    #plt.hist(Tb,  bins=50)
    #sys.exit(0)
    return Tb



def Create_Bimodal_Synth_Img(Context, mu=180, sdev=0.1, alpha=0.25):
    n_Vars=Context['Dict_Vars']['n_Vars']    

    Tb=np.zeros(n_Vars)

    #sh=Context['Param']['DX']/12500.0/2
    x_length = 20
    y_length = 18
#%%        
    for v in range(n_Vars):
        x_coord = Context['Dict_Vars']['Var'][v]['Coords'][0][0]+.5
        y_coord = Context['Dict_Vars']['Var'][v]['Coords'][0][1]+.5
        #print(x_coord,y_coord)
        x_arg = (x_coord / x_length)
        y_arg = (y_coord / y_length)
        
        y_angle = y_arg*1.5*np.pi + 0.25*np.pi
        
        Tb[v]= np.sin(y_angle)+0.2*np.sin(2*np.pi*x_arg)
        #print(x_arg,y1_arg, y2_arg, Tb[v])
    Tb = (Tb/Tb.max())*100 + mu
    return Tb

#%%
def Create_Bimodal_Synth_Img_OLD2(Context, mu=180, sdev=0.1, alpha=0.25):
    n_Vars=Context['Dict_Vars']['n_Vars']    

    Tb=np.zeros(n_Vars)

    #sh=Context['Param']['DX']/12500.0/2
    x_length = 20
    y_length = 18
#%%        
    for v in range(n_Vars):
        x_coord = Context['Dict_Vars']['Var'][v]['Coords'][0][0]+.5
        y_coord = Context['Dict_Vars']['Var'][v]['Coords'][0][1]+.5
        #print(x_coord,y_coord)
        x_arg = (x_coord / x_length)-.5
        y1_arg = (y_coord / y_length)-alpha
        y2_arg = (y_coord / y_length)-(1-alpha)
        
        Tb[v]= sp.stats.norm.pdf(np.sqrt(x_arg**2+y1_arg**2)/sdev)-sp.stats.norm.pdf(np.sqrt(x_arg**2+y2_arg**2)/sdev)
        #print(x_arg,y1_arg, y2_arg, Tb[v])
    Tb = (Tb/Tb.max())*100 + mu
    return Tb



def Create_Slope_Synth_Img(Context, minTb=180, maxTb=270):
    n_Vars=Context['Dict_Vars']['n_Vars']    

    Tb=np.zeros(n_Vars)

    #sh=Context['Param']['DX']/12500.0/2
    
    minX = None
    maxX = None
    for v in range(n_Vars):
        x = Context['Dict_Vars']['Var'][v]['Coords'][0][0]+.5
        if minX is None:
            minX = x
            maxX = x
        elif minX > x:
            minX = x
        elif maxX < x:
            maxX = x
    print("Min X", minX)
    print("Max X", maxX)
#%%        
    for v in range(n_Vars):
        x = Context['Dict_Vars']['Var'][v]['Coords'][0][0]+.5        
        Tb[v]= minTb + ((maxTb - minTb)/(maxX - minX))*(x - minX)
        #print(x_arg,y1_arg, y2_arg, Tb[v])
    return Tb





#x=np.arange(0,6.28,0.01)
#X=np.sin(x)
#y=np.arange(-2,2,0.01)
#Y=sp.stats.norm.pdf(y)+.7#Y=np.cos(y)
#M=(X[:, np.newaxis].dot(Y[ np.newaxis,:]))


def Create_RandTrig_Synth_Img(Context,sgm0=5,sgm1=10,mu0=180,mu1=270,sgmR0=3,sgmR1=3):
    n_Vars=Context['Dict_Vars']['n_Vars']    
    VType=Context['Dict_Vars']['VType']
    nlt=VType.max()+1

    Vars_of_type=[]
    for lt in range(nlt):
        Vars_of_type.append(np.where(VType==lt)[0])
        
    nvt=list(map(len,Vars_of_type))        
    sigma=[sgmR0,sgmR1]
    
    
    TbR=np.zeros(n_Vars)
    if (sgmR0*sgmR1!=0):
      for lt in range(nlt):
        TbR[Vars_of_type[lt]]=np.random.normal(0,sigma[lt],nvt[lt])
        
        
    Tb=np.zeros(n_Vars)
    k=.2*Context['Param']['DX']/12500.0
    #sh=Context['Param']['DX']/12500.0/2
    
    for v in Vars_of_type[0]:
        Tb[v]=mu0+sgm0*np.sqrt(2)*np.sin((Context['Dict_Vars']['Var'][v]['Coords'][0][1]+.5)*k - (Context['Dict_Vars']['Var'][v]['Coords'][0][0]+.5)*k)
    for v in Vars_of_type[1]:
        Tb[v]=mu1+sgm1*np.sqrt(2)*np.sin((Context['Dict_Vars']['Var'][v]['Coords'][0][1]+.5)*k + (Context['Dict_Vars']['Var'][v]['Coords'][0][0]+.5)*k)

  
    return Tb+TbR

def Simulate_PMW(TbH,TbV,Context,Observation,Obs_error_std=0):
    Wt=Observation['Wt']
    n_ell=Observation['Tb']['H'].shape[0]
    if (Obs_error_std==0): #no error
        Observation['Tb']['H']=Wt.dot(TbH)
        Observation['Tb']['V']=Wt.dot(TbV)
    else:                  #error ~ N(0,Obs_error_std)  
        Observation['Tb']['H']=Wt.dot(TbH)+np.random.normal(0,Obs_error_std,n_ell)
        Observation['Tb']['V']=Wt.dot(TbV)+np.random.normal(0,Obs_error_std,n_ell)
        
    print ("Computing means for each land type by lsqr")
    M=Observation['LandPropEll']
    Sh=lsqr(M,Observation['Tb']['H'])
    Sv=lsqr(M,Observation['Tb']['V'])
    
    print ("  Sol H (mu for each land type, LSQR):", Sh[0])
    print ("  Sol V (mu for each land type, LSQR):", Sv[0])
    print("   LandType approximation error:%.2f, %.2f"%(Observation['LSQRSols']['H']['norm'],Observation['LSQRSols']['V']['norm']))
    
    Observation['LSQRSols']={'H':{'Sol':Sh[0],'itn':Sh[2],'norm':Sh[3]},'V':{'Sol':Sv[0],'itn':Sv[2],'norm':Sv[3]}}    
        
    return Observation


def Simulate_NoPol_PMW(K,Tb,Obs_error_std=0):
    n_ell=K.shape[0]
    if (Obs_error_std==0): #no error
        Tb_sim=K.dot(Tb)
    else:                  #error ~ N(0,Obs_error_std)  
        Tb_sim=K.dot(Tb)+np.random.normal(0,Obs_error_std,n_ell)        
    return Tb_sim

    
def RMSE(Tb, Sol):
    return np.sqrt(((Tb-Sol)*(Tb-Sol)).mean())

def CC(Tb, Sol):
    return np.corrcoef(Tb,Sol)[0,1]
    
def COMP_COND(Tb, Sol, Context, Margin=None, Regular=None, LT=None):
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
    
    return RMSE(Tb[Cells], Sol[Cells]),CC(Tb[Cells], Sol[Cells])

def Set_Margins(Context):
    nvars=Context['Dict_Vars']['n_Vars']
    LTG=Context['Dict_Grids']['LandType_Grid']
    DX=Context['Param']['DX']
    dx=Context['Param']['dx']
    nx=DX/dx
    NX=LTG.shape[0]/nx #grid dimension in X
    NY=LTG.shape[1]/nx #grid dim in Y
    #Margin?
    nm=int(np.round(Context['Param']['WA_margin']/Context['Param']['DX'])) #number of margin cells
    Margin=np.zeros(nvars,dtype=bool)
    for vi in  Context['Dict_Vars']['Var']:
        v=Context['Dict_Vars']['Var'][vi]
        Coord=v['Coords']
        
        for i in range(len(Coord)):
            if ((Coord[i][0]<nm)  or (Coord[i][0]>=NX-nm)) or ((Coord[i][1]<nm)  or (Coord[i][1]>=NY-nm)):
               Margin[vi]=True
            
    Context['Dict_Vars']['Margin']=Margin
    
    #Regular cell?
    RegularArea=Context['Param']['DX']*Context['Param']['DY']
    Areas=[Context['Dict_Vars']['Var'][i]['Area'] for i in range(Context['Dict_Vars']['n_Vars'])]
    RegularCell=[Areas[i]==RegularArea for i in range(nvars)]
    Context['Dict_Vars']['SqCell']=RegularCell
