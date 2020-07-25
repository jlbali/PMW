# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 10:37:27 2018

@author: rgrimson
"""

import numpy as np
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


HDFfilename='GW1AM2_201207310439_227D_L1SGBTBR_2210210'
Band='KA'
Context=L_Context.Load_Context(wdir)
L_Syn_Img.Set_Margins(Context)
Observation, tita=L_ReadObs.Load_Band_Kernels(HDFfilename,Context,Band)
np.random.seed(1)
TbH = L_Syn_Img.Create_Random_Synth_Img(Context)
np.random.seed(1)
TbV = L_Syn_Img.Create_Trig_Synth_Img(Context)
Observation=L_Syn_Img.Simulate_PMW(TbH,TbV,Context,Observation,Obs_error_std=1.0)


X=[]
Y=[]
MaxObs=4
w=0.001

def BG(gamma=0):
    print("Computing %.5f"%gamma)
    X.append(gamma)

    BG_Coef=L_Solvers.BG_Precomp_MaxObs(Context, Observation, gamma=gamma,w=w,MaxObs=MaxObs)
    Sol=[BG_Coef.dot(Observation['Tb']['H']), BG_Coef.dot(Observation['Tb']['V'])]
    RMSE=L_Syn_Img.RMSE_M(TbH,Sol[0],Context)
    print("RMSE: %.5f\n*****************************************"%RMSE)        
    Y.append(RMSE)

    return RMSE
    #return L_Syn_Img.RMSE_M(TbV,Sol[1],Context)
    #return -(gamma-.1)*(gamma-.3)        
    
    
XI=0
XD=np.pi/2
XC=np.pi/4

YI=BG(gamma=XI)
YD=BG(gamma=XD)
YC=BG(gamma=XC)
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
            Yn=BG(gamma=Xn)
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
            Yn=BG(gamma=Xn)
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
            YC=BG(gamma=XC)
        else:
            print("a izquierda")
            XD=XC
            XC=(XI+XC)/2
            YD=YC
            YC=BG(gamma=XC)
            
    D=np.max([np.abs(YI-YC),np.abs(YD-YC),np.abs(XI-XC),np.abs(XD-XC)])
    if D<0.001: 
        End=True
        
        
        
# Banda KA 