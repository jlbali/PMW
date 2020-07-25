# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 10:37:27 2018

@author: rgrimson
"""

import numpy as np

def BG(gamma=0):
    print(gamma)
    return (gamma-1)*(gamma-.5)        
    
    
XI=0
XD=np.pi/2
XC=np.pi/4

YI=BG(gamma=XI)
YD=BG(gamma=XI)
YC=BG(gamma=XI)
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
    