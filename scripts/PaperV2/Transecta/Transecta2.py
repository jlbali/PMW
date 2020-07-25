## -*- coding: utf-8 -*-
#"""
#Created on Wed Jun 20 14:13:40 2018
#
#@author: rgrimson
#"""

#%%

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#df = pd.read_csv('/home/rgrimson/Projects/PMW_Code/Scripts/PaperV2/Transecta/points_values_line.csv')
#df = pd.read_csv('/home/rgrimson/Projects/PMW_Repo/Scripts/PaperV2/Transecta/points_values_line.csv')
df = pd.read_csv('/home/rgrimson/Projects/PMW_Repo/Scripts/PaperV2/Transecta/points_values_all_km.csv')
c1='EM_HKA_125,N,21,5' #'EM_H_KA_12'
c2='2516_37HD,N,21,5'  #'EM_H_KA_25'
c3='31251637HD,N,21,5' #'F16_37HD'
cx='xcoord,N,24,15'    #'xcoord'

#costax=-56.9776
#xc=df[cx].values[mx:Mx]
#k=(xc<costax)*2+(xc>costax)*-3
k= 0 
mx=45
Mx=-26
c=-mx
plt.figure(figsize=(8, 3))
plt.clf()
plt.plot(df[cx].values[mx:Mx],df[c1].values[mx:Mx],label='STAT (12.5km)', linewidth=2.0,c='b',linestyle='-')
#plt.plot(df[cx].values[mx:Mx],df[c2].values[mx:Mx],label='GRD (25km)', linewidth=2.0,c='g',linestyle='--')
plt.plot(df[cx].values[mx:Mx],df[c3].values[mx+c:Mx+c]+k,label='SIR (3.125km)',  linewidth=2.0,c='g',linestyle='--')
plt.plot([costax,costax],[120,280],linewidth=3.0,c='k')
plt.xlim(-60.8,-55.1)
plt.legend(loc='lower left')
#plt.legend(loc='upper right')
plt.xlabel("Longitude")
plt.ylabel("Brightness Temperature")
#plt.show()
#%
plt.savefig('transectas_2.png', dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches='tight', pad_inches=0.1,
        frameon=None)



#%%

##%%
#df = pd.read_csv('~/Downloads/graphics/points_values_land.csv')
#df = pd.read_csv('~/Downloads/graphics/points_values_line.csv')
##df = pd.read_csv('~/Downloads/graphics/points_values_river.csv') 
#
#
#X='deg'
#plt.figure(1)
#plt.clf()
#Y1='EM_H_KA_25'
#Y2='F16_37HD'
#plt.plot(df[X].values,df[Y1].values,label='Y1',  alpha=0.5,linewidth=2.0,c='g')
#plt.plot(df[X].values,df[Y2].values,label='Y2',  alpha=0.5,linewidth=2.0,c='b')
#plt.ylim([100,270])
#plt.figure(2)
#plt.clf()
#Y1='EM_H_KU_25'
#Y2='F16_19HD'
#plt.plot(df[X].values,df[Y1].values,label='Y1',  alpha=0.5,linewidth=2.0,c='g')
#plt.plot(df[X].values,df[Y2].values,label='Y2',  alpha=0.5,linewidth=2.0,c='b')
#plt.ylim([90,270])
