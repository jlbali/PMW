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

df = pd.read_csv('/home/rgrimson/Projects/PMW_Code/Scripts/PaperV2/Transecta/points_values_line.csv')
costax=-56.9776
mx=0
Mx=-26
plt.figure(figsize=(8, 3))
plt.clf()
plt.plot(df['xcoord'].values[mx:Mx],df['EM_H_KA_25'].values[mx:Mx],label='STAT', linewidth=2.0,c='b',linestyle='-')
plt.plot(df['xcoord'].values[mx:Mx],df['F16_37HD'].values[mx:Mx],label='GRD',  linewidth=2.0,c='g',linestyle='--')
plt.plot([costax,costax],[120,280],linewidth=3.0,c='k')
#plt.ylim(0.0, 10.0)
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
