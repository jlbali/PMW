## -*- coding: utf-8 -*-
#"""
#Created on Wed Jun 20 14:13:40 2018
#
#@author: rgrimson
#"""
# -*- coding: utf-8 -*-

import my_base_dir
BASE_DIR = my_base_dir.get()
csv_dir=BASE_DIR + 'Output/'
csv_name ='corrida_noise'
import matplotlib.pyplot as plt

#%%

import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(csv_dir+csv_name+'.csv')
dg=df.groupby(['Method','Band','Pol','Obs_std_error']).mean()['RMSE']

#%%
c={'BGF':'b', 'LSQR':'g', 'EM':'k'}
ls={'BGF':'-.', 'LSQR':'--', 'EM':'-'}
label={'BGF':'BG', 'LSQR':'LSQR', 'EM':'STAT'}
plt.figure(figsize=(6.5, 3.5))
#plt.figure(figsize=(8, 5))
plt.clf()
Band='K'
Pol='V'
fs1=15
fs2=fs1
Methods=['EM','LSQR','BGF']
for Method in Methods:
        #for Band in ['X','K']:
        #for Pol in ['H','V']:
            RMSE=dg[Method,Band, Pol]
            plt.plot(RMSE,label=label[Method], c=c[Method],linewidth=2.0,linestyle=ls[Method])
            plt.xlim(0.0, 25.0)
            plt.ylim(0.0, 12.0)
            plt.xlabel(r'$\sigma_{Obs}$',fontsize=fs1)
            plt.ylabel("RMSE",fontsize=fs2)
            #plt.title("Influence of observational error on r'$\alpha > \beta$')
            plt.legend(loc='upper left')
#%%
plt.figure(2)
plt.clf()
Band='X'
Pol='H'
for Method in Methods:
        #for Band in ['X','K']:
        #for Pol in ['H','V']:
            RMSE=dg[Method,Band, Pol]
            plt.plot(RMSE,label=label[Method], c=c[Method],linewidth=2.0,linestyle=ls[Method])
            plt.xlim(0.0, 25.0)
            plt.ylim(0.0, 15.0)
            plt.xlabel(r'$\sigma_{Obs}$',fontsize=fs1)
            plt.ylabel("RMSE",fontsize=fs2)
            plt.legend(loc='upper left')

      
#%%      
(df['Band']==Band)&(df['Pol']=Pol)

df['RMSE']

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
