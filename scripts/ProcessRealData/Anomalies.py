# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import L_Files
import numpy as np
import matplotlib.pyplot as plt
import L_Context
import L_Kernels as L_Kernels

import my_base_dir
BASE_DIR = my_base_dir.get()
MAPA='Ajo_10km'
wdir=BASE_DIR + 'Mapas/%s/'%MAPA

Context=L_Context.Load_Context(wdir)
IDGrid=Context['Dict_Grids']['CellID_Grid']

#%%
wdirG=wdir+'/Out_Grids/'
FileList=L_Files.listFilesWithExtension(wdirG,'.csv')
#%%

frame = pd.DataFrame()
for CSVfilename in FileList:
    df=pd.read_csv(wdirG+CSVfilename+'.csv', sep=',', delimiter=None, header='infer', names=None)
    frame[CSVfilename[15:32]]=df['IP']
#%%
ncells=frame.shape[0]
nimgs=frame.shape[1]
meanIP=np.zeros(ncells)
for i in range(ncells):
    meanIP[i]=np.array(frame.iloc[[i]]).mean()

#%%

df = pd.DataFrame()
for k in frame.keys():
    df[k]=frame[k]-meanIP

#%%
W=[]
for i in range(ncells):
    W.append(np.where(IDGrid==i))
    
out_img=np.zeros(IDGrid.shape)    
#%%
for k in df.keys():
    out_fn = wdirG+'Anomalies/'+k+'_IP_anom'
    for i in range(ncells):
        out_img[W[i]]=df[k][i]
    plt.imshow(out_img[:,::-1].T,interpolation='none',vmin=-.1,vmax=.1,cmap=plt.get_cmap('gist_rainbow'))
    plt.savefig(out_fn+'.jpg',bbox_inches='tight')
