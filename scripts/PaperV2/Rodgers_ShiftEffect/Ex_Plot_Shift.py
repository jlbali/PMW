# -*- coding: utf-8 -*-



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time



data_KA = np.loadtxt("Shift_EM_KA.csv", delimiter=",")
data_KU = np.loadtxt("Shift_EM_KU.csv", delimiter=",")
data_C = np.loadtxt("Shift_EM_C.csv", delimiter=",")
data_K = np.loadtxt("Shift_EM_K.csv", delimiter=",")
data_X = np.loadtxt("Shift_EM_X.csv", delimiter=",")

x = data_KA[:, 0]/1000
RMSE_KA = data_KA[:,2]
RMSE_KU = data_KU[:,2]
RMSE_C = data_C[:,2]
RMSE_K = data_K[:,2]
RMSE_X = data_X[:,2]


#%%
plt.figure(figsize=(6.5, 3.5))

plt.plot(x, RMSE_KA,  '-', label="Ka", c='k',linewidth=2)
plt.plot(x, RMSE_KU, '--', label="Ku", c='k',linewidth=2)
plt.plot(x, RMSE_K, ':', label="K", c='k',linewidth=2)
plt.plot(x, RMSE_X, ':', label="X", linewidth=5.0, c='k')
plt.plot(x, RMSE_C, '-.', label="C", c='k',linewidth=2)

plt.ylim(-0, 5.8)
plt.legend(loc='upper center', ncol=5)
plt.xlabel("Shift in kilometers")
plt.ylabel("RMSE")
#plt.show()
#%
plt.savefig('Shift.png', dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches='tight', pad_inches=0.1,
        frameon=None)
#%%

#data_KA = np.loadtxt("Shift_LSQR_KA.csv", delimiter=",")
#data_KU = np.loadtxt("Shift_LSQR_KU.csv", delimiter=",")
#data_C = np.loadtxt("Shift_LSQR_C.csv", delimiter=",")

#x = data_KA[:, 0]
#RMSE_KA = data_KA[:,1]
#RMSE_KU = data_KU[:,1]
#RMSE_C = data_C[:,1]



#plt.plot(x, RMSE_KA, label="Ka Band")
#plt.plot(x, RMSE_KU, label="Ku Band")
#plt.plot(x, RMSE_C, label="C Band")
#plt.ylim(0.0, 9.0)
#plt.legend(loc='upper left')
#plt.xlabel("Shift in meters")
#plt.ylabel("RMSE")
#plt.show()



