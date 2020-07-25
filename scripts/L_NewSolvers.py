# -*- coding: utf-8 -*-

"""
Solvers simplificados, que solamente reciben una imagen y omiten todo el quilombo de la polarizacion.
Esto es importante para el nuevo paper.
"""


from __future__ import print_function

from __future__ import division
import numpy as np
import scipy.linalg as linalg
import numpy.random as random

from scipy.sparse.linalg import lsqr
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
from itertools import product
import L_ParamEst
import copy
import sys
import logging
import pickle
import sys



#%%############################################################################
#%                                                                           %#
#%                                                                           %#
#%   Solver LSQR                                                             %#
#%                                                                           %#
##%############################################################################
def Solve_LSQR(K, Tb):
        #obsstd=1
        #print ("Solving System |", end=" ")
        S=lsqr(K,Tb)
        Solu=S[0]  
        return Solu



#%%############################################################################
#%                                                                           %#
#%                                                                           %#
#%   Solver       RODGERS                                                    %#
#%                                                                           %#
##%############################################################################

        
def Rodgers(VType, K, Tb, tita, obsstd=1.0): #compute solution for one pol
        n_ells=K.shape[0]
        n_Vars=K.shape[1]
        sigma2=tita['sigma2']
        mu=tita['mu']
        Sx=np.zeros([n_Vars])
        x0=np.zeros(n_Vars)
        for v in range(n_Vars):
            Sx[v]=abs(sigma2[VType[v]])
            x0[v]=mu[VType[v]]
        I=np.eye(n_ells)
        Sy=(K*Sx).dot(K.transpose()) + obsstd*obsstd*I 
        dif=Tb-K.dot(x0)
        v=np.linalg.solve(Sy,dif)
        u=Sx*((K.transpose()).dot(v))
        x=x0+u
        return x
            
def recompute_param(Sol,VType):    
    #VType=Observation['VType']
    nlt=VType.max()+1
    tita = {'mu': np.zeros(nlt), 'sigma2': np.zeros(nlt)}
    for lt in range(nlt):
        LT_Vars=np.where(VType==lt)[0]
        Tb_LT_Vars=Sol[LT_Vars]
        tita['sigma2'][lt] = Tb_LT_Vars.var()
        tita['mu'][lt] = Tb_LT_Vars.mean()
    return tita


def recompute_param_var(Sol,VType,var=100):    
    #VType=Observation['VType']
    nlt=VType.max()+1
    tita = {'mu': np.zeros(nlt), 'sigma2': np.zeros(nlt)}
    for lt in range(nlt):
        LT_Vars=np.where(VType==lt)[0]
        Tb_LT_Vars=Sol[LT_Vars]
        tita['sigma2'][lt] = 100
        tita['mu'][lt] = Tb_LT_Vars.mean()
    return tita


def Solve_Rodgers_IT(VType,K,Tb,obsstd=1.0, tol=0.005, max_iter=50): #compute solution iterating - NO BOUND
    Sol_initial = Solve_LSQR(K, Tb)
    tita = recompute_param_var(Sol_initial, VType)
    #print("\nIterating Rodgers... error std:%.2f"%(obsstd))
    again=True
    i=0
    while again:
        Sol=Rodgers(VType, K, Tb, tita, 'H', obsstd=obsstd)
        tita_old=copy.deepcopy(tita)
        tita=L_ParamEst.recompute_param(Sol,VType)
        #tita=L_ParamEst.minSigma2(tita,var_obs_land)
        #L_ParamEst.pr(tita,"it %d"%(i+1))
        e=np.max(np.abs(np.sqrt(tita['sigma2'])-np.sqrt(tita_old['sigma2'])))
        #again?
        i+=1
        if (i>=max_iter):
            again=False
                 
        if e<tol:
            again=False
            #print("- converged " %e, end=' ')
        sys.stdout.flush()
            
        
    #print('- %dit -'%i)
    #print error norm
    #dif=Tb-K.dot(Sol)
    #sn=np.sqrt(Tb.shape[0])
    #rmse=np.linalg.norm(dif)/sn
    sys.stdout.flush()
    return Sol



#%%############################################################################
#%                                                                           %#
#%                                                                           %#
#%   EM                                                                      %#
#%                                                                           %#
##%############################################################################

    

def EM_get_posteriors_parameters(y, K, mu_t, sigma_t, obsstd=1.0):
    obsvar = obsstd**2
    sigma_pos_inv = np.diag(1.0 / sigma_t) + (1.0/obsvar)*K.T.dot(K)
    Tb = y
    I=np.eye(K.shape[0])
    Sy=(K*sigma_t).dot(K.transpose()) + obsstd*obsstd*I 
    dif=Tb-K.dot(mu_t)
    v=np.linalg.solve(Sy,dif)
    u=sigma_t*((K.transpose()).dot(v))
    x=mu_t+u
    return x, sigma_pos_inv

# Se estanca con sigmas muy altos...
# Los mues son tambien medio altos.
# Va alternandose entre resultados muy malos y otros mediocres.



#%%############################################################################
#%                                                                           %#
#%                                                                           %#
#%   EM-MAP                                                                  %#
#%                                                                           %#
##%############################################################################

# Aproximando la inversa con series de Neumann.
def approx_inverse(A, n):
    B = np.eye(A.shape[0]) - A
    accum = np.eye(A.shape[0])
    prev_B = np.eye(A.shape[0])
    for i in range(1,n+1):
        prev_B = prev_B.dot(B)
        accum = accum + prev_B
    return accum

def test_approx_inverse():
    A = np.array([[4.0, 2.0], [7.0,-2.0]])*1e-7
    A_inv = np.linalg.inv(A)
    A_inv_approx = approx_inverse(A,10000)
    #print("A_inverse: ", A_inv)
    #print("A approx inverse: ", A_inv_approx)

#test_approx_inverse() # Parece andar horrible.


def Solve_EM_MAP(VType, K, Tb, obsstd=1.0,tol=0.005, max_iter=50, omegas=None, sigma2_clis=None, cholesky_inv=False):
    #print("Solving EM...")
    nlt=VType.max() + 1
    if omegas == None or sigma2_clis == None:
        #print("NO HAY PRIOR, usando ML")
        omegas = np.ones(nlt)
    Sol = Solve_LSQR(K,Tb)
    tita = recompute_param_var(Sol, VType)

    old_mus = np.zeros(nlt)
    old_sigma2 = np.zeros(nlt)
    for lt in range(nlt):
        old_mus[lt] = tita['mu'][lt]
        old_sigma2[lt] = tita['sigma2'][lt]

    #print("\nIterating EM... error std:%.2f"%(obsstd))

    again=True
    it=0
    n = K.shape[1]
    y = Tb
    sigmas = []
    sigmas.append(old_sigma2)
    mus = []
    mus.append(old_mus)
    while again:
        #print("Iteracion ", it)
        #print("Mus por landtype: ", old_mus)
        #print("Sigma2 por landtype: ", old_sigma2)
        it = it + 1
        # Paso 1, armamos las matrices mu_t y sigma_t
        mu_t = np.zeros(n)
        sigma_t = np.zeros(n)
        for i in range(n):
            mu_t[i] = old_mus[VType[i]]
            sigma_t[i] = old_sigma2[VType[i]]
        # Paso 2, obtenemos los posteriors.
        mu_pos, sigma_pos_inv = EM_get_posteriors_parameters(y, K, mu_t,sigma_t, obsstd)
        # Paso 3. Resolvemos los mu_k.
        new_mus = np.zeros(nlt)
        for lt in range(nlt):
            LT_Vars = np.where(VType == lt)[0]
            new_mus[lt] = np.mean(mu_pos[LT_Vars])
        # Paso 4. Resolvemos los sigma2_k
        new_sigma2 = np.zeros(nlt)
        for lt in range(nlt):
            # Computamos la diferencia de mues.
            LT_Vars = np.where(VType == lt)[0]
            nk = len(LT_Vars)
            mu_pos_lt = mu_pos[LT_Vars]
            M2 = np.sum( (mu_pos_lt - new_mus[lt])**2)
            if cholesky_inv:
                U = linalg.cholesky(sigma_pos_inv)
                U_inv = linalg.inv(U)
                sigma_pos = U_inv.dot(U_inv.T)
                M1 = np.diag(sigma_pos)[LT_Vars].sum()
            else:
                M1 = (np.diag(L_ParamEst.InvSym(sigma_pos_inv))[LT_Vars]).sum()
            M = M1 + M2
            new_sigma2[lt] =(M / nk)*omegas[lt] + (1-omegas[lt])*sigma2_clis[lt]
        #print("OldSigmas: ", old_sigma2)
        #print("NewSigma2: ", new_sigma2)
        sys.stdout.flush()
        e= np.max(np.abs(np.sqrt(new_sigma2)-np.sqrt(old_sigma2)))
        if (it>=max_iter):
            again=False
        if e<tol:
            again=False
            #print("- converged " %e, end=' ')
        sys.stdout.flush()
            
        old_mus = new_mus
        old_sigma2 = new_sigma2
        sigmas.append(old_sigma2)
        mus.append(old_mus)
        
    mu_t = np.zeros(n)
    sigma_t = np.zeros(n)
    for i in range(n):
        mu_t[i] = old_mus[VType[i]]
        sigma_t[i] = old_sigma2[VType[i]]
    mu_pos, sigma_pos_inv = EM_get_posteriors_parameters(y, K, mu_t,sigma_t, obsstd)
    return mu_pos



#%%############################################################################
#%                                                                           %#
#%                                                                           %#
#%   Solver       Tichonov                                                   %#
#%                                                                           %#
##%############################################################################

# Auxiliares...
def grillaLogaritmica(base, tope, cantidad):
    baseLog = np.log(base)
    topeLog = np.log(tope)
    grillaLog = np.linspace(baseLog,topeLog, cantidad)
    return np.exp(grillaLog)    


def constructH(K, neighbors):
    cantVars = K.shape[1]
    H = np.zeros((cantVars, cantVars))
    for i in range(cantVars):
        H[i,i] = 1.0
        localNeighbors = neighbors[i]
        for neighbor in localNeighbors:
            H[i, neighbor] = -1.0 / float(len(localNeighbors))
    return H

# Solvers Tichonov...
def Solve_Tichonov(VType, K, Tb,lamb, H):
    #print ("Solving System |")
    nlt=VType.max()+1
    Vars_of_type=[]
    for lt in range(nlt):
        Vars_of_type.append(np.where(VType==lt)[0])        
    KH = lamb*H
    Solu = L_ParamEst.InvSym(K.T.dot(K) + KH.T.dot(KH)).dot(K.T).dot(Tb)        
    return Solu

def Solve_GCV_Tichonov_OLD(VType, K, Tb, neighbors, gridLambs):
    H = constructH(K, neighbors)
    cantElipses = K.shape[0]
    Asharps = []
    #print("Precomputing A#...")
    for lamb in gridLambs:
        #print("For lambda = ", lamb)
        Asharps.append(L_ParamEst.InvSym(K.T.dot(K) + (lamb**2)*H.T.dot(H)).dot(K.T))
    CV_values = []
    for i in range(len(gridLambs)):
        lamb = gridLambs[i]
        Asharp = Asharps[i]
        KH = lamb*H
        x = L_ParamEst.InvSym(K.T.dot(K) + KH.T.dot(KH)).dot(K.T).dot(Tb)
        I = np.eye(cantElipses)
        value = cantElipses*(linalg.norm(K.dot(x) - Tb))**2 / (np.trace(I - K.dot(Asharp)))**2
        #print("Lambda: ", lamb, " with GCV value: ", value)
        CV_values.append(value)
    lambOptimo = gridLambs[np.argmin(CV_values)]
    #print("Lambda optimo elegido: " + str(lambOptimo))
    KH = lambOptimo*H
    Solu = L_ParamEst.InvSym(K.T.dot(K) + KH.T.dot(KH)).dot(K.T).dot(Tb)        
    return Solu

def Solve_GCV_Tichonov(VType, K, Tb, neighbors, gridLambs):
    H = constructH(K, neighbors)
    cantElipses = K.shape[0]
    CV_values = []
    for i in range(len(gridLambs)):
        lamb = gridLambs[i]
        Asharp = L_ParamEst.InvSym(K.T.dot(K) + (lamb**2)*H.T.dot(H)).dot(K.T)
        KH = lamb*H
        x = Asharp.dot(Tb)
        I = np.eye(cantElipses)
        value = cantElipses*(linalg.norm(K.dot(x) - Tb))**2 / (np.trace(I - K.dot(Asharp)))**2
        CV_values.append(value)
    lambOptimo = gridLambs[np.argmin(CV_values)]
    print("Lambda optimo elegido: " + str(lambOptimo))
    KH = lambOptimo*H
    Solu = L_ParamEst.InvSym(K.T.dot(K) + KH.T.dot(KH)).dot(K.T).dot(Tb)        
    return Solu




#%%############################################################################
#%                                                                           %#
#%                                                                           %#
#%   SOLVE                                                                   %#
#%                                                                           %#
##%############################################################################
def Solve(VType, K, Tb,Method, extras):
    
    if Method=='LSQR':
        Sol=Solve_LSQR(K, Tb)
    elif Method=='Rodgers_IT': #Iterative version of Rodgers
        obsstd = extras['obsstd']
        Sol=Solve_Rodgers_IT(VType, K, Tb,obsstd=obsstd)
    elif Method=='EM': #Iterative version of Rodgers
        omega = 1.0 # Anulamos el a priori por completo.
        sigma2_cli = 100
        obsstd = extras['obsstd']
        Sol=Solve_EM_MAP(VType, K, Tb,obsstd=obsstd,max_iter=50, omegas = [omega,omega], sigma2_clis=[sigma2_cli, sigma2_cli])
    elif Method == 'GCV_Tichonov':
        #print("Applying GCV-Tichonov method...")
        #gridLambs = grillaLogaritmica(1e-6, 100.0, 9)
        #gridLambs = grillaLogaritmica(1e-1, 1e1, 10)
        gridLambs = grillaLogaritmica(1e-3, 1e1, 20)
        neighbors = extras["neighbors"]
        Sol = Solve_GCV_Tichonov(VType, K, Tb, neighbors, gridLambs)
    else:
        #print("##################################\nUnkown method '%s', not solving!" %(Method))
        return None
    return Sol



