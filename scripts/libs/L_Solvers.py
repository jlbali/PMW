#Tychonov no invertir la matrix
#Invertir EN la matrix
#Vertir la matrz
#ir a la matrix
#La matrix

# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 13:47:57 2017

@author: rgrimson
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
import libs.L_ParamEst as L_ParamEst
import copy
import sys
import logging
import pickle
import sys

#%%############################################################################
#%                                                                           %#
#%                                                                           %#
#%   Solver       Weights                                                    %#
#%                                                                           %#
##%############################################################################
def Solve_Weights(Context, Observation):
    Wt=Observation['Wt']
    n_Vars=Context['Dict_Vars']['n_Vars']
    CIdG=Context['Dict_Grids']['CellID_Grid']

    SolH=np.zeros(n_Vars)
    SolV=np.zeros(n_Vars)
    for w in range(n_Vars):
        W=np.where(CIdG==w)
        N=len(W[0])
        SolH[w]=Observation['Img']['H'][W].sum()/N
        SolV[w]=Observation['Img']['V'][W].sum()/N
    return [SolH,SolV]        
    #return [Wt.transpose().dot(Observation['Tb']['H'])/(Wt.sum(axis=0)),Wt.transpose().dot(Observation['Tb']['V'])/(Wt.sum(axis=0))]
    # Como distingue el LandType este esquema? No le importa, aunque las celdas ya lo tienen distinguido.


def Solve_SimpleWeights(Observation, pol):
    K=Observation['Wt']
    y=Observation['Tb'][pol] 
    cantElipses = K.shape[0]
    cantCells = K.shape[1]
    x = []
    for i in range(cantCells):
        accumValue = 0.0
        accumWeights = 0.0
        for j in range(cantElipses):
            accumValue += y[j]*K[j,i]
            accumWeights += K[j,i]
        if accumWeights != 0.0:
            x.append(accumValue/accumWeights)
        else:
            x.append(0.0)
    return x

#%%############################################################################
#%                                                                           %#
#%                                                                           %#
#%   Solver LSQR                                                             %#
#%                                                                           %#
##%############################################################################
def Solve_LSQR(Observation):
        #obsstd=1
        #print ("Solving System |", end=" ")
        Wt=Observation['Wt']
        VType=Observation['VType']
        #nlt=VType.max()+1
        Tb=Observation['Tb'] #'H' and 'V'
        

        Solu={}
        for pol in ['H','V']:
            S=lsqr(Wt,Tb[pol])
            Solu[pol]=S[0]
  
        #Solu['H'] = L_ParamEst.InvSym(Wt.T.dot(Wt)).dot(Wt.T).dot(Tb['H'])
        #Solu['V'] = L_ParamEst.InvSym(Wt.T.dot(Wt)).dot(Wt.T).dot(Tb['V'])
        return [Solu['H'],Solu['V']]

#%%############################################################################
#%                                                                           %#
#%                                                                           %#
#%   Precomputation for Backus - Gilbert  FINE - J=distance to centroid      %#
#%                                                                           %#
##%############################################################################
def BG_Precomp_MaxObs_Centroid(Context, Observation, gamma=np.pi/4,w=.1,error_std=1.0,MaxObs=10): #compute solution 
        #dx=Context['WA_Cords']['dx']
        #dy=Context['WA_Cords']['dy']
        Tb=Observation['Tb']
        K=Observation['Wt']
        VType=Observation['VType']
        #n_ells=K.shape[0]
        n_Vars=K.shape[1]
        n_Ells=K.shape[0]
        
        SolH=np.zeros(n_Vars)
        SolV=np.zeros(n_Vars)
        KGrid=Observation['KGrid']
        CIdG=Context['Dict_Grids']['CellID_Grid']
        LTG=Context['Dict_Grids']['LandType_Grid']##
        X=range(CIdG.shape[0])##
        Y=range(CIdG.shape[1])##
        M=np.mgrid[0:CIdG.shape[0],0:CIdG.shape[1]]##
        A=np.zeros([n_Vars,n_Ells])

        print('Precomputing coefficients for Backus Gilbert (fine, %d cells)'%n_Vars)
        print("gamma: %.3f, w: %.5f, error_std: %.3f "%(gamma,w,error_std))
        for ID in range(n_Vars):
            k=K[:,ID].copy()
            k.sort()
            tol=np.max([k[-MaxObs],0.001])

            SelElls=np.where(K[:,ID]>=tol)[0]
            n_sel_ells=SelElls.shape[0]
            if ID%50==0:
                print("%d (%d)"%(ID,n_sel_ells),end=' ')
                sys.stdout.flush()
            
            F=(CIdG==ID)*1
            F=F/F.sum()

            LT=VType[ID]##
            Xm=F.sum(axis=1).dot(X)##
            Ym=F.sum(axis=0).dot(Y)##
            J=((M[0]-Xm)**2+(M[1]-Ym)**2)*(CIdG!=ID)##

            G=np.zeros([n_sel_ells, n_sel_ells])
            for i in range(n_sel_ells):
                for j in np.arange(i,n_sel_ells,1):
                    G[i,j]=(KGrid[:,:,SelElls[i]]*KGrid[:,:,SelElls[j]]*J).sum() ##
                    G[j,i]=G[i,j]
            v=np.zeros(n_sel_ells)
            #for e in range(n_sel_ells):
            #    v[e]=(KGrid[:,:,e]*F*J).sum() ##
            u=np.ones(n_sel_ells)
            E=np.eye(n_sel_ells)*error_std**2
            V=G*np.cos(gamma)+np.sin(gamma)*E*w
            iVu=np.linalg.solve(V,u)
            iVv=np.linalg.solve(V,v)
            b=np.cos(gamma)*v+u*(1-np.cos(gamma)*u.dot(iVv))/(u.dot(iVu))
            a=np.linalg.solve(V,b)
            #show(KGrid[:,:,SelElls].dot(a))
            A[ID,SelElls]=a
        return A

#%%############################################################################
#%                                                                           %#
#%                                                                           %#
#%   Precomputation for Backus - Gilbert  FINE                               %#
#%                                                                           %#
##%############################################################################

def precompute_all_optimized(Context):
        n_Vars=Context['Dict_Vars']['n_Vars']
        CIdG=Context['Dict_Grids']['CellID_Grid']
        tx,ty=CIdG.shape
        #M=np.mgrid[0:CIdG.shape[0],0:CIdG.shape[1]]##
        #A=np.zeros([n_Vars,n_Ells])

        print('Precomputing distances for Backus Gilbert (fine, %d cells)'%n_Vars)
        #print("gamma: %.3f, w: %.5f, error_std: %.3f "%(gamma,w,error_std))
        distances_matrix_list = []
        J=CIdG*0
        for ID in range(n_Vars):
            print("Preprocesando ", ID, " de ", n_Vars)
            distances_matrix = np.zeros(J.shape)
            [Wx,Wy]=np.where(CIdG==ID)
            mx=Wx.min()
            Mx=Wx.max()
            my=Wy.min()
            My=Wy.max()
            cx=(mx+Mx)/2
            cy=(my+My)/2
            width=Mx-mx
            height=My-my
            area_celda = len(Wx)
            area_bbox = (width+1)*(height+1)            
            #print("Area de celda: ", area_celda, " Area de bbox: ", area_bbox)            
            if area_celda != area_bbox:
                # Es una celda irregular...
                print("Celda irregular, busqueda exhaustiva")                
                print("Area de celda: ", area_celda, " Area de bbox: ", area_bbox)
                borde=np.ones(len(Wx),dtype=bool) #estudiare solo los bordes de la celda
                for k in range(len(Wx)):
                    puntoX = Wx[k]
                    puntoY = Wy[k]
                    if (puntoX == 0) or (puntoX == tx-1) or (puntoY == 0) or (puntoY == ty-1):
                        pass
                    elif (CIdG[puntoX-1, puntoY] == ID) and (CIdG[puntoX, puntoY-1] == ID) and (CIdG[puntoX+1, puntoY] == ID) and (CIdG[puntoX, puntoY+1] == ID) :
                        borde[k]=False
                Wx2=Wx[borde]
                Wy2=Wy[borde]
                for i in range(tx):
                    print("Fila ", i)                
                    for j in range(ty):
                        if CIdG[i,j]==ID:
                            dmin=0
                        else:
                            dmin = None
                            for k in range(len(Wx2)):
                                puntoX = Wx2[k]
                                puntoY = Wy2[k]
                                d = (i - puntoX)**2 + (j-puntoY)**2
                                if dmin == None or d < dmin:
                                    dmin = d
                        #distances_matrix[i,j]=np.sqrt(dx**2 + dy**2)
                        distances_matrix[i,j]=dmin
            else:
                print("Celda regular")
                for i in range(tx):
                    for j in range(ty):
                        dx = np.max((0.0, np.abs(i-cx) - width/2.0))                    
                        dy = np.max((0.0, np.abs(j-cy) - height/2.0))                    
                        #distances_matrix[i,j]=np.sqrt(dx**2 + dy**2)
                        distances_matrix[i,j]=dx**2 + dy**2

            distances_matrix_list.append(distances_matrix)
            #if ID == 5:
            #    return distances_matrix_list
        return distances_matrix_list



def save_distances(path, distances):
    with open(path, 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
       pickle.dump(distances, f, pickle.HIGHEST_PROTOCOL)


def load_distances(path):
    with open(path, 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
        distances_matrix_list_recovered = pickle.load(f)
    return distances_matrix_list_recovered





def BG_Precomp_MaxObs_Cell(distances, Context, Observation, gamma=np.pi/4,w=.001,error_std=1.0,MaxObs=10): #compute solution 
        #dx=Context['WA_Cords']['dx']
        #dy=Context['WA_Cords']['dy']
        Tb=Observation['Tb']
        K=Observation['Wt'] # EN EL FONDO, NO PARECE QUE SE ESTE USANDO.
        VType=Observation['VType']
        #n_ells=K.shape[0]
        n_Vars=K.shape[1]
        n_Ells=K.shape[0]
        
        SolH=np.zeros(n_Vars)
        SolV=np.zeros(n_Vars)
        KGrid=Observation['KGrid']
        CIdG=Context['Dict_Grids']['CellID_Grid']
        LTG=Context['Dict_Grids']['LandType_Grid']##
        X=range(CIdG.shape[0])##
        Y=range(CIdG.shape[1])##
        M=np.mgrid[0:CIdG.shape[0],0:CIdG.shape[1]]##
        A=np.zeros([n_Vars,n_Ells])

        print('Precomputing coefficients for Backus Gilbert (fine, %d cells)'%n_Vars)
        print("gamma: %.3f, w: %.5f, error_std: %.3f "%(gamma,w,error_std))
        
        J=CIdG*0
        nx,ny=J.shape
        for ID in range(n_Vars):
            #print("Procesando ", ID)            
            k=K[:,ID].copy()
            k.sort() # NO PARECE ESTAR SIENDO USADO!
            tol=np.max([k[-MaxObs],0.001])

            SelElls=np.where(K[:,ID]>=tol)[0]
            n_sel_ells=SelElls.shape[0]
            if n_sel_ells:
                if ID%50==0:
                    print("%d (%d)"%(ID,n_sel_ells),end=' ')
                    sys.stdout.flush()
                else:
                    print(".",end='')
                    sys.stdout.flush()
                
                [Wx,Wy]=np.where(CIdG==ID)
                J = distances[ID]
                #dx = max(abs(px - x) - width / 2, 0);
                #dy = max(abs(py - y) - height / 2, 0);
                #return dx * dx + dy * dy;                        
    
    
                G=np.zeros([n_sel_ells, n_sel_ells])
                for i in range(n_sel_ells):
                    for j in np.arange(i,n_sel_ells,1):
                        G[i,j]=(KGrid[:,:,SelElls[i]]*KGrid[:,:,SelElls[j]]*J).sum() ##
                        G[j,i]=G[i,j]
                v=np.zeros(n_sel_ells)
                #for e in range(n_sel_ells):
                #    v[e]=(KGrid[:,:,e]*F*J).sum() ##
                u=np.ones(n_sel_ells)
                E=np.eye(n_sel_ells)*error_std**2
                V=G*np.cos(gamma)+np.sin(gamma)*E*w
                while not(np.linalg.matrix_rank(V) == V.shape[0]):
                    gamma=gamma+0.000000000000001/w
                    V=G*np.cos(gamma)+np.sin(gamma)*E*w
                    
                iVu=np.linalg.solve(V,u)
                iVv=np.linalg.solve(V,v)
                b=np.cos(gamma)*v+u*(1-np.cos(gamma)*u.dot(iVv))/(u.dot(iVu))
                a=np.linalg.solve(V,b)
                #show(KGrid[:,:,SelElls].dot(a))
            A[ID,SelElls]=a
        return A

#PLOT BG KERNELS
#n=50
#Observation['KGrid'].shape
#A.shape
#
#J = distances[n]
#J=J/J.max()
#
#A[n][A[n]!=0]
#
#W=A[n]
#K=Observation['KGrid'].dot(W)
#plt.imshow((.001*(CIdG==n)).transpose()+K.transpose())
#plt.figure()
#plt.imshow((J==0) + 2*(CIdG==n))


#%%############################################################################
#%                                                                           %#
#%                                                                           %#
#%   K-FOLD Backus-Gilbert                                                   %#
#%                                                                           %#
##%############################################################################

def BG_Step(distances, indices, pol,  Context, Observation, gamma=np.pi/4,w=.001,error_std=1.0,MaxObs=10): #compute solution 
        #dx=Context['WA_Cords']['dx']
        #dy=Context['WA_Cords']['dy']
        Tb=Observation['Tb'][pol][indices]
        K=Observation['Wt'][indices,:] # EN EL FONDO, NO PARECE QUE SE ESTE USANDO.
        #n_ells=K.shape[0]
        n_Vars=K.shape[1]
        n_Ells=K.shape[0]
        
        KGrid=Observation['KGrid']
        CIdG=Context['Dict_Grids']['CellID_Grid']
        #LTG=Context['Dict_Grids']['LandType_Grid']##
        #X=range(CIdG.shape[0])##
        #Y=range(CIdG.shape[1])##
        #M=np.mgrid[0:CIdG.shape[0],0:CIdG.shape[1]]##
        A=np.zeros([n_Vars,n_Ells])

        print('Precomputing coefficients for Backus Gilbert (fine, %d cells)'%n_Vars)
        print("gamma: %.3f, w: %.5f, error_std: %.3f "%(gamma,w,error_std))
        
        J=CIdG*0
        nx,ny=J.shape
        for ID in range(n_Vars):
            #print("Procesando ", ID)            
            k=K[:,ID].copy()
            k.sort() # NO PARECE ESTAR SIENDO USADO!
            tol=np.max([k[-MaxObs],0.001])

            SelElls=np.where(K[:,ID]>=tol)[0]
            n_sel_ells=SelElls.shape[0]
            if n_sel_ells:
                if ID%50==0:
                    print("%d (%d)"%(ID,n_sel_ells),end=' ')
                    sys.stdout.flush()
                else:
                    print(".",end='')
                    sys.stdout.flush()
                
                [Wx,Wy]=np.where(CIdG==ID)
                J = distances[ID]
                #dx = max(abs(px - x) - width / 2, 0);
                #dy = max(abs(py - y) - height / 2, 0);
                #return dx * dx + dy * dy;                        
    
    
                G=np.zeros([n_sel_ells, n_sel_ells])
                for i in range(n_sel_ells):
                    for j in np.arange(i,n_sel_ells,1):
                        G[i,j]=(KGrid[:,:,SelElls[i]]*KGrid[:,:,SelElls[j]]*J).sum() ##
                        G[j,i]=G[i,j]
                v=np.zeros(n_sel_ells)
                #for e in range(n_sel_ells):
                #    v[e]=(KGrid[:,:,e]*F*J).sum() ##
                u=np.ones(n_sel_ells)
                E=np.eye(n_sel_ells)*error_std**2
                V=G*np.cos(gamma)+np.sin(gamma)*E*w
                while not(np.linalg.matrix_rank(V) == V.shape[0]):
                    gamma=gamma+0.000000000000001/w
                    V=G*np.cos(gamma)+np.sin(gamma)*E*w
                    
                iVu=np.linalg.solve(V,u)
                iVv=np.linalg.solve(V,v)
                b=np.cos(gamma)*v+u*(1-np.cos(gamma)*u.dot(iVv))/(u.dot(iVu))
                a=np.linalg.solve(V,b)
                #show(KGrid[:,:,SelElls].dot(a))
            A[ID,SelElls]=a
        return A.dot(Tb)


# Con un gamma, calcula el error de K-Fold.
def BG_KFold_pol(distances, pol,  Context, Observation, folds, gamma=np.pi/4,w=.001,error_std=1.0,MaxObs=10): #compute solution 
    y = Observation['Tb'][pol]
    K=Observation['Wt']
    m = K.shape[0]
    error = 0.0
    for j in range(len(folds)):
        print("Fold ", j)
        test_indices = folds[j]
        train_indices = np.delete(np.array(range(m)), test_indices)
        x = BG_Step(distances, train_indices, pol,  Context, Observation, gamma,w,error_std,MaxObs)
        test_y = y[test_indices]
        test_K = K[test_indices,:]
        partial_error = (1.0/len(folds[j]))*((test_y - test_K.dot(x))**2).sum()
        print("Partial error: ", partial_error)
        error = error + partial_error
    return error


def BG_KFold_pol_complete(distances, pol, n_folds, Context, Observation, grilla_gamma, w=.001,error_std=1.0,MaxObs=10):
    # Armamos los folds.
    K=Observation['Wt']
    folds = []
    m = K.shape[0]
    bag = np.array(range(m))
    fold_size = int(m/n_folds)
    for i in range(n_folds-1):
        fold = random.choice(bag, size = fold_size, replace=False)
        bag = np.delete(bag, fold)
        folds.append(fold)
    if len(bag) > 0:
        folds.append(bag) # Adjuntamos el remanente del bag.
    errores = np.zeros(len(grilla_gamma))
    for i in range(len(grilla_gamma)):
        gamma = grilla_gamma[i]
        print("Procesando gamma: ", gamma)
        errores[i] = BG_KFold_pol(distances, pol,  Context, Observation, folds, gamma,w,error_std,MaxObs)
        print("Error obtenido: ", errores[i])
    min_gamma = grilla_gamma[np.argmin(errores)]
    print("Gano Gamma: ", min_gamma)
    return BG_Step(distances, range(m), pol,  Context, Observation, min_gamma,w,error_std,MaxObs)

def BG_KFold_complete(distances, n_folds, Context, Observation, grilla_gamma, w=.001,error_std=1.0,MaxObs=10):
    Sol = []
    Sol.append(BG_KFold_pol_complete(distances, 'H', n_folds, Context, Observation, grilla_gamma, w,error_std,MaxObs))
    Sol.append(BG_KFold_pol_complete(distances, 'V', n_folds, Context, Observation, grilla_gamma, w,error_std,MaxObs))
    return Sol



#%%############################################################################
#%                                                                           %#
#%                                                                           %#
#%  PRECOMP K-FOLD Backus-Gilbert                                            %#
#%                                                                           %#
##%############################################################################

def preComp_BG_Step(distances,  Context, Observation, gamma=np.pi/4,w=.001,error_std=1.0,MaxObs=10): #compute solution 
        #dx=Context['WA_Cords']['dx']
        #dy=Context['WA_Cords']['dy']
        K=Observation['Wt']
        #n_ells=K.shape[0]
        n_Vars=K.shape[1]
        n_Ells=K.shape[0]
        
        KGrid=Observation['KGrid']
        CIdG=Context['Dict_Grids']['CellID_Grid']
        #LTG=Context['Dict_Grids']['LandType_Grid']##
        #X=range(CIdG.shape[0])##
        #Y=range(CIdG.shape[1])##
        #M=np.mgrid[0:CIdG.shape[0],0:CIdG.shape[1]]##
        A=np.zeros([n_Vars,n_Ells])

        print('Precomputing coefficients for Backus Gilbert (fine, %d cells)'%n_Vars)
        print("gamma: %.3f, w: %.5f, error_std: %.3f "%(gamma,w,error_std))
        
        J=CIdG*0
        nx,ny=J.shape
        for ID in range(n_Vars):
            #print("Procesando ", ID)            
            k=K[:,ID].copy()
            k.sort() # NO PARECE ESTAR SIENDO USADO!
            tol=np.max([k[-MaxObs],0.001])

            SelElls=np.where(K[:,ID]>=tol)[0]
            n_sel_ells=SelElls.shape[0]
            if n_sel_ells:
                if ID%50==0:
                    print("%d (%d)"%(ID,n_sel_ells),end=' ')
                    sys.stdout.flush()
                else:
                    print(".",end='')
                    sys.stdout.flush()
                
                [Wx,Wy]=np.where(CIdG==ID)
                J = distances[ID]
                #dx = max(abs(px - x) - width / 2, 0);
                #dy = max(abs(py - y) - height / 2, 0);
                #return dx * dx + dy * dy;                        
    
    
                G=np.zeros([n_sel_ells, n_sel_ells])
                for i in range(n_sel_ells):
                    for j in np.arange(i,n_sel_ells,1):
                        G[i,j]=(KGrid[:,:,SelElls[i]]*KGrid[:,:,SelElls[j]]*J).sum() ##
                        G[j,i]=G[i,j]
                v=np.zeros(n_sel_ells)
                #for e in range(n_sel_ells):
                #    v[e]=(KGrid[:,:,e]*F*J).sum() ##
                u=np.ones(n_sel_ells)
                E=np.eye(n_sel_ells)*error_std**2
                V=G*np.cos(gamma)+np.sin(gamma)*E*w
                while not(np.linalg.matrix_rank(V) == V.shape[0]):
                    gamma=gamma+0.000000000000001/w
                    V=G*np.cos(gamma)+np.sin(gamma)*E*w
                    
                iVu=np.linalg.solve(V,u)
                iVv=np.linalg.solve(V,v)
                b=np.cos(gamma)*v+u*(1-np.cos(gamma)*u.dot(iVv))/(u.dot(iVu))
                a=np.linalg.solve(V,b)
                #show(KGrid[:,:,SelElls].dot(a))
            A[ID,SelElls]=a
        return A


def preComp_BG(path, mapStr, bandStr, distances,  Context, Observation, gammas,w=.001,error_std=1.0,MaxObs=10):
    for i in range(len(gammas)):
        print("Precomputando para ", i)
        gamma = gammas[i]
        A = preComp_BG_Step(distances, Context, Observation, gamma,w,error_std,MaxObs)
        filePath = path + "/BGC_" + mapStr + "_" + bandStr + "_" +  str(i) + ".pickle"
        with open(filePath, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
           pickle.dump(A, f, pickle.HIGHEST_PROTOCOL)


def preComp_BG(path, mapStr, bandStr, distances,  Context, Observation, gammas,w=.001,error_std=1.0,MaxObs=10):
    for i in range(len(gammas)):
        print("Precomputando para ", i)
        gamma = gammas[i]
        A = preComp_BG_Step(distances, Context, Observation, gamma,w,error_std,MaxObs)
        filePath = path + "/BGC_" + mapStr + "_" + bandStr + "_" +  str(i) + ".pickle"
        with open(filePath, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
           pickle.dump(A, f, pickle.HIGHEST_PROTOCOL)



def post_BG_Step(As, indices, pol, Observation, gamma_index):
    A = As[gamma_index][:,indices]
    Tb=Observation['Tb'][pol][indices]
    return A.dot(Tb)


def post_BG_KFold_pol(As, pol,  Observation, folds, gamma_index): #compute solution 
    y = Observation['Tb'][pol]
    K=Observation['Wt']
    m = K.shape[0]
    error = 0.0
    for j in range(len(folds)):
        print("Fold ", j)
        test_indices = folds[j]
        train_indices = np.delete(np.array(range(m)), test_indices)
        #print("Test indices: ", test_indices)
        #print("Train indices: ", train_indices)
        x = post_BG_Step(As, train_indices, pol, Observation, gamma_index)
        test_y = y[test_indices]
        test_K = K[test_indices,:]
        partial_error = (1.0/len(folds[j]))*((test_y - test_K.dot(x))**2).sum()
        print("Partial error: ", partial_error)
        error = error + partial_error
    return error


def post_BG_KFold_pol_complete(As, pol, n_folds,Observation, cant_gammas):
    # Armamos los folds.
    K=Observation['Wt']
    folds = []
    m = K.shape[0]
    bag = np.array(range(m))
    fold_size = int(m/n_folds)
    for i in range(n_folds-1):
        fold = random.choice(bag, size = fold_size, replace=False)
        bag = np.delete(bag, fold)
        folds.append(fold)
    if len(bag) > 0:
        folds.append(bag) # Adjuntamos el remanente del bag.
    errores = np.zeros(cant_gammas)
    for i in range(cant_gammas):
        gamma_index = i
        print("Procesando gamma index: ", gamma_index)
        errores[i] = post_BG_KFold_pol(As, pol,  Observation, folds, gamma_index)
        print("Error obtenido: ", errores[i])
    min_gamma_index = np.argmin(errores)
    print("Gano Gamma index: ", min_gamma_index)
    return post_BG_Step(As, range(m), pol, Observation, min_gamma_index)

def post_BG_KFold_complete(As, n_folds, Observation, cant_gammas):
    Sol = []
    Sol.append(post_BG_KFold_pol_complete(As, 'H', n_folds,Observation, cant_gammas))
    Sol.append(post_BG_KFold_pol_complete(As, 'V', n_folds,Observation, cant_gammas))
    return Sol


#%%############################################################################
#%                                                                           %#
#%                                                                           %#
#%   Solver       SIR                                                        %#
#%                                                                           %#
##%############################################################################


def SIR_single_pol(pol, Context,Observation,tita, max_iter=50):
    K=Observation['Wt']    
    y = Observation['Tb'][pol]
    n = K.shape[1]
    m = K.shape[0]
    nlt=Context['Dict_Grids']['nlt']
    VType=Observation['VType']
    tita = copy.deepcopy(tita)
    x = np.zeros(n)
    # Armando solucion inicial, tomando promedios por landtype segun LSQR.
    for lt in range(nlt):
        tita['H']['sigma2'][lt]=100 #a high var
        tita['V']['sigma2'][lt]=100 #a high var
        tita['H']['mu'][lt]=Observation['LSQRSols']['H']['Sol'][lt]
        tita['V']['mu'][lt]=Observation['LSQRSols']['V']['Sol'][lt]
        for i in range(n):
            if (VType[i] == lt):
                x[i] = tita[pol]['mu'][lt]
    for it in range(max_iter):
        # Long eq 6 y 7
        f = K.dot(x)
        for i in range(m):
            f[i] = f[i] / np.sum(K[i,:]) # La suma no deberia dar 1?
        # Long eq di_k
        d = np.zeros(m)
        for i in range(m):
            d[i] = np.sqrt(y[i] / f[i])
        # Long eq 8
        u = np.zeros((m,n))
        for i in range(m):
            for j in range(n):
                if d[i] >= 1:
                    u[i,j] = 1.0 / ((0.5/f[i])*(1.0 - 1.0/d[i]) + 1.0/(x[j]*d[i]))
                else:
                    u[i,j] = 0.5*f[i]*(1.0 - d[i]) + x[j]*d[i]
        for j in range(n):
            g_j = np.sum(K[:,j]) 
            accum = 0.0
            for i in range(m):
                accum = accum + K[i,j]*u[i,j]
            x[j] = accum / g_j
        return x


def SIR(Context,Observation,tita, max_iter=50):
    return [SIR_single_pol('H', Context, Observation, tita, max_iter), SIR_single_pol('V', Context, Observation, tita, max_iter)]





    
#%%############################################################################
#%                                                                           %#
#%                                                                           %#
#%   Solver       RODGERS                                                    %#
#%                                                                           %#
##%############################################################################
def Rodgers_Compute_Covarince(Context, Observation,tita,pol,obsstd=1.0): #compute sol covariance matrix
        Wt=Observation['Wt']
        VType=Observation['VType']
        n_ells=Wt.shape[0]
        n_Vars=Wt.shape[1]
        sigma2=tita[pol]['sigma2']
        
        Sx=np.zeros([n_Vars,n_Vars])
        for v in range(n_Vars):
            Sx[v,v]=abs(sigma2[VType[v]])
        I=np.eye(n_ells)
        Sy=Wt.dot(Sx.dot(Wt.transpose())) + obsstd*obsstd*I 
        InvSy=L_ParamEst.InvSym(Sy)
        InvSx=L_ParamEst.InvSym(Sx)
        InvS=InvSx+(Wt.transpose()).dot(InvSy.dot(Wt))
        S=L_ParamEst.InvSym(InvS)
        return S
        
def Rodgers(Context, Observation, tita, pol, obsstd=1.0): #compute solution for one pol
        Tb=Observation['Tb'][pol]
        K=Observation['Wt']
        VType=Observation['VType']
        n_ells=K.shape[0]
        n_Vars=K.shape[1]
        sigma2=tita[pol]['sigma2']
        mu=tita[pol]['mu']

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
            
def Solve_Rodgers_IT_with_stdLB(Context,Observation, tita,obsstd=1.0, verbose=True): #compute solution iterating
    
    K=Observation['Wt']
    W=K.dot(K.transpose())
    var_obs_land=obsstd/W.diagonal().min()

    for i in range(Context['Dict_Grids']['nlt']):
        tita['H']['sigma2'][i]=25*var_obs_land
        tita['V']['sigma2'][i]=25*var_obs_land
        tita['H']['mu'][i]=Observation['LSQRSols']['H']['Sol'][i]
        tita['V']['mu'][i]=Observation['LSQRSols']['V']['Sol'][i]
    #    tita['V']['mu'][1]=200
    #obsstd=1.0
    print("\nIterating Rodgers... error std:%.2f min_std:%.2f"%(np.sqrt(obsstd),np.sqrt(var_obs_land)))

    if verbose:
        L_ParamEst.pr1(tita,"Initial parameters")
    again=True
    i=0
    max_iter=50
    tol=0.005
    if verbose:
        print ("D_it: ", end=' ')
    while again:
        SolH=Rodgers(Context, Observation, tita, 'H', obsstd=obsstd)
        SolV=Rodgers(Context, Observation, tita, 'V', obsstd=obsstd)
        tita_old=copy.deepcopy(tita)
        tita=L_ParamEst.recompute_param([SolH,SolV],Observation,tita)
        tita=L_ParamEst.minSigma2(tita,var_obs_land)
        #L_ParamEst.pr(tita,"it %d"%(i+1))
        e=np.max([np.abs(np.sqrt(tita['H']['sigma2'])-np.sqrt(tita_old['H']['sigma2'])),
          np.abs(np.sqrt(tita['V']['sigma2'])-np.sqrt(tita_old['V']['sigma2']))])
        if verbose: 
            L_ParamEst.pr1(tita,"it %d (d=%.3f)"%(i+1,e))
        #again?
        i+=1
        if (i>=max_iter):
            again=False
                 
        if e<tol:
            again=False
            print("- converged " %e, end=' ')
        sys.stdout.flush()
            
        
    print('- %dit -'%i)
    #print error norm
    K=Observation['Wt']
    Tbh=Observation['Tb']['H']
    Tbv=Observation['Tb']['V']
    difH=Tbh-K.dot(SolH)
    difV=Tbv-K.dot(SolV)
    sn=np.sqrt(Tbh.shape[0])
    rmseH=np.linalg.norm(difH)/sn
    rmseV=np.linalg.norm(difV)/sn
    MSG="\nSolved with Rodgers_IT (Band " + Observation['Band'] + ')\n'
    err_str="RMSE, H:%.2f, V:%.2f "%(rmseH,rmseV)
    print(err_str, end=' ')
    sys.stdout.flush()
    MSG+='- %dit -'%i + err_str + '\n'
    logging.info(MSG)
    return [SolH, SolV]


def Solve_Rodgers_IT(Context,Observation, tita,obsstd=1.0, verbose=True,tol=0.005, max_iter=50): #compute solution iterating - NO BOUND
    K=Observation['Wt']

    for i in range(Context['Dict_Grids']['nlt']):
        tita['H']['sigma2'][i]=100 #a high var
        tita['V']['sigma2'][i]=100 #a high var
        tita['H']['mu'][i]=Observation['LSQRSols']['H']['Sol'][i]
        tita['V']['mu'][i]=Observation['LSQRSols']['V']['Sol'][i]

    print("\nIterating Rodgers... error std:%.2f"%(obsstd))

    if verbose:
        L_ParamEst.pr1(tita,"Initial parameters")
    again=True
    i=0
    if verbose:
        print ("D_it: ", end=' ')
    while again:
        SolH=Rodgers(Context, Observation, tita, 'H', obsstd=obsstd)
        SolV=Rodgers(Context, Observation, tita, 'V', obsstd=obsstd)
        tita_old=copy.deepcopy(tita)
        tita=L_ParamEst.recompute_param([SolH,SolV],Observation,tita)
        #tita=L_ParamEst.minSigma2(tita,var_obs_land)
        #L_ParamEst.pr(tita,"it %d"%(i+1))
        e=np.max([np.abs(np.sqrt(tita['H']['sigma2'])-np.sqrt(tita_old['H']['sigma2'])),
          np.abs(np.sqrt(tita['V']['sigma2'])-np.sqrt(tita_old['V']['sigma2']))])
        if verbose: 
            L_ParamEst.pr1(tita,"it %d (d=%.3f)"%(i+1,e))
        #again?
        i+=1
        if (i>=max_iter):
            again=False
                 
        if e<tol:
            again=False
            print("- converged " %e, end=' ')
        sys.stdout.flush()
            
        
    print('- %dit -'%i)
    #print error norm
    K=Observation['Wt']
    Tbh=Observation['Tb']['H']
    Tbv=Observation['Tb']['V']
    difH=Tbh-K.dot(SolH)
    difV=Tbv-K.dot(SolV)
    sn=np.sqrt(Tbh.shape[0])
    rmseH=np.linalg.norm(difH)/sn
    rmseV=np.linalg.norm(difV)/sn
    MSG="\nSolved with Rodgers_IT (Band " + Observation['Band'] + ')\n'
    err_str="Obs RMSE, H:%.2f, V:%.2f "%(rmseH,rmseV)
    print(err_str, end=' ')
    sys.stdout.flush()
    MSG+='- %dit -'%i + err_str + '\n'
    logging.info(MSG)
    return [SolH, SolV]

def Solve_Rodgers_ITH(Context,Observation, tita,obsstd=1.0, verbose=True,tol=0.005, max_iter=50): #compute solution iterating - NO BOUND
    K=Observation['Wt']    
    #k=K.shape[1]
    pc = [0.5, 0.5] # PROPORCION CLIMATOLOGICA
    vc = [25.0,100.0]  # VARIANZA CLIMATOLOGICA
    
    #alpha=1.0
    #beta=100.0
    #print("a:",a,", b:",b,", n:",k)
    nlt=Context['Dict_Grids']['nlt']
    VType=Observation['VType']

    for lt in range(nlt):
        tita['H']['sigma2'][lt]=100 #a high var
        tita['V']['sigma2'][lt]=100 #a high var
        tita['H']['mu'][lt]=Observation['LSQRSols']['H']['Sol'][lt]
        tita['V']['mu'][lt]=Observation['LSQRSols']['V']['Sol'][lt]

    print("\nIterating Rodgers... error std:%.2f"%(obsstd))

    if verbose:
        L_ParamEst.pr1(tita,"Initial parameters")
    again=True
    i=0
    if verbose:
        print ("D_it: ", end=' ')
    while again:
        SolH=Rodgers(Context, Observation, tita, 'H', obsstd=obsstd)
        SolV=Rodgers(Context, Observation, tita, 'V', obsstd=obsstd)
        tita_old=copy.deepcopy(tita)

        #just for fun        
        tita=L_ParamEst.recompute_param([SolH,SolV],Observation,tita)        
        L_ParamEst.pr1(tita,"Muestral it %d"%(i+1))

        #Compute new sigma with Inverse Gamma prior and posterior
        for lt in range(nlt):
            LT_Vars=np.where(VType==lt)[0]
            n=len(LT_Vars)
            #a=alpha*n/3
            #b=beta*n/3
            a = (1.0 - pc[lt] - 0.5*n*pc[lt]) / (pc[lt] - 1.0)
            b = vc[lt] * (a + 1.0)
            TbH_LT_Vars=SolH[LT_Vars]
            TbV_LT_Vars=SolV[LT_Vars]
            tita['H']['mu'][lt] = TbH_LT_Vars.mean()
            tita['V']['mu'][lt] = TbV_LT_Vars.mean()

            tita['H']['sigma2'][lt] = (b+.5*((TbH_LT_Vars-tita['H']['mu'][lt])**2).sum())/(a+1+n/2)
            tita['V']['sigma2'][lt] = (b+.5*((TbV_LT_Vars-tita['V']['mu'][lt])**2).sum())/(a+1+n/2)

        e=np.max([np.abs(np.sqrt(tita['H']['sigma2'])-np.sqrt(tita_old['H']['sigma2'])),
          np.abs(np.sqrt(tita['V']['sigma2'])-np.sqrt(tita_old['V']['sigma2']))])
        if verbose: 
            L_ParamEst.pr1(tita,"it %d (d=%.3f)"%(i+1,e))
        #again?
        i+=1
        if (i>=max_iter):
            again=False
                 
        if e<tol:
            again=False
            print("- converged " %e, end=' ')
        sys.stdout.flush()
            
        
    print('- %dit -'%i)
    #print error norm
    K=Observation['Wt']
    Tbh=Observation['Tb']['H']
    Tbv=Observation['Tb']['V']
    difH=Tbh-K.dot(SolH)
    difV=Tbv-K.dot(SolV)
    sn=np.sqrt(Tbh.shape[0])
    rmseH=np.linalg.norm(difH)/sn
    rmseV=np.linalg.norm(difV)/sn
    MSG="\nSolved with Rodgers_IT (Band " + Observation['Band'] + ')\n'
    err_str="Obs RMSE, H:%.2f, V:%.2f "%(rmseH,rmseV)
    print(err_str, end=' ')
    sys.stdout.flush()
    MSG+='- %dit -'%i + err_str + '\n'
    logging.info(MSG)
    return [SolH, SolV]

def Solve_Rodgers(Context,Observation, tita=None,obsstd=1.0,verbose=True): #compute solution for H&V
    if (tita==None):
      tita=L_ParamEst.compute_params_from_pure_ells(Observation,tita)
    if verbose:
      L_ParamEst.pr(tita,'Input paramters for Rodgers')
        
    #print("\nSolving Rodgers with param tita...")
    SolH=Rodgers(Context, Observation, tita, 'H', obsstd=obsstd)
    SolV=Rodgers(Context, Observation, tita, 'V', obsstd=obsstd)
    #print("Solution stats:")
    #L_ParamEst.compute_param([SolH,SolV],Observation)
    return [SolH, SolV]


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


def constructH(Wt, neighbors):
    cantVars = Wt.shape[1]
    H = np.zeros((cantVars, cantVars))
    for i in range(cantVars):
        H[i,i] = 1.0
        localNeighbors = neighbors[i]
        for neighbor in localNeighbors:
            H[i, neighbor] = -1.0 / float(len(localNeighbors))
    return H

# Solvers Tichonov...
def Solve_Tichonov(Observation,lamb, H):
        #obsstd=1
        #print ("Solving System |", end=" ")
        print ("Solving System |")
        Wt=Observation['Wt']
        VType=Observation['VType']
        nlt=VType.max()+1
        Tb=Observation['Tb'] #'H' and 'V'

        Vars_of_type=[]
        for lt in range(nlt):
            Vars_of_type.append(np.where(VType==lt)[0])        
            
        Solu={}
        for pol in ['H','V']:
            K = lamb*H
            Solu[pol] = L_ParamEst.InvSym(Wt.T.dot(Wt) + K.T.dot(K)).dot(Wt.T).dot(Tb[pol])        
        return [Solu['H'],Solu['V']]

def Solve_GCV_Tichonov(Observation, neighbors, gridLambs):
    Wt=Observation['Wt']
    Tb=Observation['Tb'] #'H' and 'V'
    H = constructH(Wt, neighbors)
    cantElipses = Wt.shape[0]
    Asharps = []
    print("Precomputing A#...")
    Solu={}
    for lamb in gridLambs:
        print("For lambda = ", lamb)
        Asharps.append(L_ParamEst.InvSym(Wt.T.dot(Wt) + (lamb**2)*H.T.dot(H)).dot(Wt.T))
    for pol in ['H','V']:
        print("Polarization: ", pol)
        CV_values = []
        for i in range(len(gridLambs)):
            lamb = gridLambs[i]
            Asharp = Asharps[i]
            K = lamb*H
            x = L_ParamEst.InvSym(Wt.T.dot(Wt) + K.T.dot(K)).dot(Wt.T).dot(Tb[pol])
            I = np.eye(cantElipses)
            value = cantElipses*(linalg.norm(Wt.dot(x) - Tb[pol]))**2 / (np.trace(I - Wt.dot(Asharp)))**2
            print("Lambda: ", lamb, " with GCV value: ", value)
            CV_values.append(value)
        lambOptimo = gridLambs[np.argmin(CV_values)]
        print("Lambda optimo elegido: " + str(lambOptimo))
        K = lambOptimo*H
        Solu[pol] = L_ParamEst.InvSym(Wt.T.dot(Wt) + K.T.dot(K)).dot(Wt.T).dot(Tb[pol])        
    return [Solu['H'],Solu['V']]


#%%############################################################################
#%                                                                           %#
#%                                                                           %#
#%   Solver   Global Tichonov                                                %#
#%                                                                           %#
##%############################################################################



def constructGlobalH(Vtypes, Wt):
    cantVars = Wt.shape[1]
    H = np.zeros((cantVars, cantVars))
    for i in range(cantVars):
        vtype = Vtypes[i]
        H[i,i] = 1.0
        globalNeighbors = []
        for j in range(len(Vtypes)):
            if Vtypes[j] == vtype and j != i:
                globalNeighbors.append(j)
        for neighbor in globalNeighbors:
            H[i, neighbor] = -1.0 / float(len(globalNeighbors))
    return H


def Solve_Global_GCV_Tichonov(Vtypes,Observation, gridLambs):
    Wt=Observation['Wt']
    Tb=Observation['Tb'] #'H' and 'V'
    H = constructGlobalH(Vtypes, Wt)
    cantElipses = Wt.shape[0]
    Asharps = []
    print("Precomputing A#...")
    Solu={}
    for lamb in gridLambs:
        print("For lambda = ", lamb)
        Asharps.append(L_ParamEst.InvSym(Wt.T.dot(Wt) + (lamb**2)*H.T.dot(H)).dot(Wt.T))
    for pol in ['H','V']:
        print("Polarization: ", pol)
        CV_values = []
        for i in range(len(gridLambs)):
            lamb = gridLambs[i]
            Asharp = Asharps[i]
            K = lamb*H
            x = L_ParamEst.InvSym(Wt.T.dot(Wt) + K.T.dot(K)).dot(Wt.T).dot(Tb[pol])
            I = np.eye(cantElipses)
            value = cantElipses*(linalg.norm(Wt.dot(x) - Tb[pol]))**2 / (np.trace(I - Wt.dot(Asharp)))**2
            print("Lambda: ", lamb, " with GCV value: ", value)
            CV_values.append(value)
        lambOptimo = gridLambs[np.argmin(CV_values)]
        print("Lambda optimo elegido: " + str(lambOptimo))
        K = lambOptimo*H
        Solu[pol] = L_ParamEst.InvSym(Wt.T.dot(Wt) + K.T.dot(K)).dot(Wt.T).dot(Tb[pol])        
    return [Solu['H'],Solu['V']]



#%%############################################################################
#%                                                                           %#
#%                                                                           %#
#%   Solver       Tichonov K-dimensional                                     %#
#%                                                                           %#
##%############################################################################



# Recibe los lambdas de cada tipo de land type.
def K_constructH(Vtypes, lambdas, Wt, neighbors):
    cantVars = Wt.shape[1]
    H = np.zeros((cantVars, cantVars))
    for i in range(cantVars):
        H[i,i] = 1.0
        localNeighbors = neighbors[i]
        for neighbor in localNeighbors:
            H[i, neighbor] = -1.0 / float(len(localNeighbors))
        H[i,:] = H[i,:] * lambdas[Vtypes[i]]
    return H

def K_Solve_GCV_Tichonov(Vtypes, Observation, neighbors, gridLambs):
    Wt=Observation['Wt']
    Tb=Observation['Tb'] #'H' and 'V'
    #H = constructH(Wt, neighbors)
    cantElipses = Wt.shape[0]
    cantTipos = np.max(Vtypes) + 1
    Asharps = []
    Hs = []
    print("Precomputing A#...")
    Solu={}
    print("Cantidad de celdas " + str(len(Vtypes)))
    print("Cantidad de tipos: " + str(cantTipos))
    for lambComb in product(gridLambs,repeat=cantTipos):
        print("For lambComb = ", lambComb)
        #Asharps.append(linalg.inv(Wt.T.dot(Wt) + (lamb**2)*H.T.dot(H)).dot(Wt.T))
        H = K_constructH(Vtypes, lambComb, Wt, neighbors)
        Hs.append(H)
        Asharps.append(L_ParamEst.InvSym(Wt.T.dot(Wt) + H.T.dot(H)).dot(Wt.T))
    for pol in ['H','V']:
        print("Polarization: ", pol)
        CV_values = []
        i = 0
        combLambdas = list(product(gridLambs,repeat=cantTipos))
        for lambComb in combLambdas:
            Asharp = Asharps[i]
            H = Hs[i]
            x = L_ParamEst.InvSym(Wt.T.dot(Wt) + H.T.dot(H)).dot(Wt.T).dot(Tb[pol])
            I = np.eye(cantElipses)
            value = cantElipses*(linalg.norm(Wt.dot(x) - Tb[pol]))**2 / (np.trace(I - Wt.dot(Asharp)))**2
            print("Lambda Comb: ", lambComb, " with GCV value: ", value)
            CV_values.append(value)
            i = i + 1
        iOptimo = np.argmin(CV_values)
        combLambdaOptimo = combLambdas[iOptimo]
        print("Comb Lambda optimo elegido: " + str(combLambdaOptimo))
        H = Hs[iOptimo]
        Solu[pol] = L_ParamEst.InvSym(Wt.T.dot(Wt) + H.T.dot(H)).dot(Wt.T).dot(Tb[pol])        
    return [Solu['H'],Solu['V']]


#%%############################################################################
#%                                                                           %#
#%                                                                           %#
#%   Solver   Global    Tichonov K-dimensional                               %#
#%                                                                           %#
##%############################################################################



# Recibe los lambdas de cada tipo de land type.
def K_constructGlobalH(Vtypes, lambdas, Wt):
    cantVars = Wt.shape[1]
    H = np.zeros((cantVars, cantVars))
    for i in range(cantVars):
        vtype = Vtypes[i]
        H[i,i] = 1.0
        globalNeighbors = []
        for j in range(len(Vtypes)):
            if Vtypes[j] == vtype and j != i:
                globalNeighbors.append(j)
        for neighbor in globalNeighbors:
            H[i, neighbor] = -1.0 / float(len(globalNeighbors))
        H[i,:] = H[i,:] * lambdas[Vtypes[i]]
    return H

def K_Solve_Global_GCV_Tichonov(Vtypes, Observation, gridLambs):
    Wt=Observation['Wt']
    Tb=Observation['Tb'] #'H' and 'V'
    #H = constructH(Wt, neighbors)
    cantElipses = Wt.shape[0]
    cantTipos = np.max(Vtypes) + 1
    Asharps = []
    Hs = []
    print("Precomputing A#...")
    Solu={}
    print("Cantidad de celdas " + str(len(Vtypes)))
    print("Cantidad de tipos: " + str(cantTipos))
    for lambComb in product(gridLambs,repeat=cantTipos):
        print("For lambComb = ", lambComb)
        #Asharps.append(linalg.inv(Wt.T.dot(Wt) + (lamb**2)*H.T.dot(H)).dot(Wt.T))
        H = K_constructGlobalH(Vtypes, lambComb, Wt)
        Hs.append(H)
        Asharps.append(L_ParamEst.InvSym(Wt.T.dot(Wt) + H.T.dot(H)).dot(Wt.T))
    for pol in ['H','V']:
        print("Polarization: ", pol)
        CV_values = []
        i = 0
        combLambdas = list(product(gridLambs,repeat=cantTipos))
        for lambComb in combLambdas:
            Asharp = Asharps[i]
            H = Hs[i]
            x = L_ParamEst.InvSym(Wt.T.dot(Wt) + H.T.dot(H)).dot(Wt.T).dot(Tb[pol])
            I = np.eye(cantElipses)
            value = cantElipses*(linalg.norm(Wt.dot(x) - Tb[pol]))**2 / (np.trace(I - Wt.dot(Asharp)))**2
            print("Lambda Comb: ", lambComb, " with GCV value: ", value)
            CV_values.append(value)
            i = i + 1
        iOptimo = np.argmin(CV_values)
        combLambdaOptimo = combLambdas[iOptimo]
        print("Comb Lambda optimo elegido: " + str(combLambdaOptimo))
        H = Hs[iOptimo]
        Solu[pol] = L_ParamEst.InvSym(Wt.T.dot(Wt) + H.T.dot(H)).dot(Wt.T).dot(Tb[pol])        
    return [Solu['H'],Solu['V']]




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

def Solve_EM(Context,Observation,tita, obsstd=1.0, verbose=True,tol=0.005, max_iter=50):
    print("Solving H")
    Sol = [Solve_EM_single_pol('H', Context,Observation,tita, obsstd, verbose,tol, max_iter)]
    print("Solving V")
    Sol.append(Solve_EM_single_pol('V', Context,Observation,tita, obsstd, verbose,tol, max_iter))
    return Sol


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
    print("A_inverse: ", A_inv)
    print("A approx inverse: ", A_inv_approx)

#test_approx_inverse() # Parece andar horrible.




def Solve_EM_MAP_single_pol(pol, Context,Observation,tita, obsstd=1.0, verbose=True,tol=0.005, max_iter=50, omegas=None, sigma2_clis=None, cholesky_inv = False):
    K=Observation['Wt']    
    nlt=Context['Dict_Grids']['nlt']
    if omegas == None or sigma2_clis == None:
        print("NO HAY PRIOR, usando ML")
        omegas = np.ones(nlt)
        sigmas2_clis = np.zeros(nlt)
    VType=Observation['VType']
    tita = copy.deepcopy(tita)
    for lt in range(nlt):
        tita['H']['sigma2'][lt]=100 #a high var
        tita['V']['sigma2'][lt]=100 #a high var
        tita['H']['mu'][lt]=Observation['LSQRSols']['H']['Sol'][lt]
        tita['V']['mu'][lt]=Observation['LSQRSols']['V']['Sol'][lt]

    old_mus = np.zeros(nlt)
    old_sigma2 = np.zeros(nlt)
    for lt in range(nlt):
        old_mus[lt] = tita[pol]['mu'][lt]
        old_sigma2[lt] = tita[pol]['sigma2'][lt]

    print("\nIterating EM... error std:%.2f"%(obsstd))

    if verbose:
        L_ParamEst.pr1(tita,"Initial parameters")
    again=True
    it=0
    if verbose:
        print ("D_it: ", end=' ')
    m = K.shape[0]
    n = K.shape[1]
    y = Observation['Tb'][pol]
    sigmas = []
    sigmas.append(old_sigma2)
    mus = []
    mus.append(old_mus)
    while again:
        print("Iteracion ", it)
        print("Mus por landtype: ", old_mus)
        print("Sigma2 por landtype: ", old_sigma2)
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
            #sigma_pos =  L_ParamEst.InvSym(sigma_pos_inv)
            #proj_sigma_pos = sigma_pos[LT_Vars,:][:, LT_Vars]
            #M1 = proj_sigma_pos.trace()
            # Forma alternativa con autovalores.
            #eigenvals = np.linalg.eigvalsh(sigma_pos_inv)
            #M1 = (L_ParamEst.InvSym(sigma_pos_inv)[LT_Vars,:][:, LT_Vars]).trace()                        
            # Metodo Golub
            # OJO!!!! ESTA MAL... ES TOMAR PARTE DE LA MATRIZ DESPUES DE INVERTIR.            
            #proj_sigma_pos_inv = sigma_pos_inv[LT_Vars,:][:, LT_Vars]
            #traceA = proj_sigma_pos_inv.trace()
            #nA = proj_sigma_pos_inv.shape[0]
            #frobA = (proj_sigma_pos_inv*proj_sigma_pos_inv).sum()
            #espectro = np.linalg.eigvalsh(proj_sigma_pos_inv)[0]            
            #cota = np.array([[traceA, nA]]).dot(np.array([[frobA, traceA], [espectro**2, 1.0/espectro]])).dot([[nA], [1.0]])
            #M1 = L_ParamEst.InvSym(sigma_pos_inv).trace()
            #np.linalg.cholesk            
            #            print("Chequeando complejidad...")            
            #            for eigenval in eigenvals:
            #                if eigenval.imag != 0.0:
            #                    print("HAY AUTOVALOR COMPLEJO!!!")
            #                    print(eigenval)
            #                if eigenval < 0.0:
            #                    print("HAY AUTOVALOR NEGATIVO!")
            #                    print(eigenval)
            #                if eigenval == 0.0:
            #                    print("HAY AUTOVALOR CERO!")
            #                    print(eigenval)
            #                sys.stdout.flush()
            #eigenvals [eigenvals < 1] = 1
            #M1 = np.sum(1.0 / eigenvals[LT_Vars])
            #if approx_inv_trace:
                #M1 = sigma_t[LT_Vars].sum() # Aproximamos por la diagonal, la suma de las varianzas. Es decir, se ignora lo de K. Anda muy feo (hace diverger) en banda C 25 KM.
            #    M1 = ( 1.0 / np.diag(sigma_pos_inv)[LT_Vars]).sum()
            if cholesky_inv:
                U = linalg.cholesky(sigma_pos_inv)
                U_inv = linalg.inv(U)
                sigma_pos = U_inv.dot(U_inv.T)
                M1 = np.diag(sigma_pos)[LT_Vars].sum()
            else:
                #M1 = (L_ParamEst.InvSym(sigma_pos_inv)[LT_Vars,:][:, LT_Vars]).trace()
                M1 = (np.diag(L_ParamEst.InvSym(sigma_pos_inv))[LT_Vars]).sum()
            M = M1 + M2
            new_sigma2[lt] =(M / nk)*omegas[lt] + (1-omegas[lt])*sigma2_clis[lt]
        print("OldSigmas: ", old_sigma2)
        print("NewSigma2: ", new_sigma2)
        sys.stdout.flush()
        e= np.max(np.abs(np.sqrt(new_sigma2)-np.sqrt(old_sigma2)))
        if (it>=max_iter):
            again=False
        if e<tol:
            again=False
            print("- converged " %e, end=' ')
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

def Solve_EM_MAP(Context,Observation,tita, obsstd=1.0, verbose=True,tol=0.005, max_iter=50, omegas=None, sigma2_clis=None, cholesky_inv=False):
    print("Solving H")
    Sol = [Solve_EM_MAP_single_pol('H', Context,Observation,tita, obsstd, verbose,tol, max_iter,omegas,sigma2_clis, cholesky_inv)]
    print("Solving V")
    Sol.append(Solve_EM_MAP_single_pol('V', Context,Observation,tita, obsstd, verbose,tol, max_iter,omegas,sigma2_clis, cholesky_inv))
    return Sol




####### K-FOLD

def Solve_EM_MAP_single_pol_specificK_y(K, y, pol, Context,Observation,tita, obsstd=1.0, verbose=True,tol=0.005, max_iter=50, omegas=None, sigma2_clis=None, cholesky_inv = False):
    nlt=Context['Dict_Grids']['nlt']
    if omegas == None or sigma2_clis == None:
        if verbose:
            print("NO HAY PRIOR, usando ML")
        omegas = np.ones(nlt)
        sigmas2_clis = np.zeros(nlt)
    VType=Observation['VType']
    tita = copy.deepcopy(tita)
    for lt in range(nlt):
        tita['H']['sigma2'][lt]=100 #a high var
        tita['V']['sigma2'][lt]=100 #a high var
        tita['H']['mu'][lt]=Observation['LSQRSols']['H']['Sol'][lt]
        tita['V']['mu'][lt]=Observation['LSQRSols']['V']['Sol'][lt]

    old_mus = np.zeros(nlt)
    old_sigma2 = np.zeros(nlt)
    for lt in range(nlt):
        old_mus[lt] = tita[pol]['mu'][lt]
        old_sigma2[lt] = tita[pol]['sigma2'][lt]

    if verbose:
        print("\nIterating EM... error std:%.2f"%(obsstd))

    if verbose:
        L_ParamEst.pr1(tita,"Initial parameters")
    again=True
    it=0
    if verbose:
        print ("D_it: ", end=' ')
    m = K.shape[0]
    n = K.shape[1]
    sigmas = []
    sigmas.append(old_sigma2)
    mus = []
    mus.append(old_mus)
    while again:
        if verbose:
            print("Iteracion ", it)
            print("Mus por landtype: ", old_mus)
            print("Sigma2 por landtype: ", old_sigma2)
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
            #sigma_pos =  L_ParamEst.InvSym(sigma_pos_inv)
            #proj_sigma_pos = sigma_pos[LT_Vars,:][:, LT_Vars]
            #M1 = proj_sigma_pos.trace()
            # Forma alternativa con autovalores.
            #eigenvals = np.linalg.eigvalsh(sigma_pos_inv)
            #M1 = (L_ParamEst.InvSym(sigma_pos_inv)[LT_Vars,:][:, LT_Vars]).trace()                        
            # Metodo Golub
            # OJO!!!! ESTA MAL... ES TOMAR PARTE DE LA MATRIZ DESPUES DE INVERTIR.            
            #proj_sigma_pos_inv = sigma_pos_inv[LT_Vars,:][:, LT_Vars]
            #traceA = proj_sigma_pos_inv.trace()
            #nA = proj_sigma_pos_inv.shape[0]
            #frobA = (proj_sigma_pos_inv*proj_sigma_pos_inv).sum()
            #espectro = np.linalg.eigvalsh(proj_sigma_pos_inv)[0]            
            #cota = np.array([[traceA, nA]]).dot(np.array([[frobA, traceA], [espectro**2, 1.0/espectro]])).dot([[nA], [1.0]])
            #M1 = L_ParamEst.InvSym(sigma_pos_inv).trace()
            #np.linalg.cholesk            
            #            print("Chequeando complejidad...")            
            #            for eigenval in eigenvals:
            #                if eigenval.imag != 0.0:
            #                    print("HAY AUTOVALOR COMPLEJO!!!")
            #                    print(eigenval)
            #                if eigenval < 0.0:
            #                    print("HAY AUTOVALOR NEGATIVO!")
            #                    print(eigenval)
            #                if eigenval == 0.0:
            #                    print("HAY AUTOVALOR CERO!")
            #                    print(eigenval)
            #                sys.stdout.flush()
            #eigenvals [eigenvals < 1] = 1
            #M1 = np.sum(1.0 / eigenvals[LT_Vars])
            #if approx_inv_trace:
                #M1 = sigma_t[LT_Vars].sum() # Aproximamos por la diagonal, la suma de las varianzas. Es decir, se ignora lo de K. Anda muy feo (hace diverger) en banda C 25 KM.
            #    M1 = ( 1.0 / np.diag(sigma_pos_inv)[LT_Vars]).sum()
            if cholesky_inv:
                U = linalg.cholesky(sigma_pos_inv)
                U_inv = linalg.inv(U)
                sigma_pos = U_inv.dot(U_inv.T)
                M1 = np.diag(sigma_pos)[LT_Vars].sum()
            else:
                #M1 = (L_ParamEst.InvSym(sigma_pos_inv)[LT_Vars,:][:, LT_Vars]).trace()
                M1 = (np.diag(L_ParamEst.InvSym(sigma_pos_inv))[LT_Vars]).sum()
            M = M1 + M2
            new_sigma2[lt] =(M / nk)*omegas[lt] + (1-omegas[lt])*sigma2_clis[lt]
        if verbose:
            print("OldSigmas: ", old_sigma2)
            print("NewSigma2: ", new_sigma2)
        sys.stdout.flush()
        e= np.max(np.abs(np.sqrt(new_sigma2)-np.sqrt(old_sigma2)))
        if (it>=max_iter):
            again=False
        if e<tol:
            again=False
            if verbose:
                print("- converged " %e, end=' ')
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



def Solve_EM_MAP_KFold_single_pol(pol, Context,Observation,tita, grilla_obsstd, n_folds, verbose=True,tol=0.005, max_iter=50, omegas=None, sigma2_clis=None, cholesky_inv = False):
    K=Observation['Wt']
    y = Observation['Tb'][pol]
    folds = []
    m = K.shape[0]
    bag = np.array(range(m))
    fold_size = int(m/n_folds)
    for i in range(n_folds-1):
        fold = random.choice(bag, size = fold_size, replace=False)
        bag = np.delete(bag, fold)
        folds.append(fold)
    if len(bag) > 0:
        folds.append(bag) # Adjuntamos el remanente del bag.
    errores = np.zeros(len(grilla_obsstd))
    for i in range(len(grilla_obsstd)):
        obsstd = grilla_obsstd[i]
        error = 0.0
        for j in range(len(folds)):
            print("Fold ", j)
            test_indices = folds[j]
            test_y = y[test_indices]
            test_K = K[test_indices,:]
            train_indices = np.delete(np.array(range(m)), test_indices)
            train_y = y[train_indices]
            train_K = K[train_indices,:]
            x = Solve_EM_MAP_single_pol_specificK_y(train_K, train_y, pol, Context,Observation,tita, obsstd, verbose=False,tol=tol, max_iter=max_iter, omegas=omegas, sigma2_clis=sigma2_clis, cholesky_inv = cholesky_inv)
            partial_error = (1.0/len(folds[j]))*((test_y - test_K.dot(x))**2).sum()
            print("Partial error: ", partial_error)
            error = error + partial_error
        errores[i] = error
        print("Obs Std: ", obsstd, " Error: ", error)
    best_obsstd = grilla_obsstd[np.argmin(errores)]
    print("Gano ", best_obsstd)
    return Solve_EM_MAP_single_pol_specificK_y(K, y, pol, Context,Observation,tita, obsstd=best_obsstd, verbose=True,tol=tol, max_iter=max_iter, omegas=omegas, sigma2_clis=sigma2_clis, cholesky_inv = cholesky_inv)
                
def Solve_EM_MAP_K_Fold(Context,Observation,tita, grilla_obsstd, n_folds, verbose=True,tol=0.005, max_iter=50, omegas=None, sigma2_clis=None, cholesky_inv=False):
    print("Solving H")
    Sol = [Solve_EM_MAP_KFold_single_pol('H', Context,Observation,tita, grilla_obsstd, n_folds, verbose,tol, max_iter,omegas,sigma2_clis, cholesky_inv)]
    print("Solving V")
    Sol.append(Solve_EM_MAP_KFold_single_pol('V', Context,Observation,tita, grilla_obsstd, n_folds, verbose,tol, max_iter,omegas,sigma2_clis, cholesky_inv))
    return Sol
    
    
#%%############################################################################
#%                                                                           %#
#%                                                                           %#
#%   K-Fold TSVD                                                             %#
#%                                                                           %#
##%############################################################################


def Solve_TSVD_fixed_k(k, train_indices, pol, Context,Observation,tita, U,s,V):
    K=Observation['Wt'][train_indices,:]
    y = Observation['Tb'][pol][train_indices]
    #U, s, V = linalg.svd(K, full_matrices=True)
    Sigma = np.zeros((K.shape[1], K.shape[0]))
    for i in range(k):
        if i >= len(s):
            return np.zeros(K.shape[1])
        Sigma[i,i] = 1/s[i]
    invK = V.T.dot(Sigma).dot(U.T)
    return invK.dot(y)


def Solve_TSVD_single_pol(pol, Context,Observation,tita, n_folds, max_k):
    K=Observation['Wt']
    y = Observation['Tb'][pol]
    folds = []
    m = K.shape[0]
    bag = np.array(range(m))
    fold_size = int(m/n_folds)
    for i in range(n_folds-1):
        fold = random.choice(bag, size = fold_size, replace=False)
        bag = np.delete(bag, fold)
        folds.append(fold)
    if len(bag) > 0:
        folds.append(bag) # Adjuntamos el remanente del bag.
    errores = np.zeros(max_k)
    print("Precomputando descomposiciones SVD")
    Us = []
    ss = []
    Vs = []
    for j in range(len(folds)):
        test_indices = folds[j]
        train_indices = np.delete(np.array(range(m)), test_indices)
        U,s,V = linalg.svd(K[train_indices,:], full_matrices=True)
        Us.append(U)
        ss.append(s)
        Vs.append(V)
    for i in range(len(errores)):
        k = i + 1
        error = 0.0
        # Precomputo de U, s y V
        for j in range(len(folds)):
            #print("Fold ", j)
            test_indices = folds[j]
            test_y = y[test_indices]
            test_K = K[test_indices,:]
            train_indices = np.delete(np.array(range(m)), test_indices)
            x = Solve_TSVD_fixed_k(k, train_indices, pol, Context,Observation,tita, Us[j], ss[j], Vs[j])
            partial_error = (1.0/len(folds[j]))*((test_y - test_K.dot(x))**2).sum()
            #print("Partial error: ", partial_error)
            error = error + partial_error
        errores[i] = error
        print("k: ", k, " Error: ", error)
    best_k = np.argmin(errores) + 1
    print("Gano ", best_k)
    #return Solve_TSVD_fixed_k(best_k, pol, Context,Observation,tita)
    U,s,V = linalg.svd(K, full_matrices=True)
    return Solve_TSVD_fixed_k(best_k, range(m), pol, Context,Observation,tita, U,s,V)

def Solve_TSVD_K_Fold(Context,Observation,tita, n_folds, max_k):
    print("Solving H")
    Sol = [Solve_TSVD_single_pol('H', Context,Observation,tita, n_folds, max_k)]
    print("Solving V")
    Sol.append(Solve_TSVD_single_pol('V', Context,Observation,tita, n_folds, max_k))
    return Sol




#%%############################################################################
#%                                                                           %#
#%                                                                           %#
#%   SOLVE                                                                   %#
#%                                                                           %#
##%############################################################################
def Solve(Context,Observation,Method, tita,obsstd=1.0,verbose=True, distances=None):
    if Observation=={}:
        return []
    elif Method=='LSQR':
        Sol=Solve_LSQR(Observation)
    elif Method=="LSTSQR": # Es un caso especial de Tichonov con lambda = 0.
        neighbors = Context["Neighbors"]
        Wt=Observation['Wt']
        H = constructH(Wt, neighbors)
        lamb = 0.1
        Sol = Solve_Tichonov(Observation, lamb, H)        
    elif Method=='Weights': #Simple method
        Sol=Solve_Weights(Context,Observation)
    elif Method=='Rodgers': #1977 method Rodgers Method
        Sol=Solve_Rodgers(Context,Observation, tita=tita,obsstd=obsstd,verbose=verbose)
    elif Method=='Rodgers_IT': #Iterative version of Rodgers
        Sol=Solve_Rodgers_IT(Context,Observation,tita,obsstd=obsstd,verbose=verbose)
    elif Method=='Rodgers_ITH': #Iterative version of Rodgers
        Sol=Solve_Rodgers_ITH(Context,Observation,tita,obsstd=obsstd,verbose=verbose)
    elif Method=='BG_KFold':
        grilla_gamma = np.linspace(start=0.0001, stop=np.pi/2.0, num=3)
        n_folds = 5
        Sol = BG_KFold_complete(distances, n_folds, Context, Observation, grilla_gamma, w=.001,error_std=1.0,MaxObs=10)
    elif Method=='TSVD_KFold':
        Sol = Solve_TSVD_K_Fold(Context,Observation,tita, n_folds= 5, max_k=100)
    elif Method=='EM': #Iterative version of Rodgers
        n=Observation['Wt'].shape[1]
        omega = 1.0 # Anulamos el a priori por completo.
        sigma2_cli = 100
        Sol=Solve_EM_MAP(Context,Observation,tita,obsstd=obsstd,verbose=verbose, max_iter=50, omegas = [omega,omega], sigma2_clis=[sigma2_cli, sigma2_cli])
    elif Method=='EM_APPROX': #Iterative version of Rodgers
        n=Observation['Wt'].shape[1]
        omega = 1.0 # Anulamos el a priori por completo.
        sigma2_cli = 100
        Sol=Solve_EM_MAP(Context,Observation,tita,obsstd=obsstd,verbose=verbose, max_iter=50, omegas = [omega,omega], sigma2_clis=[sigma2_cli, sigma2_cli], approx_inv_trace=True)
    elif Method=='EM_MAP': #Iterative version of Rodgers
        n=Observation['Wt'].shape[1]
        omega = 0.5
        sigma2_cli0 = 25
        sigma2_cli1 = 100
        Sol=Solve_EM_MAP(Context,Observation,tita,obsstd=obsstd,verbose=verbose, max_iter=50, omegas = [omega,omega], sigma2_clis=[sigma2_cli0, sigma2_cli1])
    elif Method=='EM_CHOLESKY': #Iterative version of Rodgers
        n=Observation['Wt'].shape[1]
        omega = 1.0 # Anulamos el a priori por completo.
        sigma2_cli = 100
        Sol=Solve_EM_MAP(Context,Observation,tita,obsstd=obsstd,verbose=verbose, max_iter=50, omegas = [omega,omega], sigma2_clis=[sigma2_cli, sigma2_cli], cholesky_inv = True)
    elif Method=='EM_MAP_CHOLESKY': #Iterative version of Rodgers
        n=Observation['Wt'].shape[1]
        omega = 0.5 # Anulamos el a priori por completo.
        sigma2_cli0 = 25
        sigma2_cli1 = 100
        Sol=Solve_EM_MAP(Context,Observation,tita,obsstd=obsstd,verbose=verbose, max_iter=50, omegas = [omega,omega], sigma2_clis=[sigma2_cli0, sigma2_cli1], cholesky_inv = True)
    elif Method=='EM_K_Fold': #Iterative version of Rodgers
        n=Observation['Wt'].shape[1]
        omega = 1.0 # No prior.
        sigma2_cli0 = 25
        sigma2_cli1 = 100
        #grilla_obsstd= grillaLogaritmica(1e-2, 1e2, 10)
        grilla_obsstd= grillaLogaritmica(1e-2, 1e2, 5)
        Sol=Solve_EM_MAP_K_Fold(Context,Observation,tita,grilla_obsstd=grilla_obsstd, n_folds = 5, verbose=verbose, max_iter=50, omegas = [omega,omega], sigma2_clis=[sigma2_cli0, sigma2_cli1])
    elif Method=='EM_TEST': #Iterative version of Rodgers
        Sol=Solve_EM_single_pol_TEST('H', Context,Observation,tita,obsstd=obsstd,verbose=verbose, max_iter=200)
    elif Method == "BGF":
        Sol = BGF(Context, Observation)
    ### VERSIONES de TYCHONOV ###
    ### VERSIONES de TYCHONOV ###
    elif Method == "SIR":
        Sol = SIR(Context, Observation, tita, max_iter=50)
    elif Method == 'Tichonov':
        print("Applying Tichonov method...")
        neighbors = Context["Neighbors"]
        Wt=Observation['Wt']
        H = constructH(Wt, neighbors)
        lamb = 0.0
        Sol = Solve_Tichonov(Observation, lamb, H)
    elif Method == 'GCV_Tichonov':
        print("Applying GCV-Tichonov method...")
        #gridLambs = grillaLogaritmica(1e-6, 100.0, 9)
        gridLambs = grillaLogaritmica(1e-1, 1e1, 10)
        neighbors = Context["Neighbors"]
        Sol = Solve_GCV_Tichonov(Observation, neighbors, gridLambs)
    elif Method == "Global_GCV_Tichonov":
        #gridLambs = grillaLogaritmica(1e-6, 100.0, 9)
        gridLambs = grillaLogaritmica(1e-1, 1e1, 10)
        Vtypes = Observation['VType']
        Sol = Solve_Global_GCV_Tichonov(Vtypes, Observation, gridLambs)
    elif Method == "K_GCV_Tichonov":
        neighbors = Context["Neighbors"]
        Vtypes = Observation['VType']
        gridLambs = grillaLogaritmica(1e-1, 1e1, 10)
        Sol = K_Solve_GCV_Tichonov(Vtypes, Observation, neighbors, gridLambs)
    elif Method == "K_Global_GCV_Tichonov":
        gridLambs = grillaLogaritmica(1e-1, 1e1, 10)
        #gridLambs = grillaLogaritmica(1e-3, 100.0, 6)
        Vtypes = Observation['VType']
        Sol = K_Solve_Global_GCV_Tichonov(Vtypes, Observation,gridLambs)
    elif Method == 'Debug':
        print("Context Keys: ", Context.keys())
        print("Param: ", Context["Param"])
        print("Dict Grid: ", Context["Dict_Grids"])
        print("Neighbors: ", Context["Neighbors"])
        print("Observation: ", Observation)
        print("tita: ", tita)
    else:
        print("##################################\nUnkown method '%s', not solving!" %(Method))
        return None
    return Sol

  