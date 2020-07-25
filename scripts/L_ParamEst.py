# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 14:17:05 2017

@author: rgrimson
"""


from __future__ import print_function
from __future__ import division
import numpy as np
import sys
from scipy.sparse.linalg import lsqr
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

import random

###############################################################################
#%                                                                         %%#
#%                                                                         %%#
#%   Print parameters                                                      %%#
#%                                                                         %%#
#%%############################################################################
def pr(tita,title=''):
    nlt=len(tita['H']['mu'])
    
    #print title line
    w=67
    l=len(title)+2
    hw=int((w-l)/2)
    for i in range(hw):
        print ('-', end='')
    print (' %s '%title,end='')
    for i in range(w-2-l-hw):
        print ('-', end='')
    print('')

    #print column names   
    print("LT| H pol\tmu\tsigma\t| V pol\tmu\tsigma\t|")
    
    #print data
    for lt in range(nlt):
        print("%d |\t%.2f  \t%.2f\t| \t%.2f \t%.2f\t|" %(lt, tita['H']['mu'][lt], np.sqrt(tita['H']['sigma2'][lt]), tita['V']['mu'][lt], np.sqrt(tita['V']['sigma2'][lt])))


def pr_NoPol(tita,title=''):
    nlt=len(tita['mu'])
    
    #print title line
    w=67
    l=len(title)+2
    hw=int((w-l)/2)
    for i in range(hw):
        print ('-', end='')
    print (' %s '%title,end='')
    for i in range(w-2-l-hw):
        print ('-', end='')
    print('')

    #print column names   
    print("LT| \tmu\tsigma\t|")
    
    #print data
    for lt in range(nlt):
        print("%d |\t%.2f  \t%.2f\t|" %(lt, tita['mu'][lt], np.sqrt(tita['sigma2'][lt])))

        
def pr1(tita,title=''):
    nlt=len(tita['H']['mu'])
    
    #print title line
    w=67
    l=len(title)+2
    hw=int((w-l)/2)
    for i in range(hw):
        print ('-', end='')
    print (' %s '%title,end='')
    for i in range(w-2-l-hw):
        print ('-', end='')
    print('')

    #print column names   
    print("LT| H pol\tmu\tsigma\t| V pol\tmu\tsigma\t|")
    
    #print data
    for lt in range(nlt):
        print("%d |\t%.2f  \t%.2f\t| \t%.2f \t%.2f\t|" %(lt, tita['H']['mu'][lt], np.sqrt(tita['H']['sigma2'][lt]), tita['V']['mu'][lt], np.sqrt(tita['V']['sigma2'][lt])))

def toString(tita, titaReal, title=''):
    nlt=len(tita['H']['mu'])
    s= "+-+-+-+-+- " + title + " +-+-+-+-+-+-+- \n"


    #print column names   
    #print("LT| H pol\tmu\tsigma2\t| V pol\tmu\tsigma2\t|")
    
    #print data
    s += " Polarizacion H\n"
    for lt in range(nlt):
        s += "LandType: " + str(lt) + "\n"
        s += "Mu: " + str(tita['H']['mu'][lt]) + " vs Mu Real: " + str(titaReal['H']['mu'][lt]) +  "\n"
        s += "Sigma2: " + str(tita['H']['sigma2'][lt]) + "vs Sigma2 Real: " + str(titaReal['H']['sigma2'][lt]) +  "\n"
    s += " Polarizacion V\n"
    for lt in range(nlt):
        s += "LandType: " + str(lt) + "\n"
        s += "Mu: " + str(tita['V']['mu'][lt]) + " vs Mu Real: " + str(titaReal['V']['mu'][lt]) +  "\n"
        s += "Sigma2: " + str(tita['V']['sigma2'][lt]) + "vs Sigma2 Real: " + str(titaReal['V']['sigma2'][lt]) +  "\n"

    return s

#%%
def p_tita(tita):
    for pol in ['H','V']:
        print ("----------------------------------")
        print ("Pol: %s  \t LT: 0 ...\t1 ..." %pol)
        for m in tita[pol]:
            print ("%6s: \t" %m,end=" ")
            for v in tita[pol][m]:
                 if v:
                   print ("%.3f \t" %v,end=" ")
                 else:
                   print ("  *** \t",end=" ")
            print ("|")
    print ("----------------------------------")
    sys.stdout.flush()
      

def p_tita_(tita):
    for pol in ['H','V']:
        for m in ['mu','sigma2']:
            for v in tita[pol][m]:
                 if v:
                   print ("%3.3f " %v,end=" ")
                 else:
                   print ("   ***  ",end=" ")
            print ("|",end=" ")
    sys.stdout.flush()

def p_sigma2_(tita):
    print ("sigma2 (H|V): ",end=" ")
    for pol in ['H','V']:
        for v in tita[pol]['sigma2']:
             if v:
               print ("%3.3f " %v,end=" ")
             else:
               print ("   ***  ",end=" ")
        print ("|",end=" ")
    print(" ")
    sys.stdout.flush()

#%%############################################################################
#%                                                                          %%#
#%                                                                          %%#
#½  LSQR methods to estimate mu                                             %%#
#%                                                                          %%#
#%#############################################################################
def find_mu_by_lsqr(Observation):
    print ("Finding means with lsqr method...")
    M=Observation['LandPropEll']
    Tbh=Observation['Tb']['H']
    Tbv=Observation['Tb']['V']
    Sh=lsqr(M,Tbh)
    Sv=lsqr(M,Tbv)
    print ("Pol H:", Sh[0])
    print ("Pol V:", Sv[0])

#%%
def solve_some_lts_mu_by_lsqr(Observation,mu,pol):
    mu=np.array(mu)
    print ("Finding unknown means with lsqr method...")
    print ("input:",mu)
    print ("Pol",pol)
    VType=Observation['VType']
    nlt=VType.max()+1
    known=np.ones(nlt)
    unknown=np.zeros(nlt,dtype=bool)
    for lt in range(nlt):
        if mu[lt]==None:
            known[lt]=0    
            unknown[lt]=True
        else:
            known[lt]=mu[lt]
    M=Observation['LandPropEll']
    Tb=Observation['Tb'][pol]
    b=Tb-M.dot(known)
    Sol=lsqr(M[:,unknown],b)
    mu[unknown]=Sol[0] 
    print ("Sol:",mu)
    return mu

def compute_missing_mu_using_lsqr(Observation,tita):
    for pol in ['H','V']:
        mu=tita[pol]['mu']
        mu=solve_some_lts_mu_by_lsqr(Observation,mu,pol)
        tita[pol]['mu']=mu


#%%############################################################################
#%                                                                           %#
#%                                                                           %#
#%   Compute solution's params                                               %#
#%                                                                           %#
##%############################################################################
def compute_param(Sol,Observation):    
    VType=Observation['VType']
    nlt=VType.max()+1

    print("LT|  \t\t mu\tsigma \t\tmu\tsigma ")
    for lt in range(nlt):
        LT_Vars=np.where(VType==lt)[0]
        TbH_LT_Vars=Sol[0][LT_Vars]
        TbV_LT_Vars=Sol[1][LT_Vars]
        print("%d |    Sol (H|V):\t%.3f  \t%.3f\t| \t%.3f \t%.3f" %(lt, TbH_LT_Vars.mean(), TbH_LT_Vars.std(), TbV_LT_Vars.mean(), TbV_LT_Vars.std()))

def Compute_param(Sol,Context):    
    VType=Context['Dict_Vars']['VType']
    nlt=VType.max()+1

    tita={'H':{'mu':np.zeros(nlt),'sigma2':np.zeros(nlt)},\
          'V':{'mu':np.zeros(nlt),'sigma2':np.zeros(nlt)}}

    for lt in range(nlt):
        LT_Vars=np.where(VType==lt)[0]
        TbH_LT_Vars=Sol[0][LT_Vars]
        TbV_LT_Vars=Sol[1][LT_Vars]
        tita['H']['sigma2'][lt] = TbH_LT_Vars.var()
        tita['V']['sigma2'][lt] = TbV_LT_Vars.var()
        tita['H']['mu'][lt] = TbH_LT_Vars.mean()
        tita['V']['mu'][lt] = TbV_LT_Vars.mean()
    return tita


def Compute_NoPol_param(Sol,VType):    
    nlt=VType.max()+1

    tita={'mu':np.zeros(nlt),'sigma2':np.zeros(nlt)}

    for lt in range(nlt):
        LT_Vars=np.where(VType==lt)[0]
        Tb_LT_Vars=Sol[LT_Vars]
        tita['sigma2'][lt] = Tb_LT_Vars.var()
        tita['mu'][lt] = Tb_LT_Vars.mean()
    return tita

        
#compute_param(Sol,Observation)        
#%%
def multiply_sigma2(tita,l):    
    nlt=len(tita['H']['mu'])

    for lt in range(nlt):
        tita['H']['sigma2'][lt]*=l
        tita['V']['sigma2'][lt]*=l
    return tita



def recompute_param(Sol,Observation,tita):    
    VType=Observation['VType']
    nlt=VType.max()+1

    for lt in range(nlt):
        LT_Vars=np.where(VType==lt)[0]
        TbH_LT_Vars=Sol[0][LT_Vars]
        TbV_LT_Vars=Sol[1][LT_Vars]
        tita['H']['sigma2'][lt] = TbH_LT_Vars.var()
        tita['V']['sigma2'][lt] = TbV_LT_Vars.var()
        tita['H']['mu'][lt] = TbH_LT_Vars.mean()
        tita['V']['mu'][lt] = TbV_LT_Vars.mean()
    return tita




        
def recompute_sigma2(Sol,Observation,tita):
    VType=Observation['VType']
    nlt=VType.max()+1

    for lt in range(nlt):
        LT_Vars=np.where(VType==lt)[0]
        TbH_LT_Vars=Sol[0][LT_Vars]
        TbV_LT_Vars=Sol[1][LT_Vars]
        tita['H']['sigma2'][lt] = TbH_LT_Vars.var()
        tita['V']['sigma2'][lt] = TbV_LT_Vars.var()
    return tita
    
def minSigma2(tita,min_s2):
    for lt in range(tita['H']['sigma2'].shape[0]):
        if (tita['H']['sigma2'][lt] <  min_s2): tita['H']['sigma2'][lt] = min_s2
        if (tita['V']['sigma2'][lt] <  min_s2): tita['V']['sigma2'][lt] = min_s2
    return tita
    

#%%############################################################################
#%                                                                           %#
#%                                                                           %#
#%   Compute observation params                                              %#
#%                                                                           %#
##%############################################################################
def find_pure_ell_param(Observation, tita):
    #print ("Computing Ellipses params for lt %d (%s)"%(lt,pol))
    VType=Observation['VType']
    nlt=VType.max()+1
    thr=.999
    min_ell=10
    LandPropEll=Observation['LandPropEll']
    tita={'H':{'mu':np.zeros(nlt),'sigma2':np.zeros(nlt)},\
          'V':{'mu':np.zeros(nlt),'sigma2':np.zeros(nlt)}}
    
    for lt in range(nlt):
        Ells_LT=np.where(LandPropEll[:,lt]>thr)[0]
        if (len(Ells_LT)>min_ell): #enough pure?        
            print ("Found %d pure Ellipses for land_type %d. Computing observation stats" %(len(Ells_LT),lt))
            for pol in ['H','V']:
                Tb=Observation['Tb'][pol][Ells_LT]
                tita[pol]['mu'][lt]     = Tb.mean()            
                tita[pol]['sigma2'][lt] = Tb.var()
        else:
            print ("Only %d pure ellipses found, not computing stats in this way" %(len(Ells_LT)))
    return tita
    
def compute_params_from_pure_ells(Observation,tita):
    return find_pure_ell_param(Observation, tita)
#    tita=find_pure_ell_param(Observation, tita)
#    VType=Observation['VType']
#    nlt=VType.max()+1
#    Wt=Observation['Wt']
#    W=Wt*Wt
#    corr=1/(W.sum(axis=1)).mean()
#    
#    for pol in ['H','V']:
#        for lt in range(nlt):
#            tita[pol]['sigma2']*=corr
#    return tita














#%%############################################################################
#%                                                                           %#
#%                                                                           %#
#%   OLD STUFF                                                               %#
#%                                                                           %#
##%############################################################################
#%%############################################################################
#%                                                                          %%#
#%                                                                          %%#
#   Evaluate modified log likelyhood   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%                                                                          %%#
#%#############################################################################
def eval_mLL(Observation, pol,sel_ells, mu, sigma2):
    damp=0.8
    VType=Observation['VType']
    n_Vars=len(VType)
    D=np.zeros([n_Vars,n_Vars])
    Wt_sel=Observation['Wt'][sel_ells,:]
    for v in range(n_Vars):
        D[v,v]=abs(sigma2[VType[v]])
    I=np.eye(len(sel_ells))
    Sigma=Wt_sel.dot(D.dot(Wt_sel.transpose())) + damp*I # ¿?
    InvSigma=InvSym(Sigma)
    
    mu_v=mu[VType]
    mu_e=Wt_sel.dot(mu_v)
    Tb=Observation['Tb'][pol][sel_ells]
    dif=Tb-mu_e
    #print "·",
    
    rta=dif.transpose().dot(InvSigma.dot(dif)) + np.linalg.slogdet(Sigma)[1]
    #print ("mu:",mu, "\tsigma2:",sigma2, "\t mLL:",rta)
    #S.append(sigma2[0])
    #L.append(rta)
    return rta

#%%%% Evaluation with a single changing parameter
def eval_mLL_mlt(m1,Observation, pol,sel_ells, mu, sigma2,lt):
    m0=m1[0]
    mu[lt]=m0
    ll=eval_mLL(Observation, pol,sel_ells, mu, sigma2)
    return ll

def eval_mLL_slt(s1,Observation, pol,sel_ells, mu, sigma2,lt):
    s0=s1[0]
    sigma2[lt]=s0
    ll=eval_mLL(Observation, pol,sel_ells, mu, sigma2)
    #print sigma2, ll
    return ll

#%%%% Evaluation with a all parameters (mu or sigma2) changing 
def eval_mLL_mus(mu,Observation, pol,sel_ells, sigma2):
    ll=eval_mLL(Observation, pol,sel_ells, mu, sigma2)
    #print sigma2, ll
    return ll

def eval_mLL_sigma2s(sigma2,Observation, pol,sel_ells, mu):
    ll=eval_mLL(Observation, pol,sel_ells, mu, sigma2)
    #print sigma2, ll
    return ll



###############################################################################
#%%                                                                         %%#
#%%                                                                         %%#
#   Numeric solutions for sigma2 in the case "multiple variable types"   %%%%%%
#%%                                                                         %%#
#%%############################################################################
def solve_one_lt_sigma2_by_ML(Observation,mu,sigma2,s0,pol,lt):
    print ("Solving sigma2 for lt %d (%s)"%(lt,pol))
    min_ell=100
    max_ell=300
    selection=select_ells_for_lt(Observation,lt,min_ell=min_ell,max_ell=max_ell)
    asw=selection[0]
    sel_ells=selection[2]
    if asw==1:
        print ("Impossible to find enough ellipses. Skipping computation of sigma2.")
        return 0
    elif asw==4:
        n_rep=int(np.ceil(3*len(sel_ells)/max_ell))
        print ("Too many pure ellipses (%d>%d), subsampling" %(selection[1], max_ell), n_rep, "times...")
        S2=np.zeros(n_rep)
        for i in range(n_rep):
             sel_max_ells=random.sample(list(sel_ells),  max_ell)  
             res=minimize(eval_mLL_slt,[s0],args=(Observation, pol,sel_max_ells, mu, sigma2,lt),method='Nelder-Mead', tol=1e-3)        
             S2[i]=res['x'][0]
             s0=S2[i] #next star where we left
             print ("%d/%d:"%(i+1,n_rep),res['message'],res['nit'],"iterations:",S2[i])
        s0=S2.mean()
        print ("Finished solving sigma2 for lt %d (%s): %.2f"%(lt,pol,s0))
        #print (s0)
        return abs(s0)
    elif asw==3: #found a good sample or a small reasonable sample
        print ("Found a good sample of ellipses:")
    elif asw==2: #found a good sample or a small reasonable sample
        print ("Found a small sample of ellipses:")
    else:
        print ("Code not understood  *** problems *** not computing sigma2")
        return 0

    res=minimize(eval_mLL_slt,[s0],args=(Observation, pol,sel_ells, mu, sigma2,lt),method='Nelder-Mead', tol=1e-3)        
    s0=res['x'][0]
    #return "sigma2 =",s0
    return abs(s0)
    
def compute_params_using_singleML(Observation,tita):
    VType=Observation['VType']
    nlt=VType.max()+1
    for pol in ['H','V']:
        print ("pol", pol)
        mu=tita[pol]['mu']
        sigma2=tita[pol]['sigma2']
        for lt in range(nlt):
            #if (sigma2[lt]==None):
                sigma2[lt]=100
        for lt in range(nlt):
            sigma2[lt]=abs(solve_one_lt_sigma2_by_ML(Observation,mu,sigma2,sigma2[lt],pol,lt))
        tita[pol]['sigma2']=np.array(sigma2)
    return tita

#%%%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def compute_missing_sigma2_using_ML(Observation,tita):
    VType=Observation['VType']
    nlt=VType.max()+1
    for pol in ['H','V']:
        print ("pol", pol)
        mu=tita[pol]['mu']
        sigma2=tita[pol]['sigma2']
        n_missing=0
        for lt in range(nlt):
            if sigma2[lt]==None: 
                missing_lt=lt
                n_missing+=1
        if n_missing>1:
            print ("too many sigmas missing, dont know how to solve")
        elif n_missing==0:
            print ("All parameters already found!")
        else:#only one missing
            print ("Finding sigma2 for landtype", missing_lt, "pol:", pol)
            s0=50 # ¿?
            s2=solve_one_lt_sigma2_by_ML(Observation,mu,sigma2,s0,pol,missing_lt)
            tita[pol]['sigma2'][missing_lt]=s2
    return tita

def compute_missing_mu_using_ML(Observation,tita):
    VType=Observation['VType']
    nlt=VType.max()+1
    for pol in ['H','V']:
        print ("pol", pol)
        mu=tita[pol]['mu']
        #sigma2=tita[pol]['sigma2']
        n_missing=0
        for lt in range(nlt):
            if mu[lt]==None: 
                missing_lt=lt
                n_missing+=1
        if n_missing>1:
            print ("too many sigmas missing, dont know how to solve")
        elif n_missing==0:
            print ("All parameters already found!")
        else:#only one missing
            print ("Finding mu for landtype", missing_lt, "pol:", pol)
            #m0=200 # ¿?
            m2=1# FRUTA! falta escribir la funcion solve_one_lt_mu_by_ML(Observation,mu,sigma2,m0,pol,missing_lt)
            tita[pol]['mu'][missing_lt]=m2
    return tita


def solve_all_sigma2_by_ML(Observation,mu,pol):
    max_ell=300
    n_ell=Observation['Wt'].shape[0]
    VType=Observation['VType']
    nlt=VType.max()+1
    n_rep=int(np.ceil(3*n_ell/max_ell))
    print ("Subsampling (%d>%d)" %(n_ell, max_ell), n_rep, "times...")
    s0=np.repeat(100,nlt) # ¿?
    S2=np.zeros([n_rep,nlt])
    for i in range(n_rep):
         sel_max_ells=random.sample(range(n_ell),  max_ell)  
         res=minimize(eval_mLL_sigma2s,s0,args=(Observation, pol,sel_max_ells, mu),method='Nelder-Mead', tol=1e-3)        
         S2[i,:]=res['x']
         s0=S2[i,:] #next star where we left
         print (res['message'],res['nit'],"iterations:",S2[i,:])
    s0=S2.mean(axis=0)
    print (s0)

#%%############################################################################
#%                                                                          %%#
#%                                                                          %%#
#   Exact solutions for the case "1 variable type"   %%%%%%%%%%%%%%%%%%%%%%%%%#
#%                                                                          %%#
#%#############################################################################
def InvSym(Sigma):
    #Sigma is a symetric matrix, InvSigma should also be. 
    #We fix numeric problems coping LowerTringle into UpperTriangle
    InvSigma=np.linalg.inv(Sigma)
    math_symm = (np.triu_indices(len(InvSigma), 1))
    InvSigma[math_symm]=np.tril(InvSigma, -1).T[math_symm]
    return InvSigma

def mu_1vtype(Observation, pol,sel_ells): 
    #compute mu for one land type with enought ellipses
    Wt_sel=Observation['Wt'][sel_ells,:]
    Sigma=Wt_sel.dot(Wt_sel.transpose())
    InvSigma=InvSym(Sigma)
    
    Tb=Observation['Tb'][pol][sel_ells]
    return InvSigma.dot(Tb).sum()/InvSigma.sum()



def sigma2_1vtype(Observation, pol,sel_ells, mu0): 
    #compute sigma2 for one land type with enought ellipses, given mu
    Wt_sel=Observation['Wt'][sel_ells,:]
    Sigma=Wt_sel.dot(Wt_sel.transpose())
    InvSigma=InvSym(Sigma)
    
    Tb=Observation['Tb'][pol][sel_ells]
    dif=Tb-mu0
    
    return dif.dot(InvSigma.dot(dif))/len(sel_ells)
    

def compute_singleML_params(Observation,lt): 
    #compute ML mu and sigma2 for a landtype with enough ellipses

    #select ellipses to work with
    min_ells=100
    max_ells=300
   
    LandPropEll=Observation['LandPropEll']
    Ells_LT=np.where(LandPropEll[:,lt]>.999)[0]
    if (len(Ells_LT))<min_ells:
        print ("Not enough pure ellipses to compute ML params for land type, will skip and compute later", lt)
        return [None,None,None,None]
    else:
        print ("Computing ML params for land type", lt)
        print ("mean...",end=" ")
        if len(Ells_LT)>max_ells:
           mu_rep=int(np.ceil(10*len(Ells_LT)/max_ells))
           MUh=np.zeros(mu_rep)
           MUv=np.zeros(mu_rep)
           for i in range(mu_rep):
                sel_ells=random.sample(list(Ells_LT),  max_ells)  
                MUh[i]=mu_1vtype(Observation, 'H',sel_ells)
                MUv[i]=mu_1vtype(Observation, 'V',sel_ells)
           mu_h=MUh.mean()
           mu_v=MUv.mean()
        else:
           mu_h=mu_1vtype(Observation, 'H',Ells_LT)
           mu_v=mu_1vtype(Observation, 'V',Ells_LT)
            
        print ("\t H:",mu_h, "\t\t V:",mu_v)
        print ("variance...",end=" ")

        if len(Ells_LT)>max_ells:
            s2_rep=int(np.ceil(10*len(Ells_LT)/max_ells))
            S2h=np.zeros(s2_rep)
            S2v=np.zeros(s2_rep)
            for i in range(s2_rep):
                sel_ells=random.sample(list(Ells_LT),  max_ells)  
                S2h[i]=sigma2_1vtype(Observation, 'H',sel_ells,mu_h)
                S2v[i]=sigma2_1vtype(Observation, 'V',sel_ells,mu_v)
            sigma2_h=S2h.mean()
            sigma2_v=S2v.mean()
        else:
            sigma2_h=sigma2_1vtype(Observation, 'H',Ells_LT,mu_h)
            sigma2_v=sigma2_1vtype(Observation, 'V',Ells_LT,mu_v)

        print ("\t H:",sigma2_h,"\t\t V:",sigma2_v)
        return mu_h,mu_v,sigma2_h,sigma2_v
        

def compute_possible_singleML_params(Observation):
    #compute ML (mu, sigma2) for all landtypes with enough ellipses
    VType=Observation['VType']
    nlt=VType.max()+1
    tita={'H':{'mu':np.array([None]*nlt),'sigma2':np.array([None]*nlt)},\
          'V':{'mu':np.array([None]*nlt),'sigma2':np.array([None]*nlt)}}

    for lt in range(nlt):
        mu_h,mu_v,sigma2_h,sigma2_v = compute_singleML_params(Observation,lt)
        tita['H']['mu'][lt]     = mu_h
        tita['H']['sigma2'][lt] = sigma2_h
        tita['V']['mu'][lt]     = mu_v
        tita['V']['sigma2'][lt] = sigma2_v
    
    return tita    


#%%############################################################################
#%                                                                          %%#
#%                                                                          %%#
#%  Ellipses selection for a particular land type   %%%%%%%%%%%%%%%%%%%%%%%%%%#
#%                                                                          %%#
#%#############################################################################
# 1 = impossible
# 2 = a few (>min_ell, <max_ell)
# 3 = OK ~ max_ell
# 4 = too many (<max_ell), use samples
def select_ells_for_lt(Observation,lt,min_ell=100,max_ell=300):
    #select a sample of ~max_ell intersecting land_type
    #if there are too many (>max_ell) pure, return them all,
    #if there are too few with at least mthr in lt, return only them
    #otherwise, typical situation: select thr so that #(Ell int LT > thr)~max_ell
    thr=.999
    mthr=0.00001
    enough=False
    
    LandPropEll=Observation['LandPropEll']
    Ells_LT=np.where(LandPropEll[:,lt]>thr)[0]
    if (len(Ells_LT)>max_ell): #enough pure?
        #print "found enough pure ellipses for land type", lt
        #print "Why are we using this slow method??"
        #print "Proceeding..."
        return [4,len(Ells_LT),Ells_LT]
        
    while ((thr>mthr) and (not enough)):
        Ells_LT=np.where(LandPropEll[:,lt]>thr)[0] #consider ellipses that have at least thr=a0/2^i of lt
        if (len(Ells_LT)>max_ell):
            enough=True
        else:
            thr/=2
    if enough:
        #if (len(Ells_LT)<max_ell):
        #    #print "Found a good sample of ellipses, proceeding..."
        #    return [2, sel_ells]
        #else:
            print ("Not enough pure ellpises, but a lot of mixed ellipses, selecting purest...")
            mthr=thr               #this is the typical situation for the routine:
            Mthr=2*thr             #not enough pure but a lot of mixed ellipses.
            enough=False           #=> find thr to keep between 99% and 101% of max_ell
            while (not enough):    #
                thr=(mthr+Mthr)/2
                Ells_LT=np.where(LandPropEll[:,lt]>thr)[0]
                if (len(Ells_LT)<0.99*max_ell):
                    Mthr=thr
                elif (len(Ells_LT)>1.01*max_ell):
                    mthr=thr
                else:
                    enough=True
                    print ("Done selecting.")
            return [3,len(Ells_LT),Ells_LT]
    else:
        print ("Found too few ellipses intersecting land type", lt)
        if len(Ells_LT)>2:
          print (" proceeding anyway...")
          return [2, len(Ells_LT), Ells_LT]
        else:
          print ("Impossible to continue")
          return [1, 0, None]
    return [0, len(Ells_LT), Ells_LT]   #should return before this line
#%%    
    
def select_ells_for_lt_from_subset(Observation,lt,S,min_ell=100,max_ell=300):
    #select a sample of ~max_ell intersecting land_type from S
    #if there are too many (>max_ell) pure, return them all,
    #if there are too few with at least mthr in lt, return only them
    #otherwise, typical situation: select thr so that #(Ell int LT > thr)~max_ell
    thr=.999
    mthr=0.00001
    enough=False
    
    LandPropEll=Observation['LandPropEll']
    Ells_LT=np.where(LandPropEll[S,lt]>thr)[0]
    if (len(Ells_LT)>max_ell): #enough pure?
        #print "found enough pure ellipses for land type", lt
        #print "Why are we using this slow method??"
        #print "Proceeding..."
        return [4,len(Ells_LT),Ells_LT]
        
    while ((thr>mthr) and (not enough)):
        Ells_LT=np.where(LandPropEll[S,lt]>thr)[0] #consider ellipses that have at least thr=a0/2^i of lt
        if (len(Ells_LT)>max_ell):
            enough=True
        else:
            thr/=2
    if enough:
        #if (len(Ells_LT)<max_ell):
        #    #print "Found a good sample of ellipses, proceeding..."
        #    return [2, sel_ells]
        #else:
            print ("Not enough pure ellpises, but a lot of mixed ellipses, selecting purest...")
            mthr=thr               #this is the typical situation for the routine:
            Mthr=2*thr             #not enough pure but a lot of mixed ellipses.
            enough=False           #=> find thr to keep between 99% and 101% of max_ell
            while (not enough):    #
                thr=(mthr+Mthr)/2
                Ells_LT=np.where(LandPropEll[S,lt]>thr)[0]
                if (len(Ells_LT)<0.99*max_ell):
                    Mthr=thr
                elif (len(Ells_LT)>1.01*max_ell):
                    mthr=thr
                else:
                    enough=True
                    print ("Done selecting.")
            return [3,len(Ells_LT),Ells_LT]
    else:
        print ("Found too few ellipses intersecting land type", lt)
        if len(Ells_LT)>2:
          print (" proceeding anyway...")
          return [2, len(Ells_LT), Ells_LT]
        else:
          print ("Impossible to continue")
          return [1, 0, None]
    return [0, len(Ells_LT), Ells_LT]   #should return before this line
#%%    

#%% VIEJO
def Compute_Variance_Correction(HPDM,HPDm, dx, dy,rows, cols):
    RelSigma2HalfWidthDiam2=8*np.log(2) #Sigma^2=D^2/(8*ln2)
    k=10.0
    x, y = np.mgrid[0:dx*cols:dx/k, 0:dy*rows:dy/k]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    
    mu=[dx*(k*cols-1)/2.0/k,dy*(k*rows-1)/2.0/k]
    #tilt=0
    Sigma=np.array([[HPDm**2/RelSigma2HalfWidthDiam2, 0.0], [0.0, HPDM**2/RelSigma2HalfWidthDiam2]])
    rv = multivariate_normal(mu, Sigma)
    FineGrid=rv.pdf(pos)*dx*dy/k/k
    Grid=np.zeros([cols,rows])
    for u in range(cols):
        for v in range(rows):
            Grid[u,v]=FineGrid[u*10:u*10+10,v*10:v*10+10].sum()
    Grid/=Grid.sum()
    

    return 1/((Grid*Grid).sum())
