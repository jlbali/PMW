# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 13:27:16 2017

@author: rgrimson
"""

from __future__ import division
import numpy as np
import sys
import libs.L_Kernels as L_Kernels
from scipy.sparse.linalg import lsqr

#%%############################################################################
#%                                                                           %#
#%                                                                           %#
#%                                                                           %#
#%                                                                           %#
##%############################################################################
#%%
def GeoCorrectBands(Dict_Bands, Dict_Grids):
    
    WA_Cords=Dict_Bands['WA_Cords']
    for Band in Dict_Bands['Bands']:
        print ("***************")
        print ("GeoCorrecting Band", Band)
        Dict_Band = Dict_Bands[Band]
        GeoCorrect_Band(Dict_Band,WA_Cords,Dict_Grids)


#%%############################################################################
#%                                                                           %#
#%                                                                           %#
#%                                                                           %#
#%                                                                           %#
##%############################################################################
#%% Correct Georreference
def GeoCorrect_Band(Dict_Band,WA_Cords,Dict_Grids):
    print ("  Correcting HDF georreference:")
    LandPropMargin_Grid=Dict_Grids['LandTypeMargin_Grid']
    GeoCorr_margin=Dict_Grids['GeoCorr_margin']
    dx=WA_Cords['dx']
    dy=WA_Cords['dy']
    cols=WA_Cords['cols']
    rows=WA_Cords['rows']
    nlt=Dict_Grids['nlt']
    #Select ellipses that intersect at least 90% the ROI
    Orig_Ells=Dict_Band['Ell'].copy()
    Orig_n_ell=Dict_Band['n_ell']
    
    Grid=L_Kernels.Compute_Elliptic_Kernels(Dict_Band,WA_Cords,MinimalProp=0.9)
    n_ell=Dict_Band['n_ell']
    Band=Dict_Band['Band']
        

    max_error=GeoCorr_margin*dx

    mdsp_x=0
    ctr_x=int(max_error/dx)
    dsp_x=ctr_x
    Mdsp_x=2*dsp_x
    ndsp_x=Mdsp_x-mdsp_x
    
    mdsp_y=0
    ctr_y=int(max_error/dy)
    dsp_y=ctr_y
    Mdsp_y=2*dsp_y
    ndsp_y=Mdsp_y-mdsp_y
    Score=np.zeros([ndsp_x,ndsp_y,2]) #2 because of the H&V polarizations
    M=np.zeros([n_ell,nlt]) #one equation for ellipse, one varible for land_type
    bh=np.zeros(n_ell)
    bv=np.zeros(n_ell)
    
    
    found=False
    ok_corr=True
    min_score=65500

    #%% FIND GEO CORRECTION    , step 1
    while (not found):
        #evaluate arround dsp_x, dsp_y
        print ("      Evaluating displacement: %dmH, %dmV" %((dsp_x-ctr_x)*dx, (dsp_y-ctr_y)*dy))
        for evx, evy in [[dsp_x-1,dsp_y],[dsp_x+1,dsp_y],[dsp_x,dsp_y-1],[dsp_x,dsp_y+1],[dsp_x,dsp_y]]:
            if (Score[evx,evy,0]==0): # not yet evaluated
                U=LandPropMargin_Grid[evx:evx+cols, evy:evy+rows]
                for i in range(n_ell):
                  for lt in range(nlt):  
                    M[i,lt]=(Grid[:,:,i]*(U[:,:]==lt)).sum()
                  bh[i]=Dict_Band['Ell'][i]['Tbh']
                S=lsqr(M,bh)
                s=S[3]
                Score[evx,evy,0]=s
                if (s<min_score):
                  min_score=s
                  best_x=evx
                  best_y=evy
            else:                    # already evaluated
                s=Score[evx,evy,0]
                if (s<min_score):
                  min_score=s
                  best_x=evx
                  best_y=evy
    
        if ((dsp_x!=best_x) or (dsp_y!=best_y)):
            if ((best_x<mdsp_x+2) or (best_x>Mdsp_x-3) or (best_y<mdsp_y+2) or (best_y>Mdsp_y-3)):
                print ("Reached the border when correcting georreference!!")
                print ("Please, manually look at the image"               )
                print ("Band", Band)
                print ("**************************************************")
                found=True
                ok_corr=False  #geocorrection is not good
            else:
                dsp_x=best_x
                dsp_y=best_y
        else:    
            found=True
            print ("    Found best geocorrection with %dmts error margin, improving correction..." %(dx/2))
      
    #%% FIND GEO CORRECTION    , step 2          
    #found dsp_x, dsp_y        
    # EVALUATE AT:     +  
    #                  +  
    #              + + o + +  
    #                  +  
    #                  +  
    for evx, evy in [[dsp_x-2,dsp_y],[dsp_x-1,dsp_y],[dsp_x,dsp_y],[dsp_x+1,dsp_y],[dsp_x+2,dsp_y]]:
        if (Score[evx,evy,1]==0): # not yet evaluated
            U=LandPropMargin_Grid[evx:evx+cols, evy:evy+rows]
            for i in range(n_ell):
                for lt in range(nlt):  
                  M[i,lt]=(Grid[:,:,i]*(U[:,:]==lt)).sum()  
                bh[i]=Dict_Band['Ell'][i]['Tbh']
                bv[i]=Dict_Band['Ell'][i]['Tbv']
            S=lsqr(M,bh)
            s=S[3]
            Score[evx,evy,0]=s
            S=lsqr(M,bv)
            s=S[3]
            Score[evx,evy,1]=s
    
    for evx, evy in [[dsp_x,dsp_y-2],[dsp_x,dsp_y-1],[dsp_x,dsp_y],[dsp_x,dsp_y+1],[dsp_x,dsp_y+2]]:
        if (Score[evx,evy,1]==0): # not yet evaluated
            U=LandPropMargin_Grid[evx:evx+cols, evy:evy+rows]
            for i in range(n_ell):
                for lt in range(nlt):  
                  M[i,lt]=(Grid[:,:,i]*(U[:,:]==lt)).sum()
                bh[i]=Dict_Band['Ell'][i]['Tbh']
                bv[i]=Dict_Band['Ell'][i]['Tbv']
            S=lsqr(M,bh)
            s=S[3]
            Score[evx,evy,0]=s
            S=lsqr(M,bv)
            s=S[3]
            Score[evx,evy,1]=s
    
    #%% adjust quadratic and find optimum
    x = np.array([dsp_x-2,dsp_x-1,dsp_x,dsp_x+1,dsp_x+2])
    sx = Score[dsp_x-2:dsp_x+3,dsp_y,0]
    cpx = np.polyfit(x, sx, 2)
    best_x = -cpx[1]/(2*cpx[0])
    y = np.array([dsp_y-2,dsp_y-1,dsp_y,dsp_y+1,dsp_y+2])
    sy = Score[dsp_x,dsp_y-2:dsp_y+3,0]
    cpy = np.polyfit(y, sy, 2)
    best_y = -cpy[1]/(2*cpy[0])
    corr_xh=best_x*dx-max_error
    corr_yh=best_y*dy-max_error
    #Found best correction for H POL
    #comparing with V POL
    sx = Score[dsp_x-2:dsp_x+3,dsp_y,1]
    cpx = np.polyfit(x, sx, 2)
    best_x = -cpx[1]/(2*cpx[0])
    sy = Score[dsp_x,dsp_y-2:dsp_y+3,1]
    cpy = np.polyfit(y, sy, 2)
    best_y = -cpy[1]/(2*cpy[0])
    corr_xv=best_x*dx-max_error
    corr_yv=best_y*dy-max_error
    
    corr_x=(corr_xh+corr_xv)/2
    corr_y=(corr_yh+corr_yv)/2
    corr_diff=np.sqrt((corr_xh-corr_xv)**2+(corr_yh-corr_yv)**2)
    print ("    Improved geocorrection: %.0fm, %.0fm (%.0fm from best corr for each pol)" %(corr_x,corr_y,corr_diff/2))
    if (corr_diff>3*dx):
        print ("WARNING: geocorrection for different polarizations do not agree,", corr_diff)
        print ("Please, manually look at the image")
        print ("****************************************************************")
        ok_corr=False  #geocorrection is not good
    sys.stdout.flush()


    Dict_Band['GeoCorrection']={'X':corr_x,'Y':corr_y,'OK':ok_corr}
    Dict_Band.pop('Ell')
    Dict_Band['Ell']=Orig_Ells
    Dict_Band['n_ell']=Orig_n_ell
    
   

#%%############################################################################
#%                                                                           %#
#%                                                                           %#
#%                                                                           %#
#%                                                                           %#
##%############################################################################


