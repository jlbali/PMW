# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 17:53:56 2017

@author: rgrimson

This library has to be written

TBD

Compute Kernels using ndimage shift and rotate
New library to replace L_Kernels.py
@author: rgrimson
"""

from __future__ import division
from __future__ import print_function
import numpy as np
from scipy.stats import multivariate_normal
import sys
from scipy.sparse.linalg import lsqr
import L_Context

from scipy.ndimage import rotate as imrotate
from scipy.ndimage import shift as imshift

#%%
import matplotlib.pyplot as plt

def show(z):
    plt.imshow(z[:,::-1].T,interpolation='none')

def All_K(z):
    for i in range(1475):
       if i%50==0:
          print(i,end=' ')
          sys.stdout.flush()

       imrotate(z,i)
###############################################################################
#%                                                                           %#
#%                                                                           %#
#%  COMPUTE ONE KERNEL                                                       %#
#%                                                                           %#
#%%############################################################################
def Compute_Elliptic_Kernel(Dict_Band,WA_Cords,MinimalProp=.999): 
    dx=WA_Cords['dx']
    dy=WA_Cords['dy']
    rows=WA_Cords['rows']
    cols=WA_Cords['cols']
    
    RelSigma2HalfWidthDiam2=8*np.log(2) #Sigma^2=D^2/(8*ln2)
    HPDm=Dict_Band['HPDm']
    HPDM=Dict_Band['HPDM']

    X1=np.round(-cols/2)*dx
    X2=(cols+1)/2*dx
    Y1=np.round(-rows/2)*dy
    Y2=(rows+1)/2*dy
    x, y = np.mgrid[X1:X2:dx, Y1:Y2:dy]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    
    print ("Computing Original Elliptic Kernel")
    Sigma=np.array([[HPDm**2/RelSigma2HalfWidthDiam2, 0.0], [0.0, HPDM**2/RelSigma2HalfWidthDiam2]])

    rv = multivariate_normal([0,0], Sigma)
    z=rv.pdf(pos)*dx*dy
    
    #CUT at MinimalProp
    tol=z.max()/2
    while z[np.where(z>tol)].sum()<MinimalProp:
        tol/=10
    while z[np.where(z>tol)].sum()>MinimalProp:
        tol*=1.01
    z[np.where(z<tol)]=0
    z=z/(z.sum())
    W=np.where(z!=0)
    X1=W[0].min()
    X2=W[0].max()+1
    Y1=W[1].min()
    Y2=W[1].max()+1
    K=z[X1:X2,Y1:Y2]
    #plt.contour(x, y, z)
    
    #Write kernel in square array with one margin for eay rotation and small (<1) shifts
    SH=np.array(K.shape)+[2,2]
    if SH[1]>SH[0]:
        d=int((SH[1]-SH[0])/2)
        K2=np.zeros([SH[1],SH[1]])
        K2[d+1:-1-d,1:-1]=K
    else:
        d=int((SH[0]-SH[1])/2)
        K2=np.zeros([SH[0],SH[0]])
        K2[1:-1,d+1:-1-d]=K

    Dict_Band['K']=K2

def Compute_ith_Elliptic_Kernel(Dict_Band,WA_Cords,i,GC_X=0,GC_Y=0): 
    X1=WA_Cords['X1']
    Y1=WA_Cords['Y1']
    dx=WA_Cords['dx']
    dy=WA_Cords['dy']
    rows=WA_Cords['rows']
    cols=WA_Cords['cols']
    Grid=np.zeros([cols,rows])

    #set K as default kernel    
    Dict_Ell=Dict_Band['Ell'][i]            
    mu=np.array([(Dict_Ell['X']+GC_X-(X1+dx/2))/dx, (Dict_Ell['Y']+GC_Y-(Y1+dy/2))/dy])
    mu_i=[int(round(mu[0])),int(round(mu[1]))]
    mu_f=[mu[0]-mu_i[0], mu[1]-mu_i[1]]
    tilt=Dict_Ell['tilt'] #degrees
    #rotate and shift it subpixel distance
    K=imrotate(Dict_Band['K'],tilt,reshape=False,order=1)
    K=imshift(K,mu_f,order=1)
    #locate it in the big picture
    d=int(K.shape[0]/2)
    Grid=np.zeros([cols,rows])
    Grid[max(0,mu_i[0]-d):mu_i[0]+d+1,max(0,mu_i[1]-d):mu_i[1]+d+1]=K[max(0,d-mu_i[0]):2*d+1+cols-(mu_i[0]+d+1),max(0,d-mu_i[1]):2*d+1+rows-(mu_i[1]+d+1)]
    return Grid
        
def Create_Fake_Elliptic_Kernel():
    
    RelSigma2HalfWidthDiam2=8*np.log(2) #Sigma^2=D^2/(8*ln2)
    dx=1000
    dy=dx
    HPDm=7000
    HPDM=12000
    rows=40
    cols=40
    X1=-cols/2*dx
    X2=cols/2*dx
    Y1=-rows/2*dy
    Y2=rows/2*dy
    
    x, y = np.mgrid[X1+dx/2:X2:dx, Y1+dy/2:Y2:dy]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    Mu=[0,0]
    Sigma=np.array([[HPDm**2/RelSigma2HalfWidthDiam2, 0.0], [0.0, HPDM**2/RelSigma2HalfWidthDiam2]])
    rv = multivariate_normal(Mu, Sigma)
    z=rv.pdf(pos)*dx*dy
    return z

###############################################################################
#%                                                                           %#
#%                                                                           %#
#%  COMPUTE KERNELS                                                          %#
#%                                                                           %#
#%%############################################################################
#%% Compute geocorrected Elliptic Kernels
def Clean_Dict_Band(Dict_Band, cod):
    Dict_Ell={}
    for i in range(len(cod)):
        Dict_Ell[i]=Dict_Band['Ell'][cod[i]]
    Dict_Band.pop('Ell')
    Dict_Band['Ell']=Dict_Ell
    Dict_Band['n_ell']=len(cod)    

def Compute_Elliptic_Kernels(Dict_Band,WA_Cords,MinimalProp=0,GC_X=0,GC_Y=0): 
    X1=WA_Cords['X1']
    Y1=WA_Cords['Y1']
    X2=WA_Cords['X2']
    Y2=WA_Cords['Y2']
    dx=WA_Cords['dx']
    dy=WA_Cords['dy']
    rows=WA_Cords['rows']
    cols=WA_Cords['cols']
    
    n_ell=Dict_Band['n_ell']
    Grid=np.zeros([cols,rows,n_ell])
    x, y = np.mgrid[X1+dx/2:X2:dx, Y1+dy/2:Y2:dy]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    
    print ("    Computing New Elliptic Kernels: 0%",end=" ")
    pct=0
    for i in range(n_ell):
        if int(i*10/n_ell)>pct: #print advance
            pct=int(i*10/n_ell)
            print (pct*10, "%",end=" ")
            sys.stdout.flush()
        Grid[:,:,i]=Compute_ith_Elliptic_Kernel(Dict_Band,WA_Cords,i,GC_X=GC_X,GC_Y=GC_Y)
    print ("100%")
    sys.stdout.flush()
    
    print ("    Selecting Kernels with %.2fpct inside region"%(MinimalProp*100),end=" ")
    sys.stdout.flush()

    S=Grid.sum(axis=0).sum(axis=0)
    ELL=(Grid.sum(axis=0).sum(axis=0)>=MinimalProp)
    n_ell_B=ELL.sum()
    S=S[ELL]
    Grid=Grid[:,:,ELL]/S #Normalize Kernels
    cod=np.where(ELL)[0] #codes for ellipses Grid[i] ---> Dict[cod[i]]
    print ("    Using %d ellipses out of %d" %(n_ell_B, n_ell))
    sys.stdout.flush()
    if n_ell_B<n_ell: 
        Clean_Dict_Band(Dict_Band, cod)
    return Grid


def Compute_Kernels(Dict_Bands,Band, Context,Mode='Grid',MinimalProp=0.9): #Grid mode default
    if not Dict_Bands[Band]['GeoCorrection']['OK']:
        return {},{}
        
    Band=Band.upper()
    Dict_Band=Dict_Bands[Band]
    FileName=Dict_Bands['FileName']
    Dict_Grids=Context['Dict_Grids']
    Dict_Vars=Context['Dict_Vars']
    WA_Cords=Context['WA_Cords']

    LandType_Grid=Dict_Grids['LandType_Grid']
    CIdG=Dict_Grids['CellID_Grid']
    
    n_Vars=Dict_Vars['n_Vars']
    ROWS=WA_Cords['ROWS']
    nx=int(Context['WA_Cords']['DX']/Context['WA_Cords']['dx'])
    ny=int(Context['WA_Cords']['DY']/Context['WA_Cords']['dy'])
    
    #nlt=Dict_Vars['n_lt']
    
    
    #print ("  Computing Intersections for ellipses and grid cells")
    #sys.stdout.flush()
    #Compute  kernels
    GC_X=Dict_Band['GeoCorrection']['X']
    GC_Y=Dict_Band['GeoCorrection']['Y']
    
    EllK_Grid=Compute_Elliptic_Kernels(Dict_Band,WA_Cords,MinimalProp=MinimalProp,GC_X=GC_X,GC_Y=GC_Y)
    #import matplotlib.pyplot as plt
    #I=(EllK_Grid*Tbh).sum(axis=2)
    #plt.imshow(I)
    #In=I/(EllK_Grid.sum(axis=2))
    #plt.imshow(In)
    #plt.imshow(In[:,::-1].transpose())
    
    n_ell=Dict_Band['n_ell']
    ##%%
    print ("  Computing Intersections for ellipses and grid cells, Band %s"%Band)
    sys.stdout.flush()
    
    Wt=np.zeros([n_ell,n_Vars]) #inter       Wt[i,w]+=(EllK_Grid[u*nx:(u+1)*nx,v*ny:(v+1)*ny,i]*LandPsection for ellipses and cells
    if Mode=='Grid':
        print("  Total %d Cells:"%n_Vars, end=' ')
        for w in range(n_Vars):     #and variable
            if w%50==0:
                print(w,end=' ')
                sys.stdout.flush()
            WCell=np.where(CIdG==w)
            lt=Dict_Vars['Var'][w]['Land_type']
            EllK_Cell=EllK_Grid[WCell].sum(axis=0)
            Wt[:,w]=EllK_Cell
    if Mode=='Grid_OLD':
        for i in range(n_ell):        # for every ellipse...
            if i%50==0:
              print(i,end=' ')
              sys.stdout.flush()
            for w in range(n_Vars):     #and variable
                lt=Dict_Vars['Var'][w]['Land_type']
                for u,v in Dict_Vars['Var'][w]['Coords']: #sum over all cells for this var
                    v=ROWS-v-1
                    Wt[i,w]+=(EllK_Grid[u*nx:(u+1)*nx,v*ny:(v+1)*ny,i]*(LandType_Grid[u*nx:(u+1)*nx,v*ny:(v+1)*ny]==lt)).sum()   #LAND
    elif Mode=='Cell':
        
        for i in range(n_ell):        # for every ellipse...
            if i%50==0:
                print(i,end=' ')
                sys.stdout.flush()
            for v in range(n_Vars):     #and variable
                Wt[i,v]=(EllK_Grid[:,::-1,i].transpose()*(CIdG==v)).sum()   #find where LTP comes from :)
        
    #normalize elliptic kernels
    Wt=(Wt.transpose()/Wt.sum(axis=1)).transpose()
    
    nlt=Dict_Vars['nlt']
    
    Tbh=np.zeros(n_ell)
    Tbv=np.zeros(n_ell)
    for i in range(n_ell):
        Tbh[i]=Dict_Band['Ell'][i]['Tbh']
        Tbv[i]=Dict_Band['Ell'][i]['Tbv']
    
    
    VType=Dict_Vars['VType']
    
    M=np.zeros([n_ell,nlt])
    for i in range(n_ell):
        for lt in range(nlt):  
          M[i,lt]=(EllK_Grid[:,:,i]*(LandType_Grid==lt)).sum()


    #compute observation picture
    #import matplotlib.pyplot as plt
    print ("Computing means for each land type by lsqr")
    Sh=lsqr(M,Tbh)
    Sv=lsqr(M,Tbv)
    
    print ("  Sol H (mu for each land type, LSQR):", Sh[0])
    print ("  Sol V (mu for each land type, LSQR):", Sv[0])
    
    
    tita={'H':{'mu':Sh[0],'sigma2':np.zeros(nlt)},\
          'V':{'mu':Sv[0],'sigma2':np.zeros(nlt)}}


    N=(EllK_Grid.sum(axis=2))
    IH=(EllK_Grid*Tbh).sum(axis=2)
    IV=(EllK_Grid*Tbv).sum(axis=2)
    #complete cells where ellipses do not provide information with lsqr mean
    W=np.where(N==0)
    N[W]=1
    IH[W]=tita['H']['mu'][LandType_Grid[W]]
    IV[W]=tita['V']['mu'][LandType_Grid[W]]
    IH/=N
    IV/=N
    
        #ASSIGNEMENT    
    Observation={
    'Tb'            :{'H':Tbh,'V':Tbv},
    'Wt'            :Wt,
    'VType'         :VType,
    'LandPropEll'   :M,
    'FileName'      :FileName,
    'Band'          :Band,
    'GeoCorrection' :Dict_Bands[Band]['GeoCorrection'],
    'Img'           :{'H':IH,'V':IV}}
    keep_grid=True
    if keep_grid:
        Observation['KGrid']=EllK_Grid
    Observation['LSQRSols']={'H':{'Sol':Sh[0],'itn':Sh[2],'norm':Sh[3]},'V':{'Sol':Sv[0],'itn':Sv[2],'norm':Sv[3]}}    
    
    print("   LandType approximation error:%.2f, %.2f"%(Observation['LSQRSols']['H']['norm'],Observation['LSQRSols']['V']['norm']))
    return Observation, tita

#%% VARIOS
def Varios():
    #%% BIVARIATE VERSION
    from scipy.ndimage import rotate as imrotate
    from scipy.ndimage import shift as imshift
    from scipy.stats import multivariate_normal
    R=1124 #distance from sensor to beam axis intersection with earth
    tita=55/180.0*np.pi
    HPDm=35
    RelSigma2HalfWidthDiam2=8*np.log(2) #Sigma^2=D^2/(8*ln2)
    VarS=HPDm**2/RelSigma2HalfWidthDiam2
    #StdS=np.sqrt(VarS)
    rv=multivariate_normal([0,0],[[VarS,0],[0,VarS]])
    
    
    
    #Local coordinates where power has to be found
    cols=201
    rows=101
    dx=1
    dy=1
    X1=-dx*cols/2
    X2=dx*cols/2
    Y1=-dy*rows/2
    Y2=dy*rows/2
    Yl, Xl = np.mgrid[Y1+dy/2:Y2:dy, X1+dx/2:X2:dx]
    pos = np.empty(Xl.shape + (2,))
    
    
    #Sensor local coordinates
    Xs=np.sin(tita)*R
    Ys=0
    Zs=np.cos(tita)*R
    
    
    #earth tangent plane points projected to the beam normal plane, in local coords
    l=-R/(R-np.sin(tita)*Xl)
    Xp=Xs+l*(Xs-Xl)
    Yp=l*Yl
    Zp=Zs+l*Zs
    #distance from zero:
    Rp=np.sqrt(Xp*Xp+Yp*Yp+Zp*Zp)
    #earth tangent plane points projected to the beam normal plane, in beam normal plane coords
    pos[:, :, 1] = np.sqrt(Xp**2+Zp**2); pos[:, :, 0] = Yp
    
    
    
    #Jacobian of the change of coordinates
    J=np.cos(tita)/np.sqrt(1+(Xl/R)**2+(Yl/R)**2+-2*Xl/R*np.sin(tita))
    
    
    
    #
    K0=rv.pdf(pos)*J
    K0=K0/K0.sum()
    plt.imshow(K0,interpolation='none',extent=(X1,X2,Y1,Y2))
    K0.sum()
    #%%rotate and shift
    from scipy.ndimage import rotate as imrotate
    from scipy.ndimage import shift as imshift
    
    X0=10
    Y0=2.534
    tilt=15.0
    K=imrotate(K0,tilt,order=1)    
    K=imshift(K,[Y0,X0],order=1)    
    plt.imshow(K,interpolation='none',extent=(X1,X2,Y1,Y2))
    #%%
    #3d plot
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    ax.plot_surface(Xl,Yl,K, color='b')
    
    plt.show()
    ##################################
    ###%
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    
    #%% VARIOS RESTOS
    import math
    import numpy as np
    import matplotlib.mlab as mlab
    import scipy
    
    #%%
    
    
    #Efecto del angulo sobre la gaussiana
    D=1140
    tita=55/180*np.pi
    x_E = np.linspace(-100,10075,31.25, 1000)
    y_S = x_E*np.cos(tita)
    x_S = x_E*np.sin(tita)+D
    phi=np.arctan(y_S/x_S)
    Y=np.tan(phi)
    mu = 0
    sigma = math.sqrt(variance)
    pdfY=mlab.normpdf(Y, mu, sigma)
    pdfE=mlab.normpdf(x_E/2000, mu, sigma)
    
    
    plt.plot(x_E,pdfY)
    plt.plot(x_E,pdfE)
    plt.show()
    #%%
    
    H=1124
    B=947.8
    h=604.2
    tita=(90-55)/180*np.pi
    x_E = np.linspace(-100,100, 1000)
    y_S = x_E*np.sin(tita)
    x_S = x_E*np.cos(tita)+D
    phi=np.arctan(y_S/x_S)
    Y=np.tan(phi)
    mu = 0
    sigma = math.sqrt(variance)
    pdfY=mlab.normpdf(Y, mu, sigma)
    pdfE=mlab.normpdf(x_E/2000, mu, sigma)
    
    
    plt.plot(x_E,pdfY)
    plt.plot(x_E,pdfE)
    plt.show()
    #%%
    R=1124
    #B=947.8
    #h=604.2
    x_l = np.linspace(-20,20, 1000)
    tita=55/180.0*np.pi
    
    #B
    x_s=(R*np.cos(tita))/(R/x_l-np.sin(tita))
    
    HPDm=7
    RelSigma2HalfWidthDiam2=8*np.log(2) #Sigma^2=D^2/(8*ln2)
    variance=HPDm**2/RelSigma2HalfWidthDiam2
    
    mu = 0
    
    sigma = math.sqrt(variance)
    pdfY=mlab.normpdf(x_s, mu, sigma)
    pdfE=mlab.normpdf(x_l, mu, sigma)
    
    
    plt.clf()
    plt.plot(x_l,pdfY*(cos(tita)/(1-x_l/R*sin(tita))))
    plt.plot(-x_l,pdfY*(cos(tita)/(1-x_l/R*sin(tita))))
    plt.plot(x_l,pdfY*(cos(tita)/(1-x_l/R*sin(tita)))-pdfY*(cos(tita)/(1+x_l/R*sin(tita))))
    
    plt.plot(x_l,pdfE)
    plt.show()
    
     #%%
    
    cols=21
    rows=21
    dx=1000
    dy=1000
    X1=-dx*cols/2
    X2=dx*cols/2
    Y1=-dy*rows/2
    Y2=dy*rows/2
    x, y = np.mgrid[X1+dx/2:X2:dx, Y1+dy/2:Y2:dy]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    
    
    rv = multivariate_normal(mu, Sigma)
    z=rv.pdf(pos)*dx*dy
            
    #%% UNIVARIATE VERSION
    R=1124 #distance from sensor to beam axis intersection with earth
    tita=55/180.0*np.pi
    
    #Local coordinates where power has to be found
    Xl = np.linspace(-100,100, 1001)
    Yl=0
    
    #Sensor local coordinates
    Xs=np.sin(tita)*R
    Ys=0
    Zs=np.cos(tita)*R
    
    l=-R/(R-np.sin(tita)*Xl)
    
    #earth tangent plane points projected to the beam normal plane
    Xp=Xs+l*(Xs-Xl)
    Yp=l*Yl
    Zp=Zs+l*Zs
    #distance from zero:
    Rp=np.sqrt(Xp*Xp+Yp*Yp+Zp*Zp)
    J=np.cos(tita)/np.sqrt(1+(Xl/R)**2+(Yl/R)**2+-2*Xl/R*np.sin(tita))
    
    
    mu=0
    sigma=2
    pdf1=mlab.normpdf(Xl, mu, sigma)
    pdf2=mlab.normpdf(Rp, mu, sigma)*J
    
    plt.clf()
    plt.plot(Xl,pdf1)
    plt.plot(Xl,pdf2)
    plt.plot(-Xl,pdf2)
    
    
    #%%plot 3d
    
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Make data
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = 10 * np.outer(np.cos(u), np.sin(v))
    y = 10 * np.outer(np.sin(u), np.sin(v))
    z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Plot the surface
    ax.plot_surface(x, y, z, color='b')
    
    plt.show()