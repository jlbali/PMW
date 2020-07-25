import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

def main(vmin = 50, vmax=275, cmap='gray'):
    Img = []
    for i in range(0,100):
        linea = []
        for j in np.arange(vmin, vmax+2,.01):
            linea.append(j)
        Img.append(linea)
    plt.imshow(Img,vmin=vmin,vmax=vmax,cmap = plt.get_cmap(cmap),interpolation='None')
    plt.colorbar(orientation="horizontal")
    #plt.show()
    plt.savefig("tempImage.jpg")
    # Extraemos la barra de color.
    img = Image.open("tempImage.jpg")
    img2 = img.crop((50,450, 750,550))
    img2.save("colorbar.jpg")
#%%
def partial_colorbar(vmin=0, vminmin = -1, vmax=1, vmaxmax=2, ticks=[]):
    Img = []
    n_a=int(np.round((vmaxmax-vmax)/((vmax-vmin)/10)))
    n_u=int(np.round((vmin-vminmin)/((vmax-vmin)/10)))
    n=10
    nT=n_a+n+n_u
    D=np.concatenate((np.repeat(1,3*n_u),-np.repeat(np.linspace(1,0,n),3)+1,np.repeat(0,3*n_a))).reshape([nT,3])
    D[:,0]=np.linspace(0,1,nT)
    cdict = {'red':   D,
         'green': D,
         'blue':  D}
         
    from matplotlib.colors import LinearSegmentedColormap
    my_gray = LinearSegmentedColormap('My_Gray', cdict)
    
    for i in range(0,100):
        linea = []
        for j in np.arange(vminmin, vmaxmax,(vmaxmax-vminmin)/100):
            linea.append(j)
        Img.append(linea)
    plt.clf()
    plt.imshow(Img,vmin=vminmin,vmax=vmaxmax,cmap = my_gray,interpolation='None')
    if ticks==[]:
        ticks=[vminmin,vmin,(vmin+vmax)/2,vmax,vmaxmax]
    plt.colorbar(orientation="horizontal",ticks=ticks)
    #plt.show()
    plt.savefig("tempImage.jpg")
    # Extraemos la barra de color.
    img = Image.open("tempImage.jpg")
    img2 = img.crop((50,450, 750,550))
    img2.save("colorbarIF.jpg")
#%%


if __name__ == "__main__":
    partial_colorbar(vmin=0.0,vmax=1.0,vminmin=-.1,vmaxmax=1.4)
#%%
#IP colorbar
vmin=-0.05
vmax=0.4
vminmin=-.2
vmaxmax=0.8
ticks=[-.2,0,.2,.4,0.6,0.8]
partial_colorbar(vmin=vmin,vmax=vmax,vminmin=vminmin,vmaxmax=vmaxmax,ticks=ticks)
#%%
#IF colorbar
vmin=-0.03
vmax=0.23
vminmin=-.03
vmaxmax=0.23
ticks=[-.03,0,.05,.1,0.15,0.20,0.23]
partial_colorbar(vmin=vmin,vmax=vmax,vminmin=vminmin,vmaxmax=vmaxmax,ticks=ticks)
#%%
#Tb colorbar
vmin=150
vmax=300
vminmin=150
vmaxmax=300
ticks=[150, 175,200,225,250,275, 300]
partial_colorbar(vmin=vmin,vmax=vmax,vminmin=vminmin,vmaxmax=vmaxmax,ticks=ticks)
#%%
#Tb colorbar for NSIDC
vmin=120
vmax=270
vminmin=120
vmaxmax=270
ticks=[120, 150,180,210,240, 270]
partial_colorbar(vmin=vmin,vmax=vmax,vminmin=vminmin,vmaxmax=vmaxmax,ticks=ticks)
