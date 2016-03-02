# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 16:36:57 2016

@author: schiavon
"""

from glob import glob
import numpy as np
from scipy.misc import imread
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import leastsq,curve_fit
import matplotlib.pyplot as plt
from os.path import isfile

rootpath = '/home/schiavon/luxor/sagnac/'
datapath = rootpath + 'data/focus/pump/f_500_1000/'

pixelsize = 5.2e-6

files = np.sort(glob(datapath+'*.bmp'))
L = np.zeros(len(files))

wx = np.zeros(len(files))
wy = np.zeros(len(files))

def gauss2d(x,y,x0,y0,wx,wy):
    g = np.exp( - 2*(x-x0)**2/wx**2 - 2*(y-y0)**2/wy**2 )
    return g

datafile = datapath+'data.npz'
if not isfile(datafile):
    for i in range(len(files)):
        # get the filename
        filecurr = files[i].split('/')[-1]
        
        # get the distance [mm] from the file name
        L[i] = float(filecurr[1:4])/1000
        
        # apply gaussian filter and normalize
        fig = imread(datapath+filecurr)
        fig1 = fig[:,600:] # get subfigure [1:1024,600:1280] 
        fig2 = gaussian_filter(fig1,5)
        fig2 = fig2/np.max(fig2)
        
        # find coordinates of the max
        (my,mx) = np.unravel_index(fig2.argmax(),fig2.shape)
        
        
        p0 = [mx,my,5,5]
        
        x = np.arange(fig2.shape[1])
        y = np.arange(fig2.shape[0])
        
        xx,yy = np.meshgrid(x,y)
        
        p,success = leastsq(lambda p: (gauss2d(xx,yy,*p)-fig2).ravel(),p0)
        
        data_fitted = gauss2d(xx,yy,*p)
        
        wx[i] = p[2]*pixelsize
        wy[i] = p[3]*pixelsize
        
        # plot final figure with contours
        f,ax = plt.subplots(1,1)
        ax.hold(True)
        ax.imshow(fig2,origin='bottom',extent=(x.min(),x.max(),y.min(),y.max()))
        ax.contour(x,y,data_fitted,8,colors='w')
    
    np.savez(datafile,L=L,wx=wx,wy=wy)
else:
    tmp = np.load(datafile)
    L = tmp['L']
    wx = tmp['wx']
    wy = tmp['wy']

#%% fit waists
plt.close('all')

plt.figure()
plt.hold(True)
l = 404.5e-9 # wavelength
z = np.arange(L.min()-0.2,L.max()+.2,0.001)

def W(z,zW,w0,M):
    return w0*M*np.sqrt(1 + ((z-zW)*l/np.pi/w0**2)**2)
    
p0x = [L[np.argmin(wx)],np.min(wx),1]
px,pcovx = curve_fit(W,L[:-2],wx[:-2],p0=p0x)
plt.plot(L,wx,'+',z,W(z,*px))


p0y = [L[np.argmin(wy)],np.min(wy),1]
py,pcovy = curve_fit(W,L[:-2],wy[:-2],p0=p0y)
plt.plot(L,wy,'+',z,W(z,*py))

plt.text(0.5,0.0002,'Wx = '+"{:.4f}".format(p0x[1]*1e6)+' um\nWy = '+"{:.4f}".format(p0y[1]*1e6)+' um')

print('Wx = '+"{:.4f}".format(p0x[1]*1e6)+' um\nWy = '+"{:.4f}".format(p0y[1]*1e6)+' um')
print('Mx =',px[2])
print('My =',py[2])