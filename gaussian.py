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

# load configuration from yaml file
import yaml

with open('gaussian.yml') as ymlfile:
    cfg = yaml.safe_load(ymlfile)
    
cfg = cfg[cfg['use_config']]
datapath = cfg['datapath']
l = cfg['wavelength']
M_analysis = cfg['M_analysis']
max_fit = cfg['fit_interval']['max']
min_fit = cfg['fit_interval']['min']
pos_text = (cfg['pos_text']['x'],cfg['pos_text']['y'])
AOI = cfg['AOI']

plt.close('all')

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
        fig1 = fig[AOI[0]:AOI[1],AOI[2]:AOI[3]] # get subfigure [1:1024,600:1280] 
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
z = np.arange(L.min()-0.1,L.max()+.1,0.001)

if M_analysis:
    def W(z,zW,w0,M):
        return w0*M*np.sqrt(1 + ((z-zW)*l/np.pi/w0**2)**2)
        
    p0x = [L[np.argmin(wx)],np.min(wx),1]
    p0y = [L[np.argmin(wy)],np.min(wy),1]
else:
    def W(z,zW,w0):
        return w0*np.sqrt(1 + ((z-zW)*l/np.pi/w0**2)**2)
        
    p0x = [L[np.argmin(wx)],np.min(wx)]
    p0y = [L[np.argmin(wy)],np.min(wy)]
    


px,pcovx = curve_fit(W,L[min_fit:max_fit],wx[min_fit:max_fit],p0=p0x)
perrx = np.sqrt(np.diag(pcovx))
plt.plot(L,wx,'+r',z,W(z,*px),'r')

resx = W(L,*px)-wx


py,pcovy = curve_fit(W,L[min_fit:max_fit],wy[min_fit:max_fit],p0=p0y)
perry = np.sqrt(np.diag(pcovy))
plt.plot(L,wy,'xb',z,W(z,*py),'b')

resy = W(L,*py)-wy

plt.text(pos_text[0],pos_text[1],'Wx = '+"{:.4f}".format(px[1]*1e6)+' um\nWy = '+"{:.4f}".format(py[1]*1e6)+' um')

print('Wx = ( '+"{:.4f}".format(px[1]*1e6)+' +- '+"{:.4f}".format(perrx[1]*1e6)+' ) um')
print('Wy = ( '+"{:.4f}".format(py[1]*1e6)+' +- '+"{:.4f}".format(perry[1]*1e6)+' ) um')
print('zWx = ( '+"{:.4f}".format(px[0]*1e3)+' +- '+"{:.4f}".format(perrx[0]*1e3)+' ) mm')
print('zWy = ( '+"{:.4f}".format(py[0]*1e3)+' +- '+"{:.4f}".format(perry[0]*1e3)+' ) mm')

if M_analysis:
    print('Mx =',px[2],'+-',perrx[2])
    print('My =',py[2],'+-',perry[2])

plt.figure()
plt.plot(L,resx,'.r',L,resy,'.b')
plt.xlim(L.min()-0.05,L.max()+0.05)

