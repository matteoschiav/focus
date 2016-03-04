# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 14:34:33 2016

@author: schiavon
"""

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import leastsq
import matplotlib.pyplot as plt


def gauss2d(x,y,x0,y0,wx,wy,A,b):
    g = A*np.exp( - 2*(x-x0)**2/wx**2 - 2*(y-y0)**2/wy**2 ) + b
    return g

def fit2dgauss(fig1):
    # apply gaussian filter and normalize
    fig2 = gaussian_filter(fig1,5)
    fig2 = fig2/np.max(fig2)
    
    # find coordinates of the max
    (my,mx) = np.unravel_index(fig2.argmax(),fig2.shape)
    
    
    p0 = [mx,my,5,5,1,0]
    
    x = np.arange(fig2.shape[1])
    y = np.arange(fig2.shape[0])
    
    xx,yy = np.meshgrid(x,y)
    
    p,success = leastsq(lambda p: (gauss2d(xx,yy,*p)-fig2).ravel(),p0)
    
    data_fitted = gauss2d(xx,yy,*p)
    
#    wx = p[2]*pixelsize
#    wy = p[3]*pixelsize
    
    # plot final figure with contours
    indX = np.arange(max(mx-np.floor(p[2])*2,0),min(mx+np.floor(p[2])*2,fig2.shape[1]),dtype=np.int)
    indY = np.arange(max(my-np.floor(p[3])*2,0),min(my+np.floor(p[3])*2,fig2.shape[0]),dtype=np.int)
    f,ax = plt.subplots(1,1)
    ax.hold(True)        
    ax.imshow(fig2[:,indX][indY],origin='bottom',extent=(indX.min(),indX.max(),indY.min(),indY.max()))
    ax.contour(x[indX],y[indY],data_fitted[:,indX][indY],3,colors='w',levels=[0.1,0.5,0.9])

    return p