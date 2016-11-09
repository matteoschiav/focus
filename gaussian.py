# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 16:36:57 2016

@author: schiavon
"""

from glob import glob
import numpy as np
from scipy.misc import imread
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from os.path import isfile
import matplotlib

from myfit import fit2dgauss

# load configuration from yaml file
import yaml

with open('gaussian.yml') as ymlfile:
    cfg = yaml.safe_load(ymlfile)
    
cfgtype = cfg['use_config']
cfgspec = cfg[cfg['use_config']]
datapath = cfgspec['datapath']
l = float(cfgspec['wavelength'])
M_analysis = cfgspec['M_analysis']
max_fit = cfgspec['fit_interval']['max']
min_fit = cfgspec['fit_interval']['min']
pos_text = (cfgspec['pos_text']['x'],cfgspec['pos_text']['y'])
AOI = cfgspec['AOI']
force = cfg['force']

plt.close('all')

pixelsize = 5.2e-6

files = np.sort(glob(datapath+'/*.bmp'))
L = np.zeros(len(files))

wx = np.zeros(len(files))
wy = np.zeros(len(files))


datafile = datapath+'data.npz'
if not isfile(datafile) or force:
    for i in range(len(files)):
        # get the filename
        filecurr = files[i].split('/')[-1]
        
        # get the distance [mm] from the file name
        L[i] = float(filecurr[1:4])/1000
        
        fig = imread(datapath+filecurr)
        fig1 = fig[AOI[0]:AOI[1],AOI[2]:AOI[3]] # get subfigure [1:1024,600:1280] 

        p = fit2dgauss(fig1)
        
        wx[i] = p[2]*pixelsize
        wy[i] = p[3]*pixelsize
    
    np.savez(datafile,L=L,wx=wx,wy=wy)
else:
    tmp = np.load(datafile)
    L = tmp['L']
    wx = tmp['wx']
    wy = tmp['wy']

#%% fit waists
# plt.close('all')

#fm.FontManager(size=40)
matplotlib.rcParams.update({'font.size':22})

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.figure()
plt.hold(True)
z = np.arange(L[min_fit:max_fit].min()-0.02,L[min_fit:max_fit].max()+.02,0.001)

if M_analysis:
    def W(z,zW,w0,M):
        return w0*np.sqrt(1 + (M**2*(z-zW)*l/(np.pi*w0**2))**2)

    if not 'p0x' in cfg.keys():      
        p0x = [L[np.argmin(wx)],np.min(wx),1]
    else:
        p0x = [cfgspec['p0x'][0],cfgspec['p0x'][1],2]
    if not 'p0y' in cfg.keys():
        p0y = [L[np.argmin(wy)],np.min(wy),1]
    else:
        p0y = [cfgspec['p0y'][0],cfgspec['p0y'][1],2]
else:
    def W(z,zW,w0):
        return w0*np.sqrt(1 + ((z-zW)*l/np.pi/w0**2)**2)
        
    if not 'p0x' in cfg.keys():      
        p0x = [L[np.argmin(wx)],np.min(wx)]
    else:
        p0x = cfgspec['p0x']
    if not 'p0y' in cfg.keys():
        p0y = [L[np.argmin(wy)],np.min(wy)]
    else:
        p0y = cfgspec['p0y']
    

px,pcovx = curve_fit(W,L[min_fit:max_fit],wx[min_fit:max_fit],p0=p0x)
perrx = np.sqrt(np.diag(pcovx))
plt.plot(L[min_fit:max_fit],wx[min_fit:max_fit]*1e3,'or',markersize=8)
plt.plot(z,W(z,*px)*1e3,'r',linewidth=2)

resx = W(L,*px)-wx


py,pcovy = curve_fit(W,L[min_fit:max_fit],wy[min_fit:max_fit],p0=p0y)
perry = np.sqrt(np.diag(pcovy))
plt.plot(L[min_fit:max_fit],wy[min_fit:max_fit]*1e3,'db',markersize=8)
plt.plot(z,W(z,*py)*1e3,'b',linewidth=2)
plt.xlim(np.amin(z),np.amax(z))

plt.xlabel(r'$z$ $[m]$')
plt.ylabel(r'$W(z)$ $[mm]$')

resy = W(L,*py)-wy

plt.text(pos_text[0],pos_text[1],r'\centering $W_x = ('+"{:.0f}".format(px[1]*1e6)+'\pm'+ \
    '{:.0f}'.format(perrx[1]*1e6)+')\,\mu m$\n$W_y = ('+"{:.0f}".format(py[1]*1e6)+ \
    '\pm'+'{:.0f}'.format(perry[1]*1e6)+')\,\mu m$\n$M_x^2 = '+\
    '{:.2f}'.format(px[2])+'\pm'+'{:.2f}'.format(perrx[2])+'$\n$M_y^2 = '+\
    '{:.2f}'.format(py[2])+'\pm'+'{:.2f}'.format(perry[2])+'$')

print('Analysis of',cfgtype)
print('Wx = ( '+"{:.4f}".format(px[1]*1e6)+' +- '+"{:.4f}".format(perrx[1]*1e6)+' ) um')
print('Wy = ( '+"{:.4f}".format(py[1]*1e6)+' +- '+"{:.4f}".format(perry[1]*1e6)+' ) um')
print('zWx = ( '+"{:.4f}".format(px[0]*1e3)+' +- '+"{:.4f}".format(perrx[0]*1e3)+' ) mm')
print('zWy = ( '+"{:.4f}".format(py[0]*1e3)+' +- '+"{:.4f}".format(perry[0]*1e3)+' ) mm')
print('theta_x = '+"{:.4f}".format(l/np.pi/px[1]*1e3)+' mrad')
print('theta_y = '+"{:.4f}".format(l/np.pi/py[1]*1e3)+' mrad')


if M_analysis:
    print('Mx =',px[2],'+-',perrx[2])
    print('My =',py[2],'+-',perry[2])

#plt.figure()
#plt.plot(L,resx,'.r',L,resy,'.b')
#plt.xlim(L.min()-0.05,L.max()+0.05)

plt.tight_layout()
