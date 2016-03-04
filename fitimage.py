# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 14:58:10 2016

@author: schiavon
"""

from myfit import fit2dgauss
from scipy.misc import imread

from tkinter.filedialog import askopenfilename

pixelsize = 5.2e-6

filename = askopenfilename()

fig = imread(filename)

p = fit2dgauss(fig)

Wx = p[2]*pixelsize
Wy = p[3]*pixelsize

print('Wx =',"{:.4f}".format(Wx*1e6))
print('Wx =',"{:.4f}".format(Wy*1e6))