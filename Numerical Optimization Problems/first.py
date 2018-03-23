#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:17:43 2017

@author: Mike
"""

from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
def first(x, y):
    return (100 *(y-x**2)**2 + (1-x)**2)
x = arange(-6.0,5.0,0.1)
y = arange(-6.0,5.0,0.1)
X,Y = meshgrid(x, y) 
Z = first(X, Y)
'''
im = imshow(Z,cmap=cm.RdBu) # drawing the function
# adding the Contour lines with labels
cset = contour(Z,arange(-1,1.5,0.2),linewidths=2,cmap=cm.Set2)
plt([1],[1])
clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
colorbar(im) # adding the colobar on the right
# latex fashion title
title('$z=(1-x^2+y^3) e^{-(x^2+y^2)/2}$')
show()
'''
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, 
                      cmap=cm.RdBu,linewidth=0, antialiased=False)
ax.plot([1],[1],'go')
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=.5, aspect=5)
plt.show()