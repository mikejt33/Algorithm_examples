#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 16:37:56 2017

@author: Mike
"""

import numpy as np
import matplotlib.pyplot as plt
def first(x):
    return (np.cos(1/x))
def second(p):
    return (np.cos(1/p)+(np.sin(1/p)/p)-((2*np.sin(1/p)*p+np.cos(1/p))/(2*p**2)))
x = np.linspace(-np.pi, np.pi, 256, endpoint=True)
a = first(x)  
b = second(x) 
plt.ylim(-5.0,5.0)
plt.xlim(-1, 1)
plt.plot(x, a, label='cos(1/x)')
plt.plot(x, b, label='2nd-order Taylor Expansion')
plt.legend()
plt.show()