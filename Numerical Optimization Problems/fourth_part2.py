#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:29:13 2017

@author: Mike
""" 

import numpy as np
import matplotlib.pyplot as plt
def first(x):
    return (np.cos(x))
def second(p):
    return (np.cos(p)-np.sin(p)*p-0.5*np.cos(p)*p**2+(1/6)*np.sin(p)*p**3)
x = np.linspace(-np.pi, np.pi, 256, endpoint=True)
a = first(x)
b = second(x) 
plt.plot(x, a, label='cos(x)')
plt.plot(x, b, label='3rd-order Taylor Expansion')
plt.axvline(x=1, color='black', label='a found at x=1')
plt.legend()
plt.show()