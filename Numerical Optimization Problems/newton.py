#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 21:01:54 2017

@author: Mike
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

def func(x):
    return 100*np.square(np.square(x[0])-x[1])+np.square(x[0]-1)
def dfunc(x):
    df1 = 400*x[0]*(np.square(x[0])-x[1])+2*(x[0]-1)
    df2 = -200*(np.square(x[0])-x[1])
    return np.array([df1, df2])
def invhess(x):
    df11 = 1200*np.square(x[0])-400*x[1]+2
    df12 = -400*x[0]
    df21 = -400*x[0]
    df22 = 200
    hess = np.array([[df11, df12], [df21, df22]])
    return inv(hess)
def newton(x, max_int):
    miter = 1
    step = .5
    vals = []
    objectfs = []
    while miter <= max_int:
        vals.append(x)
        objectfs.append(func(x))
        temp = x-step*(invhess(x).dot(dfunc(x)))
        if np.abs(func(temp)-func(x))>0.01:
            x = temp
        else:
            break
        print(x, func(x), miter)
        miter += 1
    return vals, objectfs, miter
start = [1.2, 1.0]
val, objectf, iters = newton(start, 100)
x = np.array([i[0] for i in val])
y = np.array([i[1] for i in val])
fig = plt.figure()
plt.scatter(x, y, label='newton method')

