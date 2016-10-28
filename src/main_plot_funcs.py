# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 19:39:47 2016

@author: ajoshi
"""
import scipy as sp
import matplotlib.pyplot as plt

rhoval = sp.arange(0, 1, 0.01)
val = sp.zeros((rhoval.shape[0], 3))

val[:, 0] = sp.arcsin(rhoval)
val[:, 1] = sp.exp((-2.0*(1-rhoval))/(.72 ** 2))
val[:, 2] = 2.0 - sp.sqrt(2.0 - 2.0*rhoval)

plt.plot(rhoval, val)
plt.legend(['$\sin^{-1}$', '$\exp$', '$L^2$'], loc='upper left')
plt.show()
plt.draw()
plt.savefig('plot_func.pdf')
