# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 15:28:12 2016

@author: ajoshi
"""

from mayavi import mlab
import numpy as np
#from scipy.special import sph_harm

# Create a sphere
r = 0.3
pi = np.pi
cos = np.cos
sin = np.sin
phi, theta = np.mgrid[0:pi:101j, 0:2 * pi:101j]

x = r * sin(phi) * cos(theta)
y = r * sin(phi) * sin(theta)
z = r * cos(phi)

mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 300))
mlab.clf()
# Represent spherical harmonics on the surface of the sphere
mlab.mesh(x, y, z, color=(0.5,0.5,0.5))


phi, theta = np.mgrid[0:pi:101j, 0:2 * pi:101j]

x = r * sin(phi) * cos(theta)
y = r * sin(phi) * sin(theta)
z = r * cos(phi)

mlab.points3d(x,y,z,scale_factor=0.01,color=(1,0,0))

mlab.show(stop=True)
