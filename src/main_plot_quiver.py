# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 15:28:12 2016

@author: ajoshi
"""

from mayavi import mlab
import numpy as np
#from scipy.special import sph_harm
import math

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

# Create a sphere
r = 1.0
pi = np.pi
cos = np.cos
sin = np.sin
phi, theta = np.mgrid[0:pi:100j, 0:2 * pi:100j]

x = r * sin(phi) * cos(theta)
y = r * sin(phi) * sin(theta)
z = r * cos(phi)

mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 300))
mlab.clf()
# Represent spherical harmonics on the surface of the sphere
mlab.mesh(x, y, z, colormap=(0.5,0.5,0.5),opacity=.5)

mlab.quiver3d(0,0,0,1,1,1,mode='arrow')

mlab.quiver3d(0.0,0.0,0.0,12.707,2.707,1.0,mode='arrow',color=(0,1.0,0))

mlab.show(stop=True)

