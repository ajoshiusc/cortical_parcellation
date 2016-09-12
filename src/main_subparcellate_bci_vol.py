# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 00:08:31 2016

@author: ajoshi
"""
from fmri_methods_sipi import interpolate_labels
from dfsio import readdfs
import time
import scipy as sp
import nibabel as nib


left_mid = readdfs('/home/ajoshi/data/BCI-DNI_brain_atlas/BCI-DNI_brain.left.\
mid.cortex_refined_labs.dfs')
left_mid1 = readdfs('/home/ajoshi/data/BCI-DNI_brain_atlas/BCI-DNI_brain.left.\
mid.cortex.dfs')
left_mid.vertices = left_mid1.vertices

right_mid = readdfs('/home/ajoshi/data/BCI-DNI_brain_atlas/BCI-DNI_brain.right.\
mid.cortex_refined_labs.dfs')
right_mid1 = readdfs('/home/ajoshi/data/BCI-DNI_brain_atlas/BCI-DNI_brain.right.\
mid.cortex.dfs')
right_mid.vertices = right_mid1.vertices

vol_lab = nib.load('/home/ajoshi/data/BCI-DNI_brain_atlas/\
BCI-DNI_brain.label.nii.gz')
vol_img = vol_lab.get_data()
xres = vol_lab.header['pixdim'][1]
yres = vol_lab.header['pixdim'][2]
zres = vol_lab.header['pixdim'][3]

X, Y, Z = sp.meshgrid(sp.arange(vol_lab.shape[0]), sp.arange(vol_lab.shape[1]),
                      sp.arange(vol_lab.shape[2]), indexing='ij')

X = X*xres
Y = Y*yres
Z = Z*zres
vol_img = sp.mod(vol_img, 1000)
ind = (vol_img >= 120) & (vol_img < 600)
Xc = X[ind]
Yc = Y[ind]
Zc = Z[ind]


class t:
    pass


class f:
    pass


t.vertices = sp.concatenate((Xc[:, None], Yc[:, None], Zc[:, None]), axis=1)
f.vertices = sp.concatenate((left_mid.vertices, right_mid.vertices))
f.labels = sp.concatenate((left_mid.labels, right_mid.labels))

tic = time.time()
t = interpolate_labels(f, t)
toc = time.time()
print 'Time Elapsed = %f sec.' % (toc - tic)

vol_img[ind] = t.labels

new_img = nib.Nifti1Image(vol_img, vol_lab.affine)
nib.save(new_img, '/home/ajoshi/data/BCI-DNI_brain_atlas/\
BCI-DNI_brain.refined.label.nii.gz')
