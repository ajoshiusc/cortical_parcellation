# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 15:51:16 2016

@author: ajoshi
"""
import nibabel.freesurfer.io as fsio
from surfproc import view_patch_vtk, patch_color_labels
from dfsio import writedfs, readdfs
from nibabel.gifti.giftiio import read as gread
import os
from scipy.spatial import cKDTree
import nilearn.image as ni
from nibabel import load as gread
import scipy as sp


def interpolate_labels(fromsurf=[], tosurf=[], skipzero=0):
    ''' interpolate labels from surface to to surface'''
    tree = cKDTree(fromsurf.vertices)
    d, inds = tree.query(tosurf.vertices, k=1, p=2)
    if skipzero == 0:
        tosurf.labels = fromsurf.labels[inds]
    else:
        indz = (tosurf.labels == 0)
        tosurf.labels = fromsurf.labels[inds]
        tosurf.labels[indz] = 0
    return tosurf


class bci:
    pass


class g32k:
    pass


''' Right Hemisphere '''

g32ktmp = gread('/big_disk/ajoshi/data/standard_mesh_atlases/resample\
_fsaverage/fs_LR-deformed_to-fsaverage.R.sphere.32k_fs_LR.surf.gii')
g32k.vertices = g32ktmp.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0].data
g32k.faces = g32ktmp.get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')[0].data
g32k.vColor = sp.ones(g32k.vertices.shape)

bci.vertices, bci.faces = fsio.read_geometry('/big_disk/ajoshi/data/BCI_\
DNI_Atlas/surf/rh.sphere.reg')
bci.labels = fsio.read_annot('/big_disk/ajoshi/data/BCI_DNI_Atlas/label/rh\
.BA.thresh.annot')[0]

g32k = interpolate_labels(fromsurf=bci, tosurf=g32k)
g32ktmp = gread('/big_disk/ajoshi/HCP_data/32k_ConteAtlas_v2/Conte69.R\
.very_inflated.32k_fs_LR.surf.gii')
g32k.vertices = g32ktmp.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0].data
g32k = patch_color_labels(g32k)
view_patch_vtk(g32k)

writedfs('Boradmann_32k_right.dfs', g32k)

''' Left Hemisphere '''

g32ktmp = gread('/big_disk/ajoshi/data/standard_mesh_atlases/resample\
_fsaverage/fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii')
g32k.vertices = g32ktmp.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0].data
g32k.faces = g32ktmp.get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')[0].data
g32k.vColor = sp.ones(g32k.vertices.shape)

bci.vertices, bci.faces = fsio.read_geometry('/big_disk/ajoshi/data/BCI_\
DNI_Atlas/surf/lh.sphere.reg')
bci.labels = fsio.read_annot('/big_disk/ajoshi/data/BCI_DNI_Atlas/label/lh\
.BA.thresh.annot')[0]

g32k = interpolate_labels(fromsurf=bci, tosurf=g32k)
g32ktmp = gread('/big_disk/ajoshi/HCP_data/32k_ConteAtlas_v2/Conte69.L\
.very_inflated.32k_fs_LR.surf.gii')
g32k.vertices = g32ktmp.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0].data
g32k = patch_color_labels(g32k)
view_patch_vtk(g32k)

writedfs('Boradmann_32k_left.dfs', g32k)
