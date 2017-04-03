# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 15:51:16 2016

@author: ajoshi
"""
import nibabel.freesurfer.io as fsio
from surfproc import view_patch
from dfsio import writedfs, readdfs
from nibabel.gifti.giftiio import read as gread
import os
from scipy.spatial import cKDTree

p_dir_ref = '/big_disk/ajoshi/HCP_data'
ref_dir = os.path.join(p_dir_ref, 'reference')
ref = '100307'


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


class s:
    pass


class bci:
    pass


''' BCI to FS processed BCI '''
bci_bsti = readdfs('/big_disk/ajoshi/coding_ground/svreg-matlab/BCI-DNI\
_brain_atlas_refined/BCI-DNI_brain.right.mid.cortex.dfs')
bci_bst = readdfs('/big_disk/ajoshi/coding_ground/svreg-matlab/BCI-DNI\
_brain_atlas_refined/BCI-DNI_brain.right.inner.cortex.dfs')
bci_bst.labels = bci_bsti.labels
bci_bst.vertices[:, 0] -= 96*0.8
bci_bst.vertices[:, 1] -= 192*0.546875
bci_bst.vertices[:, 2] -= 192*0.546875
bci.vertices, bci.faces = fsio.read_geometry('/big_disk/ajoshi/data/BCI_\
DNI_Atlas/surf/rh.white')
bci = interpolate_labels(bci_bst, bci)

''' FS_BCI to FS BCI Sphere'''
bci.vertices, bci.faces = fsio.read_geometry('/big_disk/ajoshi/data/BCI_\
DNI_Atlas/surf/rh.sphere.reg')

''' FS BCI Sphere to ref FS Sphere'''
g_surf = gread('/big_disk/ajoshi/HCP_data/reference/100307/MNINon\
Linear/Native/100307.R.sphere.reg.native.surf.gii')
s.vertices = g_surf.darrays[0].data
s.faces = g_surf.darrays[1].data
s = interpolate_labels(bci, s)

''' ref BCI Sphere to FS very inflated '''
g_surf = gread('/big_disk/ajoshi/HCP_data/reference/100307/MNINon\
Linear/Native/100307.R.very_inflated.native.surf.gii')
bci.vertices = g_surf.darrays[0].data
bci.faces = g_surf.darrays[1].data
bci.labels = s.labels

''' FS very inflated to reduce3 '''
dfs = readdfs('/big_disk/ajoshi/HCP_data/reference/100307.aparc.a\
2009s.32k_fs.reduce3.very_smooth.right.dfs')
dfs = interpolate_labels(bci, dfs, skipzero=1)

# view_patch(dfs, dfs.labels, colormap='prism')
writedfs('BCI_refined_reduce3.very_smooth.right.dfs', dfs)
