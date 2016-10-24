# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 00:08:31 2016

@author: ajoshi
"""
import nibabel.freesurfer.io as fsio
from surfproc import view_patch, smooth_patch, patch_color_labels, view_patch_vtk
from dfsio import writedfs, readdfs
from nibabel.gifti.giftiio import read as gread
import os
from scipy.spatial import cKDTree
import nibabel as nib
import scipy.io
from lxml import etree
import numpy as np

def interpolate_labels(fromsurf=[], tosurf=[]):
    ''' interpolate labels from surface to to surface'''
    tree = cKDTree(fromsurf.vertices)
    d, inds = tree.query(tosurf.vertices, k=1, p=2)
    tosurf.labels = fromsurf.labels[inds]
    return tosurf


class h32k:
    pass


class h:
    pass


class s:
    pass


class bs:
    pass


class bci:
    pass


outputfile='gaurav_bci.dfs'

p_dir='/home/sgaurav/Documents/git_sandbox/cortical_parcellation/src/intensity_mode_map'
data1 = scipy.io.loadmat(os.path.join(p_dir, 'intensity_file_cingulate_184_nCluster=3_BCI.mat'))

labs = data1['labs_all']


''' reduce3 to h32k'''
r3 = readdfs('rh.Yeo2011_17Networks_N1000_reduce3.dfs')
r3.labels=np.squeeze(labs.T)

'''h32k to full res FS'''
g_surf = gread('/home/ajoshi/data/HCP_data/reference/100307/MNINonLinear/N\
ative/100307.R.very_inflated.native.surf.gii')
h.vertices = g_surf.darrays[0].data
h.faces = g_surf.darrays[1].data
h = interpolate_labels(r3, h)

''' native FS ref to native FS BCI'''
g_surf = gread('/home/ajoshi/data/HCP_data/reference/100307/MNINonLinear/Nativ\
e/100307.R.sphere.reg.native.surf.gii')
s.vertices = g_surf.darrays[0].data
s.faces = g_surf.darrays[1].data
s.labels = h.labels

''' map to bc sphere'''
bs.vertices, bs.faces = fsio.read_geometry('/home/ajoshi/data/BCI_DNI_Atlas/su\
rf/rh.sphere.reg')
bs = interpolate_labels(s, bs)
bci.vertices, bci.faces = fsio.read_geometry('/home/ajoshi/data/BCI_DNI_A\
tlas/surf/rh.white')
bci.labels = bs.labels
writedfs('BCI_orig_rh.dfs', bci)


bci.vertices, bci.faces = fsio.read_geometry('/home/ajoshi/data/BCI_DNI_A\
tlas/surf/rh.inflated')
view_patch(bci, bci.labels)

writedfs('BCI_pial_rh.dfs.', bci)

bci.vertices, bci.faces = fsio.read_geometry('/home/ajoshi/data/BCI_DNI_Atla\
s/surf/rh.white')
writedfs('BCI_white_rh.dfs.', bci)


bci.vertices[:, 0] += 96*0.8
bci.vertices[:, 1] += 192*0.546875
bci.vertices[:, 2] += 192*0.546875
bci_bst = readdfs('/home/ajoshi/data/BCI-DNI_brain_atlas/BCI-DNI_brain.right.\
inner.cortex.dfs')
bci_bst = interpolate_labels(bci, bci_bst)

#bci_bst = smooth_patch(bci_bst, iterations=1000, relaxation=0.8)

writedfs(outputfile, bci_bst)
view_patch(bci_bst, bci_bst.labels)
