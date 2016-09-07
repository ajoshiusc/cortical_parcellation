# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 15:51:16 2016

@author: ajoshi
"""
import nibabel.freesurfer.io as fsio
from surfproc import view_patch, smooth_patch, patch_color_labels
from dfsio import writedfs, readdfs
from nibabel.gifti.giftiio import read as gread
import os
from scipy.spatial import cKDTree
import nibabel as nib
from lxml import etree
import numpy as np
p_dir_ref = '/home/ajoshi/data/HCP_data'
ref_dir = os.path.join(p_dir_ref, 'reference')
ref = '100307'


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


inputfile = '/home/ajoshi/data/Glasser_et_al_2016_HCP_MMP1.0_RVVG/HCP_Phase\
Two/Q1-Q6_RelatedParcellation210/MNINonLinear/fsaverage_LR32k/Q1-Q6_Related\
Parcellation210.L.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii'
outputfile = 'BCI_DNI_Glasser.inner.left.dfs'

labs = nib.load(inputfile)

cifti = etree.XML(labs.header.extensions[0].get_content())
idxs=np.array(cifti[0][2][0][0].text.split(' ')).astype(np.int)

labels = np.squeeze(labs.get_data())
g_surf = nib.load('/home/ajoshi/data/HCP_data/reference/100307/MNINonL\
inear/fsaverage_LR32k/100307.L.very_inflated.32k_fs_LR.surf.gii')
s.vertices = g_surf.darrays[0].data; s.faces = g_surf.darrays[1].data
s.labels=np.zeros(s.vertices.shape[0])
s.labels[idxs] = labels



'''h32k to full res FS'''
g_surf = nib.load('/home/ajoshi/data/HCP_data/reference/100307/MNINonLinear/N\
ative/100307.L.very_inflated.native.surf.gii')
h.vertices = g_surf.darrays[0].data
h.faces = g_surf.darrays[1].data
h = interpolate_labels(s, h)
h = patch_color_labels(h)

view_patch(h)

''' native FS ref to native FS BCI'''
g_surf = nib.load('/home/ajoshi/data/HCP_data/reference/100307/MNINon\
Linear/Native/100307.L.sphere.reg.native.surf.gii')
s.vertices = g_surf.darrays[0].data
s.faces = g_surf.darrays[1].data
s.labels = h.labels

''' map to bc sphere'''
bs.vertices, bs.faces = fsio.read_geometry('/home/ajoshi/data/BCI_DNI_Atlas/su\
rf/lh.sphere.reg')
bs = interpolate_labels(s, bs)
bci.vertices, bci.faces = fsio.read_geometry('/home/ajoshi/data/BCI_DNI_A\
tlas/surf/lh.white')
bci.labels = bs.labels
writedfs('BCI_orig_lh.dfs', bci)


bci.vertices, bci.faces = fsio.read_geometry('/home/ajoshi/data/BCI_DNI_A\
tlas/surf/lh.inflated')
bci = patch_color_labels(bci)
view_patch(bci)

writedfs('BCI_pial_lh.dfs.', bci)

bci.vertices, bci.faces = fsio.read_geometry('/home/ajoshi/data/BCI_DNI_Atla\
s/surf/lh.white')
writedfs('BCI_white_lh.dfs.', bci)


bci.vertices[:, 0] += 96*0.8
bci.vertices[:, 1] += 192*0.546875
bci.vertices[:, 2] += 192*0.546875
bci_bst = readdfs('/home/ajoshi/data/BCI-DNI_brain_atlas/BCI-DNI_brain.left.\
inner.cortex.dfs')
bci_bst = interpolate_labels(bci, bci_bst)
bci_bst_mid = readdfs('/home/ajoshi/data/BCI-DNI_brain_atlas/BCI-DNI_brain.left.\
mid.cortex.dfs')
bci_bst.vertices=bci_bst_mid.vertices

#bci_bst = smooth_patch(bci_bst, iterations=100, relaxation=.5)
bci_bst = patch_color_labels(bci_bst)
writedfs(outputfile, bci_bst)
view_patch(bci_bst)
