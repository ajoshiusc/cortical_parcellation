# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 15:51:16 2016

@author: ajoshi
"""
import nibabel.freesurfer.io as fsio
from surfproc import view_patch, smooth_patch
from dfsio import writedfs, readdfs
import nibabel as nib
import os
import scipy as sp
import numpy as np
from scipy.spatial import cKDTree
from lxml import etree
p_dir_ref='/home/ajoshi/HCP_data'
ref_dir = os.path.join(p_dir_ref, 'reference')
ref = '100307'

def interpolate_labels(fromsurf=[],tosurf=[], skipzero=0):
    ''' interpolate labels from surface to to surface'''
    tree = cKDTree(fromsurf.vertices)
    d, inds = tree.query(tosurf.vertices, k=1, p=2)
    if skipzero==0:
        tosurf.labels = fromsurf.labels[inds]
    else:
        indz = (tosurf.labels == 0)
        tosurf.labels = fromsurf.labels[inds]
        tosurf.labels[indz] = 0
        
    return tosurf
    
class s:
    pass

class r:
    pass

''' HCP32k data'''
labs = nib.load('/home/ajoshi/data/Glasser_et_al_2016_HCP_MMP1.0_RVVG/HCP_PhaseTwo/Q1-Q6_RelatedParcellation210/MNINonLinear/fsaverage_LR32k/Q1-Q6_RelatedParcellation210.R.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii')

cifti = etree.XML(labs.header.extensions[0].get_content())
idxs=np.array(cifti[0][2][0][0].text.split(' ')).astype(np.int)

labels = sp.squeeze(labs.get_data())
g_surf = nib.load('/home/ajoshi/data/Glasser_et_al_2016_HCP_MMP1.0_RVVG/HCP_PhaseTwo/Q1-Q6_RelatedParcellation210/MNINonLinear/fsaverage_LR32k/Q1-Q6_RelatedParcellation210.R.inflated_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.surf.gii')
s.vertices = g_surf.darrays[0].data; s.faces = g_surf.darrays[1].data
s.labels=sp.zeros(s.vertices.shape[0])
s.labels[idxs] = labels
view_patch(s, s.labels,colormap='Paired')


''' ref BCI Sphere to FS very inflated '''
g_surf = nib.load('/home/ajoshi/data/HCP_data/reference/100307/MNINonLinear/fsaverage_LR32k/100307.R.very_inflated.32k_fs_LR.surf.gii')
r.vertices = g_surf.darrays[0].data; r.faces = g_surf.darrays[1].data
r.labels=s.labels
''' FS very inflated to reduce3 '''
dfs = readdfs('/home/ajoshi/data/HCP_data/reference/100307.aparc.a2009s.32k_fs.reduce3.very_smooth.right.dfs')
dfs = interpolate_labels(r,dfs, skipzero=1)

view_patch(dfs,dfs.labels, colormap='Paired')
writedfs('100307.reduce3.Glasser.right.dfs', dfs)

