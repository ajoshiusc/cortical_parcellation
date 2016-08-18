# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 15:51:16 2016

@author: ajoshi
"""
import nibabel.freesurfer.io as fsio
from surfproc import view_patch, smooth_patch
from dfsio import writedfs, readdfs
from nibabel.gifti.giftiio import read as gread
import os
from scipy.spatial import cKDTree

p_dir_ref='/home/ajoshi/HCP_data'
ref_dir = os.path.join(p_dir_ref, 'reference')
ref = '100307'

def interpolate_labels(fromsurf=[],tosurf=[]):
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

inputfile='rh.Yeo2011_17Networks_N1000_reduce3.dfs'
outputfile='reduce3_to_BCI_DNI.left.dfs.dfs'

''' reduce3 to h32k'''
r3 = readdfs('rh.Yeo2011_17Networks_N1000_reduce3.dfs')

'''h32k to full res FS'''
g_surf = gread('/home/ajoshi/data/HCP_data/reference/100307/MNINonLinear/Native/100307.R.very_inflated.native.surf.gii')
h.vertices = g_surf.darrays[0].data; h.faces = g_surf.darrays[1].data
h = interpolate_labels(r3,h)
#view_patch(h,h.labels)

''' native FS ref to native FS BCI'''
g_surf = gread('/home/ajoshi/data/HCP_data/reference/100307/MNINonLinear/Native/100307.R.sphere.reg.native.surf.gii')
s.vertices = g_surf.darrays[0].data;s.faces = g_surf.darrays[1].data
s.labels = h.labels

''' map to bc sphere'''
bs.vertices, bs.faces = fsio.read_geometry('/home/ajoshi/data/BCI_DNI_Atlas/surf/rh.sphere.reg')
bs = interpolate_labels(s, bs)
#view_patch(bs, bs.labels)
#view_patch(s,s.labels)
bci.vertices, bci.faces = fsio.read_geometry('/home/ajoshi/data/BCI_DNI_Atlas/surf/rh.white')
bci.labels=bs.labels
writedfs('BCI_orig_rh.dfs.',bci)


bci.vertices, bci.faces = fsio.read_geometry('/home/ajoshi/data/BCI_DNI_Atlas/surf/rh.inflated')
view_patch(bci,bci.labels)

writedfs('BCI_pial_rh.dfs.',bci)

bci.vertices, bci.faces = fsio.read_geometry('/home/ajoshi/data/BCI_DNI_Atlas/surf/rh.white')
writedfs('BCI_white_rh.dfs.',bci)


bci.vertices[:,0]+=96*0.8; bci.vertices[:,1]+=192*0.546875; bci.vertices[:,2]+=192*0.546875
bci_bst= readdfs('/home/ajoshi/data/BCI-DNI_brain_atlas/BCI-DNI_brain.right.inner.cortex.dfs')
bci_bst = interpolate_labels(bci,bci_bst)

bci_bst = smooth_patch(bci_bst, iterations=1000, relaxation=0.8)

writedfs(outputfile,bci_bst)
view_patch(bci_bst,bci_bst.labels)


