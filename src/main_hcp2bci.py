# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 15:51:16 2016

@author: ajoshi
"""
import scipy as sp
import nibabel.freesurfer.io as fsio
from surfproc import view_patch
from dfsio import writedfs, readdfs
from nibabel.gifti.giftiio import read as gread
import os

p_dir_ref='/home/ajoshi/HCP_data'
ref_dir = os.path.join(p_dir_ref, 'reference')
ref = '100307'

def multidim_intersect(arr1, arr2):
    arr1_view = arr1.view([('',arr1.dtype)]*arr1.shape[1])
    arr2_view = arr2.view([('',arr2.dtype)]*arr2.shape[1])
    intersected = sp.intersect1d(arr1_view, arr2_view)
    return intersected.view(arr1.dtype).reshape(-1, arr1.shape[1])

inputfile='/home/ajoshi/Downloads/Yeo_JNeurophysiol11_FreeSurfer/fsaverage/label/lh.Yeo2011_17Networks_N1000.annot'
fsavesurf='/home/ajoshi/Downloads/Yeo_JNeurophysiol11_FreeSurfer/fsaverage/surf/lh.sphere.reg.avg'
yeomap,_,_=fsio.read_annot(inputfile)
vert,faces=fsaverage_surf=fsio.read_geometry(fsavesurf)
class s:
    pass



s.vertices=vert; s.faces=faces;s.labels=yeomap
view_patch(s,yeomap)

#g_surf = gread('/home/ajoshi/HCP_data/reference/100307/MNINonLinear/fsaverage_LR32k/100307.L.sphere.32k_fs_LR.surf.gii')
g_surf = gread('/home/ajoshi/HCP_data/reference/100307/MNINonLinear/Native/100307.L.sphere.reg.native.surf.gii')
vert = g_surf.darrays[0].data
face = g_surf.darrays[1].data

class hcp32k:
    pass

hcp32k.vertices=vert; hcp32k.faces=face

view_patch(hcp32k,vert[:,1])

from scipy.spatial import cKDTree

tree = cKDTree(s.vertices)

d, inds = tree.query(hcp32k.vertices, k=1, p=2)
hcp32k.labels = s.labels[inds]

view_patch(hcp32k,hcp32k.labels)

g_surf = gread('/home/ajoshi/HCP_data/reference/100307/MNINonLinear/Native/100307.L.very_inflated.native.surf.gii')
vert = g_surf.darrays[0].data
face = g_surf.darrays[1].data
hcp32k.vertices = vert; hcp32k.faces=face
view_patch(hcp32k,hcp32k.labels)

s.vertices = hcp32k.vertices
s.faces = hcp32k.faces
s.labels = hcp32k.labels

g_surf = gread('/home/ajoshi/HCP_data/reference/100307/MNINonLinear/fsaverage_LR32k/100307.L.very_inflated.32k_fs_LR.surf.gii')
vert = g_surf.darrays[0].data
face = g_surf.darrays[1].data
hcp32k.vertices = vert; hcp32k.faces=face

a = s.vertices;
b = hcp32k.vertices

tree = cKDTree(a)

d, ind = tree.query(b)
hcp32k.labels=s.labels[ind]
view_patch(hcp32k,hcp32k.labels)

writedfs('lh.Yeo2011_17Networks_N1000.dfs',hcp32k)

dfs = readdfs(os.path.join(p_dir_ref, 'reference', ref + '.aparc.a2009s.32k_fs.reduce3.very_smooth.left.dfs'))
a = hcp32k.vertices
b = dfs.vertices
tree = cKDTree(a)
d, ind = tree.query(b)
dfs.labels=hcp32k.labels[ind]
view_patch(dfs,dfs.labels)

writedfs('lh.Yeo2011_17Networks_N1000_reduce3.dfs',dfs)

