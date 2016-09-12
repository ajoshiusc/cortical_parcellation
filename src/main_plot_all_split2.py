# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 23:31:10 2016

@author: ajoshi
"""
# ||AUM||

import scipy.io
import scipy as sp
import numpy as np
from fmri_methods_sipi import rot_sub_data, reorder_labels
from dfsio import readdfs, writedfs
#import h5py
import os
from surfproc import view_patch, view_patch_vtk, get_cmap
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.metrics import silhouette_samples
p_dir = '/home/ajoshi/HCP_data/data'
p_dir_ref='/home/ajoshi/HCP_data'
lst = os.listdir(p_dir)
r_factor = 3
ref_dir = os.path.join(p_dir_ref, 'reference')
nClusters=30

ref = '100307'
print(ref + '.reduce' + str(r_factor) + '.LR_mask.mat')
fn1 = ref + '.reduce' + str(r_factor) + '.LR_mask.mat'
fname1 = os.path.join(ref_dir, fn1)
msk = scipy.io.loadmat(fname1)  # h5py.File(fname1);
dfs_left = readdfs(os.path.join(p_dir_ref, 'reference', ref + '.aparc.a2009s.32k_fs.reduce3.left.dfs'))
dfs_left_sm = readdfs(os.path.join(p_dir_ref, 'reference', ref + '.aparc.a2009s.32k_fs.reduce3.very_smooth.left.dfs'))
count1 = 0
rho_rho=[];rho_all=[]

s1=sp.load('labs_all_split2_data_60clusters.npz');    
l=s1['lab_sub1']
l1=sp.reshape(l,(l.shape[0]*l.shape[1]),order='F')

#s2=sp.load('labs_all_split2_data_60clusters.npz');    
l=s1['lab_sub2']
l2=sp.reshape(l,(l.shape[0]*l.shape[1]),order='F')

l12=sp.concatenate((l1[:,None],l2[:,None]),axis=1)

print sp.sum(sp.absolute(l12[:,1]-l12[:,0]))
l12=reorder_labels(l12)

print sp.sum(sp.absolute(l12[:,1]-l12[:,0]))

l1=sp.reshape(l12[:,0],(l.shape[0],l.shape[1]),order='F')
l2=sp.reshape(l12[:,1],(l.shape[0],l.shape[1]),order='F')

perm1=sp.mod(17*sp.arange(max(l1.flatten())+1),max(l1.flatten())+1)

# Plot labels
for ind in range(4):#range(l.shape[1]):
    lab1=l1[:,ind]; lab2=l2[:,ind]
    view_patch(dfs_left_sm,perm1[lab1],colorbar=0,colormap='Paired',elevation=90,show=0,outfile=lst[ind]+'_view1_split1_60clusters.png')
    view_patch(dfs_left_sm,perm1[lab1],colorbar=0,colormap='Paired',elevation=-90,show=0,outfile=lst[ind]+'_view2_split1_60clusters.png')
    view_patch(dfs_left_sm,perm1[lab2],colorbar=0,colormap='Paired',elevation=90,show=0,outfile=lst[ind+l.shape[1]]+'_view1_split2_60clusters.png')
    view_patch(dfs_left_sm,perm1[lab2],colorbar=0,colormap='Paired',elevation=-90,show=0,outfile=lst[ind+l.shape[1]]+'_view2_split2_60clusters.png')

#    lab1=l2[:,ind]
#    view_patch(dfs_left_sm,lab1,show=0,outfile=lst[ind]+'_individual_rot_data2.png')
#
# Plot Silhoutte modulated 
cat_data2=s1['cat_data2']
lab_sub2=s1['lab_sub2']

for ind in range(2):#range(l.shape[1]):
    s=silhouette_samples(cat_data2[10832*(ind):10832*(ind+1),:],lab_sub2[:,ind])
    

