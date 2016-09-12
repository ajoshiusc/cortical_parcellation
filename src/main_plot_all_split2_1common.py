# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 23:31:10 2016

@author: ajoshi
"""
import scipy.io
import scipy as sp
from fmri_methods_sipi import reorder_labels
from dfsio import readdfs
import os
from surfproc import view_patch, patch_color_labels
from sklearn.metrics import silhouette_samples
p_dir = '/home/ajoshi/data/HCP_data/data'
p_dir_ref = '/home/ajoshi/data/HCP_data'
lst = os.listdir(p_dir)
r_factor = 3
ref_dir = os.path.join(p_dir_ref, 'reference')
nClusters = 30

ref = '100307'
print(ref + '.reduce' + str(r_factor) + '.LR_mask.mat')
fn1 = ref + '.reduce' + str(r_factor) + '.LR_mask.mat'
fname1 = os.path.join(ref_dir, fn1)
msk = scipy.io.loadmat(fname1)
dfs_left = readdfs(os.path.join(p_dir_ref, 'reference', ref + '.aparc.a2009s.\
32k_fs.reduce3.left.dfs'))
dfs_left_sm = readdfs(os.path.join(p_dir_ref, 'reference', ref + '.aparc.\
a2009s.32k_fs.reduce3.very_smooth.left.dfs'))
count1 = 0
rho_rho = []
rho_all = []

s1 = sp.load('labs_all_split2_data_1common_30clusters.npz')
l = s1['lab_sub1']
l1 = sp.reshape(l, (l.shape[0]*l.shape[1]), order='F')

l = s1['lab_sub2']
l2 = sp.reshape(l, (l.shape[0]*l.shape[1]), order='F')

l12 = sp.concatenate((l1[:, None], l2[:, None]), axis=1)

print sp.sum(sp.absolute(l12[:, 1] - l12[:, 0]))
l12 = reorder_labels(l12)

print sp.sum(sp.absolute(l12[:, 1] - l12[:, 0]))

l1 = sp.reshape(l12[:, 0], (l.shape[0], l.shape[1]), order='F')
l2 = sp.reshape(l12[:, 1], (l.shape[0], l.shape[1]), order='F')

perm1 = sp.mod(17*sp.arange(max(l1.flatten())+1), max(l1.flatten())+1)

# Plot labels
ind = 19
ind2 = 0
lab1 = l1[:, ind]
lab2 = l2[:, ind2]
nVert = dfs_left_sm.vertices.shape[0]

cat_data1 = s1['cat_data1']
dfs_left_sm.labels = lab1
s = silhouette_samples(cat_data1[nVert*(ind):nVert*(ind+1), :],
                       dfs_left_sm.labels)
s[s < 0] = 0
s = s/sp.median(s)
s[s > 1.0] = 1.0

dfs_left_sm = patch_color_labels(dfs_left_sm, freq=s)

view_patch(dfs_left_sm, colorbar=0, colormap='Paired', elevation=90, show=0,
           outfile=lst[ind]+'_view1_split1_30clusters_1common_modulated.png')
view_patch(dfs_left_sm, colorbar=0, colormap='Paired', elevation=-90, show=0,
           outfile=lst[ind]+'_view2_split1_30clusters_1common_modulated.png')

cat_data2 = s1['cat_data2']
dfs_left_sm.labels = lab2
s = silhouette_samples(cat_data2[nVert*(ind2):nVert*(ind2+1), :],
                       dfs_left_sm.labels)
s[s < 0] = 0
s = s/sp.median(s)
s[s > 1.0] = 1.0
dfs_left_sm = patch_color_labels(dfs_left_sm, freq=s)

view_patch(dfs_left_sm, colorbar=0, colormap='Paired',
           elevation=90, show=0, outfile=lst[ind2+l.shape[1]-1]+'_view1_split2\
_30clusters_1common_modulated.png')
view_patch(dfs_left_sm, colorbar=0, colormap='Paired',
           elevation=-90, show=0, outfile=lst[ind2+l.shape[1]-1]+'_view2_\
split2_30clusters_1common_modulated.png')
