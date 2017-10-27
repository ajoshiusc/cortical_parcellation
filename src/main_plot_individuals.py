# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 23:31:10 2016

@author: ajoshi
"""
# ||AUM||

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
nClusters = 100

ref = '196750'
print(ref + '.reduce' + str(r_factor) + '.LR_mask.mat')
fn1 = ref + '.reduce' + str(r_factor) + '.LR_mask.mat'
fname1 = os.path.join(ref_dir, fn1)
msk = scipy.io.loadmat(fname1)  # h5py.File(fname1);
dfs_left = readdfs(os.path.join(p_dir_ref, 'reference', ref + '.aparc.a2009s.\
32k_fs.reduce3.left.dfs'))
dfs_left_sm = readdfs(os.path.join(p_dir_ref, 'reference', ref + '.aparc.\
a2009s.32k_fs.reduce3.very_smooth.left.dfs'))
count1 = 0
rho_rho = []
rho_all = []




s1 = sp.load('labs_concat_data_100_clusters.npz')

catlab = s1['labs_cat']

tlab = sp.zeros((catlab.shape[0], 2))

tlab[:, 0] = catlab


s1 = sp.load('labs_all_data1_rot_individual_nclusters100_sub5.npz')
l = s1['lab_sub']
l1 = sp.reshape(l[:, :40], (l.shape[0]*40), order='F')

l2 = sp.reshape(l[:, 40:80], (l.shape[0]*40), order='F')

l12 = sp.concatenate((l1[:, None], l2[:, None]), axis=1)

print sp.sum(sp.absolute(l12[:, 1]-l12[:, 0]))

print sp.sum(sp.absolute(l12[:, 1]-l12[:, 0]))

l1 = sp.reshape(l12[:, 0], (l.shape[0], 40), order='F')
l2 = sp.reshape(l12[:, 1], (l.shape[0], 40), order='F')

#
for ind in range(l12.shape[1]):



    lab1 = sp.int32(l12[:, ind])
    dfs_left_sm.labels = lab1
    s = silhouette_samples(cat_data[nVert*(ind):nVert*(ind+1), :],
                           dfs_left_sm.labels)
    s[s < 0] = 0
    s = s/sp.median(s)
    s[s > 1.0] = 1.0
    s1 = patch_color_labels(dfs_left_sm, freq=s)
#    view_patch(s1)
    view_patch(s1, elevation=90, colorbar=0, show=0, outfile=lst[ind]+'_\
    individual_view1_nclusters30_sil_modulated.png', colormap='Paired')
    view_patch(s1, elevation=-90, colorbar=0, show=0, outfile=lst[ind]+'_\
    individual_view2_nclusters30_sil_modulated.png', colormap='Paired')
