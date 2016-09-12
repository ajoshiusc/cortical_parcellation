# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 23:31:10 2016

@author: ajoshi
"""
from scipy.stats import itemfreq
import scipy.io
import scipy as sp
from dfsio import readdfs
import os
from surfproc import view_patch, patch_color_labels, view_patch_vtk
from sklearn.metrics import silhouette_samples
p_dir = '/home/ajoshi/data/HCP_data/data'
p_dir_ref = '/home/ajoshi/data/HCP_data'
lst = os.listdir(p_dir)
r_factor = 3
ref_dir = os.path.join(p_dir_ref, 'reference')
nClusters = 17

ref = '100307'
print(ref + '.reduce' + str(r_factor) + '.LR_mask.mat')
fn1 = ref + '.reduce' + str(r_factor) + '.LR_mask.mat'
fname1 = os.path.join(ref_dir, fn1)
msk = scipy.io.loadmat(fname1)  # h5py.File(fname1);
dfs_left = readdfs(os.path.join(p_dir_ref, 'reference', ref + '.aparc.a2009s.3\
2k_fs.reduce3.left.dfs'))
dfs_left_sm = readdfs(os.path.join(p_dir_ref, 'reference', ref + '.aparc.a2009\
s.32k_fs.reduce3.very_smooth.left.dfs'))
count1 = 0
rho_rho = []
rho_all = []

s1 = sp.load('labs_all_data_bothsessions_17_clusters.npz')
l = s1['lab_sub']
l1 = sp.reshape(l[:, :40], (l.shape[0]*l.shape[1]/2.0), order='F')

l2 = sp.reshape(l[:, 40:80], (l.shape[0]*l.shape[1]/2.0), order='F')

l12 = sp.concatenate((l1[:, None], l2[:, None]), axis=1)

print sp.sum(sp.absolute(l12[:, 1]-l12[:, 0]))

print sp.sum(sp.absolute(l12[:, 1]-l12[:, 0]))

l1 = sp.reshape(l12[:, 0], (l.shape[0], l.shape[1]/2.0), order='F')
l2 = sp.reshape(l12[:, 1], (l.shape[0], l.shape[1]/2.0), order='F')

perm1 = sp.mod(19*sp.arange(max(l1.flatten())+1), max(l1.flatten())+1)
nVert = l.shape[0]
#cat_data = s1['cat_data']

for ind in range(5):
    lab1 = l1[:, ind]
    counts1 = itemfreq(lab1)
    lab2 = l2[:, ind]
    dfs_left_sm.labels = perm1[lab1]
    s = sp.ones(nVert)#silhouette_samples(cat_data[nVert*(ind):nVert*(ind+1), :],
         #                  dfs_left_sm.labels)
    s[s < 0] = 0
    s = s/sp.median(s)
    s[s > 1.0] = 1.0
    dfs_left_sm = patch_color_labels(dfs_left_sm, freq=s)
    view_patch_vtk(dfs_left_sm, azimuth=90,elevation=180, roll=90, outfile=lst[ind]+'_joint_both_session1_\
view1_17_clusters.png')

    view_patch_vtk(dfs_left_sm, azimuth=-90,elevation=-180, roll=-90, outfile=lst[ind]+'_joint_both_session1_\
view2_17_clusters.png')

    dfs_left_sm.labels = perm1[lab2]
    s = sp.ones(nVert)#silhouette_samples(cat_data[nVert*(40+ind):nVert*(40+ind+1), :],
          #                 dfs_left_sm.labels)
    s[s < 0] = 0
    s = s/sp.median(s)
    s[s > 1.0] = 1.0
    dfs_left_sm = patch_color_labels(dfs_left_sm, freq=s)

    view_patch_vtk(dfs_left_sm, azimuth=90,elevation=180, roll=90, outfile=lst[ind]+'_joint_both_session2_\
view1_17_clusters.png')

    view_patch_vtk(dfs_left_sm, azimuth=-90,elevation=-180, roll=-90, outfile=lst[ind]+'_joint_both_session2_\
view2_17_clusters.png')
