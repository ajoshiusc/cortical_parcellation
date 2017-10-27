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
from fmri_methods_sipi import reorder_labels

p_dir = '/big_disk/ajoshi/HCP_data/data'
p_dir_ref = '/big_disk/ajoshi/HCP_data'
lst = os.listdir(p_dir)
r_factor = 3
ref_dir = os.path.join(p_dir_ref, 'reference')
nClusters = 100

ref = '196750'
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


s1 = sp.load('labs_concat_data_100_clusters.npz')

catlab = s1['labs_cat']

tlab = sp.zeros((catlab.shape[0], 2))

tlab[:, 0] = catlab

dfs_left_sm.labels=catlab
dfs_left_sm = patch_color_labels(dfs_left_sm, shuffle=False)
view_patch_vtk(dfs_left_sm, azimuth=90,elevation=180, roll=90, outfile='conc_joint1'+str(nClusters)+'_clusters_new2.png',show=0)

view_patch_vtk(dfs_left_sm, azimuth=-90,elevation=-180, roll=-90, outfile='conc_joint2'+str(nClusters)+'_clusters_new2.png',show=0)






s1 = sp.load('labs_all_data_bothsessions_' + str(nClusters) + '_clusters.npz')
l = s1['lab_sub']
l1 = sp.reshape(l[:, :40], (l.shape[0]*40), order='F')

l2 = sp.reshape(l[:, 40:80], (l.shape[0]*40), order='F')

l12 = sp.concatenate((l1[:, None], l2[:, None]), axis=1)

print sp.sum(sp.absolute(l12[:, 1]-l12[:, 0]))

print sp.sum(sp.absolute(l12[:, 1]-l12[:, 0]))

l1 = sp.reshape(l12[:, 0], (l.shape[0], 40), order='F')
l2 = sp.reshape(l12[:, 1], (l.shape[0], 40), order='F')

#perm1 = sp.mod(19*sp.arange(max(l1.flatten())+1), max(l1.flatten())+1)
nVert = l.shape[0]
#cat_data = s1['cat_data']


for ind in range(30):
    lab1 = l1[:, ind]
    lab2 = l2[:, ind]
    tlab[:, 1] = lab1
    lab22, ind1 = reorder_labels(tlab)
    lab1=ind1[sp.int16(lab1), 1]
    lab2=ind1[sp.int16(lab2), 1]

    dfs_left_sm.labels = lab1
    s = sp.ones(nVert)#silhouette_samples(cat_data[nVert*(ind):nVert*(ind+1), :],
         #                  dfs_left_sm.labels)
    s[s < 0] = 0
    s = s/sp.median(s)
    s[s > 1.0] = 1.0
    dfs_left_sm = patch_color_labels(dfs_left_sm, shuffle=False)
    view_patch_vtk(dfs_left_sm, azimuth=90,elevation=180, roll=90, outfile=lst[ind]+'_joint_both_session1_\
view1_'+str(nClusters)+'_clusters_new2.png',show=0)

    view_patch_vtk(dfs_left_sm, azimuth=-90,elevation=-180, roll=-90, outfile=lst[ind]+'_joint_both_session1_\
view2_'+str(nClusters)+'_clusters_new2.png',show=0)

    dfs_left_sm.labels = lab2 #perm1[lab2]
    s = sp.ones(nVert)#silhouette_samples(cat_data[nVert*(40+ind):nVert*(40+ind+1), :],
          #                 dfs_left_sm.labels)
    dfs_left_sm = patch_color_labels(dfs_left_sm, shuffle=False)

    view_patch_vtk(dfs_left_sm, azimuth=90,elevation=180, roll=90, outfile=lst[ind]+'_joint_both_session2_\
view1_'+str(nClusters)+'_clusters_new2.png',show=0)

    view_patch_vtk(dfs_left_sm, azimuth=-90,elevation=-180, roll=-90, outfile=lst[ind]+'_joint_both_session2_\
view2_'+str(nClusters)+'_clusters_new2.png',show=0)

