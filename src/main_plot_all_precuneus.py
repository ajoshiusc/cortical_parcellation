# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 23:31:10 2016

@author: ajoshi
"""
# ||AUM||
from scipy.stats import itemfreq
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
from sklearn.metrics import silhouette_score
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

s1=sp.load('labs_all_data_bothsessions_8_clusters_precuneus.npz');    
l=s1['lab_sub']
msk_small_region=s1['msk_small_region']
l1=sp.reshape(l[:,:40],(l.shape[0]*l.shape[1]/2.0),order='F')

l2=sp.reshape(l[:,40:80],(l.shape[0]*l.shape[1]/2.0),order='F')

l12=sp.concatenate((l1[:,None],l2[:,None]),axis=1)

print sp.sum(sp.absolute(l12[:,1]-l12[:,0]))
#l12=reorder_labels(l12)

print sp.sum(sp.absolute(l12[:,1]-l12[:,0]))

l1=sp.reshape(l12[:,0],(l.shape[0],l.shape[1]/2.0),order='F')
l2=sp.reshape(l12[:,1],(l.shape[0],l.shape[1]/2.0),order='F')

perm1=sp.mod(17*sp.arange(max(l1.flatten())+1),max(l1.flatten())+1)
#
for ind in range(5): #range(l1.shape[1]):
    lab1=sp.zeros(dfs_left_sm.vertices.shape[0],dtype=sp.int16)
    lab2=lab1.copy()
    lab1[msk_small_region]=l1[:,ind]
    counts1=itemfreq(lab1)    
    lab2[msk_small_region]=l2[:,ind]
    counts2=itemfreq(lab2)        
    view_patch(dfs_left_sm,perm1[lab1],colorbar=0,show=0,elevation=-90,colormap='Paired',outfile=lst[ind]+'_joint_both_session1_view1_8_clusters_precuneus.png')
    view_patch(dfs_left_sm,perm1[lab1],colorbar=0,show=0,elevation=90,colormap='Paired',outfile=lst[ind]+'_joint_both_session1_view2_8_clusters_precuneus.png')
    view_patch(dfs_left_sm,perm1[lab2],colorbar=0,show=0,elevation=-90,colormap='Paired',outfile=lst[ind]+'_joint_both_session2_view1_8_clusters_precuneus.png')
    view_patch(dfs_left_sm,perm1[lab2],colorbar=0,show=0,elevation=90,colormap='Paired',outfile=lst[ind]+'_joint_both_session2_view2_8_clusters_precuneus.png')

#    lab1=l2[:,ind]
#    view_patch(dfs_left_sm,lab1,show=0,outfile=lst[ind]+'_individual_rot_data2.png')
#

