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
#import h5py
import os
from surfproc import view_patch
from fmri_methods_sipi import rot_sub_data, reorder_labels

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

s1=sp.load('labs_all_data1_rot_individual5.npz');    
l12=s1['lab_sub']
#l12=sp.reshape(l,(l.shape[0]*l.shape[1]),order='F')
print sp.sum(sp.absolute(l12[:,1]-l12[:,0]))
l12=reorder_labels(l12)

print sp.sum(sp.absolute(l12[:,1]-l12[:,0]))
perm1=sp.mod(17*sp.arange(max(l12.flatten())+1),max(l12.flatten())+1)

#
for ind in range(l12.shape[1]):
    lab1=sp.int32(l12[:,ind])
    view_patch(dfs_left_sm,perm1[lab1],elevation=90,colorbar=0,show=0,outfile=lst[ind]+'_individual_view1.png',colormap='Paired')
    view_patch(dfs_left_sm,perm1[lab1],elevation=-90,colorbar=0,show=0,outfile=lst[ind]+'_individual_view2.png',colormap='Paired')

#    lab1=l2[:,ind]
#    view_patch(dfs_left_sm,lab1,show=0,outfile=lst[ind]+'_individual_rot_data2.png')
#

