# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 05:39:43 2016

@author: ajoshi
"""
import scipy.io
import scipy as sp
import os
import numpy as np
import nibabel as nib
from dfsio import readdfs, writedfs
from surfproc import view_patch, smooth_surf_function, view_patch_vtk
from fmri_methods_sipi import rot_sub_data, reorder_labels
from sklearn.utils.linear_assignment_ import linear_assignment

#from scipy.spatial import cKDTree
p_dir = '/home/ajoshi/data/HCP_data/data'
p_dir_ref='/home/ajoshi/data/HCP_data'
lst = os.listdir(p_dir)
r_factor = 3
ref_dir = os.path.join(p_dir_ref, 'reference')
nClusters=3
sub='116524'
sub2 = '100307'
ref=sub2
print(ref + '.reduce' + str(r_factor) + '.LR_mask.mat')
fn1 = ref + '.reduce' + str(r_factor) + '.LR_mask.mat'
fname1 = os.path.join(ref_dir, fn1)
msk = scipy.io.loadmat(fname1)  # h5py.File(fname1);
dfs_left = readdfs(os.path.join(p_dir_ref, 'reference', ref + '.aparc.a2009s.32k_fs.reduce3.left.dfs'))
dfs_left_sm = readdfs(os.path.join(p_dir_ref, 'reference', ref + '.aparc.a2009s.32k_fs.reduce3.very_smooth.left.dfs'))



#vrest=nib.load('/home/ajoshi/HCP5/110411/MNINonLinear/Results/rfMRI_REST2_LR/rfMRI_REST2_LR_Atlas_hp2000_clean.dtseries.nii')
data = scipy.io.loadmat(os.path.join(p_dir, sub, sub + '.rfMRI_REST1_RL.reduce3.ftdata.NLM_11N_hvar_25.mat'))

LR_flag = msk['LR_flag']
LR_flag = np.squeeze(LR_flag) > 0
#data = sp.squeeze(vrest.get_data()).T
data = data['ftdata_NLM']
vrest = data[LR_flag, :]
#vrest = data[LR_flag]
m = np.mean(vrest, 1)
vrest = vrest - m[:,None]
s = np.std(vrest, 1)+1e-16
vrest1 = vrest/s[:,None]

#vrest=nib.load('/home/ajoshi/HCP5/110411/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas_hp2000_clean.dtseries.nii')
data = scipy.io.loadmat(os.path.join(p_dir, sub2, sub2 + '.rfMRI_REST1_RL.reduce3.ftdata.NLM_11N_hvar_25.mat'))

LR_flag = msk['LR_flag']
LR_flag = np.squeeze(LR_flag) > 0
data = data['ftdata_NLM']
#data = sp.squeeze(vrest.get_data()).T
vrest = data[LR_flag,:]
m = np.mean(vrest, 1)
vrest = vrest - m[:,None]
s = np.std(vrest, 1)+1e-16
vrest2 = vrest/s[:,None]

vrest2=rot_sub_data(ref=vrest1,sub=vrest2)

rho_rot = sp.dot(vrest1,vrest2.T)/vrest1.shape[1]
dist_mat = sp.absolute(sp.arccos(rho_rot))

ind = sp.argmin(dist_mat,axis=0)
#ind = linear_assignment(dist_mat)

view_patch(dfs_left_sm,outfile='before_registered_surf.png',show=1)

dfs_left_sm.faces=ind[dfs_left_sm.faces]
view_patch(dfs_left_sm,outfile='registered_surf1.png',show=1)

#rho1=smooth_surf_function(dfs_left_sm,rho1)
#rho2=smooth_surf_function(dfs_left_sm,rho2)

#view_patch(dfs_left_sm,rho_orig,clim=[0,1])
#view_patch(dfs_left_sm,rho_rot,clim=[0,1],show=1)

