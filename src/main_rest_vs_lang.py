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
from surfproc import view_patch, view_patch_vtk, smooth_surf_function
from fmri_methods_sipi import rot_sub_data, reorder_labels

p_dir = '/home/ajoshi/data/HCP_data'
p_dir_ref='/home/ajoshi/data/HCP_data'
lst = os.listdir(p_dir)
r_factor = 3
ref_dir = os.path.join(p_dir_ref, 'reference')
nClusters=3

ref = '100307'
print(ref + '.reduce' + str(r_factor) + '.LR_mask.mat')
fn1 = ref + '.reduce' + str(r_factor) + '.LR_mask.mat'
fname1 = os.path.join(ref_dir, fn1)
msk = scipy.io.loadmat(fname1)  # h5py.File(fname1);
dfs_left = readdfs(os.path.join(p_dir_ref, 'reference', ref + '.aparc.a2009s.32k_fs.reduce3.left.dfs'))
dfs_left_sm = readdfs(os.path.join(p_dir_ref, 'reference', ref + '.aparc.a2009s.32k_fs.reduce3.very_smooth.left.dfs'))


# sub = '110411'
# p_dir = '/home/ajoshi/data/HCP_data'
lst = os.listdir('/home/ajoshi/data/HCP5')
rho1=0;rho1lang=0;rho2=0;rho2lang=0;
for sub in lst:
    vrest=nib.load('/home/ajoshi/data/HCP5/' + sub + '/MNINonLinear/Results/rfMRI_REST2_LR/rfMRI_REST2_LR_Atlas_hp2000_clean.dtseries.nii')
    
    LR_flag = msk['LR_flag']
    LR_flag = np.squeeze(LR_flag) > 0
    data = sp.squeeze(vrest.get_data()).T
    vrest = data[LR_flag]
    m = np.mean(vrest, 1)
    vrest = vrest - m[:,None]
    s = np.std(vrest, 1)+1e-16
    vrest1 = vrest/s[:,None]
    
    vrest=nib.load('/home/ajoshi/data/HCP5/' + sub + '/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas_hp2000_clean.dtseries.nii')
    
    LR_flag = msk['LR_flag']
    LR_flag = np.squeeze(LR_flag) > 0
    data = sp.squeeze(vrest.get_data()).T
    vrest = data[LR_flag]
    m = np.mean(vrest, 1)
    vrest = vrest - m[:,None]
    s = np.std(vrest, 1)+1e-16
    vrest2 = vrest/s[:,None]
    
    vlang=nib.load('/home/ajoshi/data/HCP5/' + sub + '/MNINonLinear/Results/tfMRI_LANGUAGE_LR/tfMRI_LANGUAGE_LR_Atlas.dtseries.nii')
    
    LR_flag = msk['LR_flag']
    LR_flag = np.squeeze(LR_flag) > 0
    data = sp.squeeze(vlang.get_data()).T
    vrest = data[LR_flag]
    m = np.mean(vrest, 1)
    vrest = vrest - m[:,None]
    s = np.std(vrest, 1)+1e-16
    vlang1 = vrest/s[:,None]
    
    vlang=nib.load('/home/ajoshi/data/HCP5/' + sub + '/MNINonLinear/Results/tfMRI_LANGUAGE_RL/tfMRI_LANGUAGE_RL_Atlas.dtseries.nii')
    
    LR_flag = msk['LR_flag']
    LR_flag = np.squeeze(LR_flag) > 0
    data = sp.squeeze(vlang.get_data()).T
    vrest = data[LR_flag]
    m = np.mean(vrest, 1)
    vrest = vrest - m[:,None]
    s = np.std(vrest, 1)+1e-16
    vlang2 = vrest/s[:,None]
    
    
    rho1+=sp.sum(vrest1*vrest2,axis=1)/vrest1.shape[1]
    rho1lang+=sp.sum(vlang1*vlang2,axis=1)/vlang1.shape[1]
    
    vrest2=rot_sub_data(ref=vrest1,sub=vrest2)
    vlang2=rot_sub_data(ref=vlang1,sub=vlang2)
    
    rho2+=sp.sum(vrest1*vrest2,axis=1)/vrest1.shape[1]
    rho2lang+=sp.sum(vlang1*vlang2,axis=1)/vlang1.shape[1]
    
    
rho1=smooth_surf_function(dfs_left_sm,rho1)
rho2=smooth_surf_function(dfs_left_sm,rho2)

view_patch(dfs_left_sm,rho1/len(lst),clim=[0,1],outfile='rest_before_rot.png',show=0)
view_patch(dfs_left_sm,rho2/len(lst),clim=[0,1],outfile='rest_after_rot.png',show=0)

rho1lang=smooth_surf_function(dfs_left_sm,rho1lang)
rho2lang=smooth_surf_function(dfs_left_sm,rho2lang)

view_patch(dfs_left_sm,rho1lang/len(lst),clim=[.2,.6],outfile='lang_lang_before_rot.png',show=0)
view_patch(dfs_left_sm,rho2lang/len(lst),clim=[.2,.6],outfile='lang_lang_after_rot.png',show=0)
