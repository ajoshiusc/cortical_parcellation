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
from surfproc import view_patch, view_patch_vtk, smooth_surf_function, face_v_conn, patch_color_attrib
from fmri_methods_sipi import rot_sub_data, reorder_labels

p_dir = '/big_disk/ajoshi/HCP_data'
p_dir_ref='/big_disk/ajoshi/HCP_data/'
lst = os.listdir(p_dir)
r_factor = 3
ref_dir = os.path.join(p_dir_ref, 'reference')
nClusters = 3

ref = '100307'
print(ref + '.reduce' + str(r_factor) + '.LR_mask.mat')
fn1 = ref + '.reduce' + str(r_factor) + '.LR_mask.mat'
fname1 = os.path.join(ref_dir, fn1)
msk = scipy.io.loadmat(fname1)  # h5py.File(fname1);
dfs_left = readdfs(os.path.join(p_dir_ref, 'reference', ref + '.aparc\
.a2009s.32k_fs.reduce3.left.dfs'))
dfs_left_sm = readdfs(os.path.join(p_dir_ref, 'reference', ref + '.aparc\
.a2009s.32k_fs.reduce3.very_smooth.left.dfs'))

surf1 = dfs_left_sm
X = surf1.vertices[:, 0]
Y = surf1.vertices[:, 1]
Z = surf1.vertices[:, 2]
NumTri = surf1.faces.shape[0]
#    NumVertx = X.shape[0]
vertx_1 = surf1.faces[:, 0]
vertx_2 = surf1.faces[:, 1]
vertx_3 = surf1.faces[:, 2]
V1 = np.column_stack((X[vertx_1], Y[vertx_1], Z[vertx_1]))
V2 = np.column_stack((X[vertx_2], Y[vertx_2], Z[vertx_2]))
V3 = np.column_stack((X[vertx_3], Y[vertx_3], Z[vertx_3]))
x1 = np.zeros((NumTri))
y1 = np.zeros((NumTri))
v2_v1temp = V2-V1
x2 = np.linalg.norm(v2_v1temp, axis=1)
y2 = np.zeros((NumTri))
x3 = np.einsum('ij,ij->i', (V3-V1),
               (v2_v1temp/np.column_stack((x2, x2, x2))))
mynorm = np.cross((v2_v1temp), V3-V1, axis=1)
yunit = np.cross(mynorm, v2_v1temp, axis=1)
y3 = np.einsum('ij,ij->i', yunit, (V3-V1))/np.linalg.norm(yunit, axis=1)
sqrt_DT = (np.abs((x1*y2 - y1*x2)+(x2*y3 - y2*x3)+(x3*y1 - y3*x1)))
Ar = 0.5*(np.abs((x1*y2 - y1*x2)+(x2*y3 - y2*x3)+(x3*y1 - y3*x1)))

TC = face_v_conn(surf1)
Wt = (1.0/3.0)*(TC)
# Wt = sp.sparse.spdiags(Wt*Ar, (0), NumTri, NumTri)
surf_weight = Wt*Ar
surf1.attributes = surf_weight
surf_weight = surf_weight[:, None]
# smooth_surf_function(dfs_left_sm, Wt*Ar*0.1, a1=0, a2=1)
surf1 = patch_color_attrib(surf1)

view_patch(surf1, show=1)

# sub = '110411'
# p_dir = '/home/ajoshi/data/HCP_data'
lst = os.listdir('/big_disk/ajoshi/HCP5')
rho1 = 0; rho1rot = 0; rho2 = 0; rho2rot = 0;
# lst = [lst[0]]


for sub in lst:
    vmotor = nib.load('/big_disk/ajoshi/with_andrew/For_Anand/tf\
MRI_MOTOR_RL/tfMRI_MOTOR_RL_Atlas.dtseries.nii')
    LR_flag = msk['LR_flag']
    LR_flag = np.squeeze(LR_flag) > 0
    data = sp.squeeze(vmotor.get_data()).T
    vrest = data[LR_flag]
    m = np.mean(vrest, 1)
    vrest = vrest - m[:, None]
    s = np.std(vrest, 1)+1e-116
    vmotor1 = vrest/s[:, None]

    vmotor = nib.load('/big_disk/ajoshi/with_andrew/For_Anand/tf\
MRI_MOTOR_LR/tfMRI_MOTOR_LR_Atlas.dtseries.nii')
    LR_flag = msk['LR_flag']
    LR_flag = np.squeeze(LR_flag) > 0
    data = sp.squeeze(vmotor.get_data()).T
    vrest = data[LR_flag]
    m = np.mean(vrest, 1)
    vrest = vrest - m[:, None]
    s = np.std(vrest, 1)+1e-116
    vmotor2 = vrest/s[:, None]    

    vrest = nib.load('/big_disk/ajoshi/HCP5/' + sub + '/MNINonLinear/Resu\
lts/rfMRI_REST2_LR/rfMRI_REST2_LR_Atlas_hp2000_clean.dtseries.nii')    
    LR_flag = msk['LR_flag']
    LR_flag = np.squeeze(LR_flag) > 0
    data = sp.squeeze(vrest.get_data()).T
    vrest = data[LR_flag]
    vrest = vrest[:, :vmotor1.shape[1]]    # make their length same
    m = np.mean(vrest, 1)
    vrest = vrest - m[:, None]
    s = sp.std(vrest, axis=1) +1e-116
    vrest1 = vrest/s[:, None]

    vrest = nib.load('/big_disk/ajoshi/HCP5/' + sub + '/MNINonLinear/Res\
ults/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas_hp2000_clean.dtseries.nii')
    LR_flag = msk['LR_flag']
    LR_flag = np.squeeze(LR_flag) > 0
    data = sp.squeeze(vrest.get_data()).T
    vrest = data[LR_flag]
    vrest = vrest[:, :vmotor1.shape[1]]    
    m = np.mean(vrest, 1)
    vrest = vrest - m[:, None]
    s = np.std(vrest, 1) + 1e-116
    vrest2 = vrest/s[:, None]


    rho1 += sp.sum(vrest1*vmotor1, axis=1)/vrest1.shape[1]
    rho2 += sp.sum(vrest2*vmotor2, axis=1)/vmotor1.shape[1]

    vmotor1 = rot_sub_data(ref=vrest1, sub=vmotor1, area_weight=sp.sqrt(surf_weight))
    vmotor2 = rot_sub_data(ref=vrest2, sub=vmotor2, area_weight=sp.sqrt(surf_weight))
      
    rho1rot += sp.sum(vrest1*vmotor1, axis=1)/vrest1.shape[1]
    rho2rot += sp.sum(vrest2*vmotor2, axis=1)/vmotor1.shape[1]


rho1 = smooth_surf_function(dfs_left_sm, rho1, a1=0, a2=1)
rho1rot = smooth_surf_function(dfs_left_sm, rho1rot, a1=0, a2=1)

view_patch(dfs_left_sm, rho1/len(lst), clim=[0, 1],
           outfile='rest1motor_before_rot.png', show=0)
view_patch(dfs_left_sm, rho1rot/len(lst), clim=[0, 1],
           outfile='rest1motor_after_rot.png', show=0)

rho2 = smooth_surf_function(dfs_left_sm, rho2, a1=0, a2=1)
rho2rot = smooth_surf_function(dfs_left_sm, rho2rot, a1=0, a2=1)

view_patch(dfs_left_sm, rho2/len(lst), clim=[0,1],
           outfile='rest2motor_before_rot.png', show=0)
view_patch(dfs_left_sm, rho2rot/len(lst),
           clim=[0,1], outfile='rest2motor_after_rot.png', show=0)
