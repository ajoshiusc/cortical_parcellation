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
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

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

ind_subsample = sp.arange(start=0, stop=dfs_left.labels.shape[0],step=1)
ind_rois_orig = sp.in1d(dfs_left.labels,[46,3,4,28,29,68,69,70])
ind_rois = sp.full(ind_rois_orig.shape[0], False ,dtype=bool)
ind_rois = ind_rois_orig.copy()
ind_rois[ind_subsample] = True


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

surf1.attributes = ind_rois
surf1 = patch_color_attrib(surf1)
view_patch_vtk(surf1, show=1)

# sub = '110411'
# p_dir = '/home/ajoshi/data/HCP_data'
lst = os.listdir('/big_disk/ajoshi/HCP5')
rho1 = 0; rho1rot = 0; rho2 = 0; rho2rot = 0;
# lst = [lst[0]]
diffbefore = 0
diffafter = 0

sub = lst[0]

vrest = scipy.io.loadmat('/big_disk/ajoshi/with_andrew/100307/100307.\
tfMRI_MOTOR_LR.reduce3.ftdata.NLM_11N_hvar_5.mat')
LR_flag = msk['LR_flag']
LR_flag = np.squeeze(LR_flag) > 0
data = vrest['ftdata_NLM']
vrest = data[LR_flag]
vrest = vrest[ind_rois, ]
m = np.mean(vrest, 1)
vrest = vrest - m[:, None]
s = np.std(vrest, 1)+1e-116
vmotor1_1 = vrest/s[:, None]

vrest = scipy.io.loadmat('/big_disk/ajoshi/with_andrew/100307/100307.\
tfMRI_MOTOR_RL.reduce3.ftdata.NLM_11N_hvar_5.mat')
LR_flag = msk['LR_flag']
LR_flag = np.squeeze(LR_flag) > 0
data = vrest['ftdata_NLM']
vrest = data[LR_flag]
vrest = vrest[ind_rois, ]
m = np.mean(vrest, 1)
vrest = vrest - m[:, None]
s = np.std(vrest, 1)+1e-116
vmotor1_2 = vrest/s[:, None]

vrest = scipy.io.loadmat('/big_disk/ajoshi/with_andrew/100307/100307.\
tfMRI_LANGUAGE_LR.reduce3.ftdata.NLM_11N_hvar_5.mat')
LR_flag = msk['LR_flag']
LR_flag = np.squeeze(LR_flag) > 0
data = vrest['ftdata_NLM']
vrest = data[LR_flag]
vrest = vrest[ind_rois, ]
m = np.mean(vrest, 1)
vrest = vrest - m[:, None]
s = np.std(vrest, 1)+1e-116
vmotor1_3 = vrest/s[:, None]

vrest = scipy.io.loadmat('/big_disk/ajoshi/with_andrew/100307/100307.\
tfMRI_LANGUAGE_RL.reduce3.ftdata.NLM_11N_hvar_5.mat')
LR_flag = msk['LR_flag']
LR_flag = np.squeeze(LR_flag) > 0
data = vrest['ftdata_NLM']
vrest = data[LR_flag]
vrest = vrest[ind_rois, ]
m = np.mean(vrest, 1)
vrest = vrest - m[:, None]
s = np.std(vrest, 1)+1e-116
vmotor1_4 = vrest/s[:, None]



vmotor1 = sp.concatenate((vmotor1_1, vmotor1_2, vmotor1_3, vmotor1_4), axis=1)

#vmotor1 = vmotor1[ind_rois,]
#vrest = nib.load('/big_disk/ajoshi/HCP5/' + sub + '/MNINonLinear/Resu\
#lts/rfMRI_REST2_LR/rfMRI_REST2_LR_Atlas_hp2000_clean.dtseries.nii')    
vrest = scipy.io.loadmat('/big_disk/ajoshi/with_andrew/100307/100307.\
rfMRI_REST1_LR.reduce3.ftdata.NLM_11N_hvar_5.mat')
LR_flag = msk['LR_flag']
LR_flag = np.squeeze(LR_flag) > 0
data = vrest['ftdata_NLM']
#data = sp.squeeze(vrest.get_data()).T
vrest = data[LR_flag]
vrest = vrest[ind_rois, ]
vrest = vrest[:, :vmotor1.shape[1]]    # make their length same
m = np.mean(vrest, 1)
vrest = vrest - m[:, None]
s = sp.std(vrest, axis=1) + 1e-116
vrest1 = vrest/s[:, None]

rho1 = sp.sum(vrest1*vmotor1, axis=1)/vrest1.shape[1]
diffbefore = vrest1 - vmotor1

vmotor1, Rot = rot_sub_data(ref=vrest1, sub=vmotor1, area_weight=sp.sqrt(surf_weight[ind_rois]))
#vrest1 = gaussian_filter(vrest1,[0,2]) 
#vmotor1 = gaussian_filter(vmotor1,[0,2]) 
#vrest1=vrest1[:,78:95]
#vmotor1=vmotor1[:,78:95]
#vrest1=vrest1[:,57:74]
#vmotor1=vmotor1[:,57:74]

#vrest1=vrest1[:,140:157]
#vmotor1=vmotor1[:,140:157]

rho1rot = sp.sum(vrest1*vmotor1,
                 axis=1)/vrest1.shape[1]

diffafter = vrest1 - vmotor1

#diffbefore = gaussian_filter(diffbefore,[0,5]) 

plt.imshow(sp.absolute(diffbefore), aspect='auto', clim=(0, 2.0))
plt.colorbar()
plt.savefig('dist_motor_before.pdf', dpi=300)
plt.show()

diffafter = gaussian_filter(diffafter, [0, 50])

plt.imshow(sp.absolute(diffafter), aspect='auto', clim=(0, .05))
plt.colorbar()
plt.savefig('dist_motor_after.pdf', dpi=300)
plt.show()


rho_full = sp.zeros((surf1.attributes.shape[0]))
rho_full[ind_rois] = rho1
dfs_left_sm.attributes = rho_full
dfs_left_sm = patch_color_attrib(dfs_left_sm, clim=[0, 1])
view_patch_vtk(dfs_left_sm, azimuth=90, elevation=180, roll=90,
               outfile='rest1motor_before_rot.png', show=1)

   #dfs_left_sm.attributes = sp.absolute(diffafter[:,t])
#    dfs_left_sm=patch_color_attrib(dfs_left_sm,clim=[0,1])
#    view_patch_vtk(dfs_left_sm, azimuth=90, elevation=180, roll=90, outfile='rest1motrho_full=sp.zeros((surf1.attributes.shape[0]))
rho_full[ind_rois] = rho1rot
dfs_left_sm.attributes = rho_full
dfs_left_sm = patch_color_attrib(dfs_left_sm, clim=[0, 1])
view_patch_vtk(dfs_left_sm, azimuth=90, elevation=180, roll=90,
               outfile='rest_vs_hand_after_rot1.png', show=1)
view_patch_vtk(dfs_left_sm, azimuth=-90, elevation=180, roll=-90,
               outfile='rest_vs_hand_after_rot2.png', show=1)


#plt.plot(rho1)
#
#for t in sp.arange(15,32):
#    dfs_left_sm.attributes = sp.absolute(diffafter[:,t])
#    dfs_left_sm = patch_color_attrib(dfs_left_sm,clim=[0,.6])
#    view_patch_vtk(dfs_left_sm, azimuth=90, elevation=180, roll=90, show=1)
#    
#    
rho = vrest1*vmotor1
rho_s = gaussian_filter(rho,[0,10])


for ind in sp.arange(vmotor1.shape[1]):
    dfs_left_sm.attributes = rho_s[:,ind]
    fname1 = 'rest_vs_hand_after_rot_%d_d.png' % ind
    fname2 = 'rest_vs_hand_after_rot_%d_m.png' % ind
    dfs_left_sm = patch_color_attrib(dfs_left_sm, clim=[0, 1])
    view_patch_vtk(dfs_left_sm, azimuth=90, elevation=180, roll=90,
                   outfile=fname1, show=0)
#    view_patch_vtk(dfs_left_sm, azimuth=-90, elevation=180, roll=-90,
#                   outfile=fname2, show=1)
    print ind, 

