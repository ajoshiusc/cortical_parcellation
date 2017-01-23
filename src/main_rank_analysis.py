# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# ||AUM||
import scipy.io
import scipy as sp
import numpy as np
from fmri_methods_sipi import rot_sub_data
from surfproc import view_patch_vtk, patch_color_attrib
from dfsio import readdfs
import os
import matplotlib.pyplot as plt


p_dir = '/big_disk/ajoshi/HCP_data/data'
p_dir_ref = '/big_disk/ajoshi/HCP_data'
lst = os.listdir(p_dir)

r_factor = 3
ref_dir = os.path.join(p_dir_ref, 'reference')
nClusters = 30

ref = '100307'
print(ref + '.reduce' + str(r_factor) + '.LR_mask.mat')
fn1 = ref + '.reduce' + str(r_factor) + '.LR_mask.mat'
fname1 = os.path.join(ref_dir, fn1)
msk = scipy.io.loadmat(fname1)  # h5py.File(fname1);
dfs_left = readdfs(os.path.join(p_dir_ref, 'reference', ref + '.aparc.\
a2009s.32k_fs.reduce3.left.dfs'))
dfs_left_sm = readdfs(os.path.join(p_dir_ref, 'reference', ref + '.aparc.\
a2009s.32k_fs.reduce3.very_smooth.left.dfs'))
# view_patch_vtk(dfs_left_sm)
rho_rho = []
rho_all = []
#lst=lst[:1]
labs_all = sp.zeros((len(dfs_left.labels), len(lst)))
sub = lst[0]
data = scipy.io.loadmat(os.path.join(p_dir, sub, sub + '.rfMRI_REST1_LR.\
reduce3.ftdata.NLM_11N_hvar_25.mat'))
LR_flag = msk['LR_flag']
LR_flag = np.squeeze(LR_flag) != 0
data = data['ftdata_NLM']
temp = data[LR_flag, :]
#temp[5000:6000, 500:700] = sp.randn(1000, 200) # temp[1000, :]
m = np.mean(temp, 1)
temp = temp - m[:, None]
s = np.std(temp, 1)+1e-16
temp = temp/s[:, None]
d1 = temp

data = scipy.io.loadmat(os.path.join(p_dir, sub, sub + '.rfMRI_REST1_RL.\
reduce3.ftdata.NLM_11N_hvar_25.mat'))
LR_flag = msk['LR_flag']
LR_flag = np.squeeze(LR_flag) != 0
data = data['ftdata_NLM']
temp = data[LR_flag, :]
#temp[5000:6000, 500:700] = sp.randn(1000, 200) # temp[1000, :]
m = np.mean(temp, 1)
temp = temp - m[:, None]
s = np.std(temp, 1)+1e-16
temp = temp/s[:, None]
d2 = temp

## 

ev = np.linalg.eigvals(sp.dot(d1.T,d1))
ev = -sp.sort(-ev)
plt.plot(ev[2:100])
plt.savefig('Eigvalue_spread.pdf')
