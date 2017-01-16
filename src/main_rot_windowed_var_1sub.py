# ||AUM||
import scipy.io
import scipy as sp
import numpy as np
from fmri_methods_sipi import rot_sub_data
# from surfproc import view_patch_vtk, patch_color_attrib
from dfsio import readdfs
import os
import nibabel as nib
# import matplotlib.pyplot as plt
import itertools
from random import randint

p_dir = '/big_disk/ajoshi/HCP5'
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
count1 = 0
rho_rho = []
rho_all = []
# lst=lst[:1]
labs_all = sp.zeros((len(dfs_left.labels), len(lst)))

sub = lst[0]
# data = scipy.io.loadmat(os.path.join(p_dir, sub, sub + '.rfMRI_REST1_LR.\
# reduce3.ftdata.NLM_11N_hvar_25.mat'))
vrest = nib.load('/big_disk/ajoshi/HCP5/' + sub + '/MNINonLinear/Resu\
lts/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas_hp2000_clean.dtseries.nii')
data = sp.squeeze(vrest.get_data()).T

LR_flag = msk['LR_flag']
LR_flag = np.squeeze(LR_flag) != 0
# data = data['ftdata_NLM']


# data2 = scipy.io.loadmat(os.path.join(p_dir, sub, sub + '.rfMRI_REST2_LR.\
# reduce3.ftdata.NLM_11N_hvar_25.mat'))
vrest = nib.load('/big_disk/ajoshi/HCP5/' + sub + '/MNINonLinear/Resu\
lts/rfMRI_REST2_LR/rfMRI_REST2_LR_Atlas_hp2000_clean.dtseries.nii')
data2 = sp.squeeze(vrest.get_data()).T

LR_flag = msk['LR_flag']
LR_flag = np.squeeze(LR_flag) != 0
# data2 = data2['ftdata_NLM']

# vlang=nib.load('/big_disk/ajoshi/HCP5/' + '100307'
# + '/MNINonLinear/Results/tfMRI_LANGUAGE_LR/tfMRI_LANGUAGE_LR_Atlas\
# .dtseries.nii')
# LR_flag = msk['LR_flag']
# LR_flag = np.squeeze(LR_flag) > 0
# data = sp.squeeze(vlang.get_data()).T

# Length of window
win_lengths = sp.arange(5, data.shape[1], 20)
nboot = 20
dist = sp.zeros((len(win_lengths), nboot))
corr_diff = dist.copy()
corr_mtx_diff = dist.copy()

nbootiter = sp.arange(nboot)
# for nsboot in :
# for WinT in win_lengths:
temp = data[LR_flag, :]
m = np.mean(temp, 1)
temp = temp - m[:, None]
s = sp.std(temp, axis=1)+1e-116
temp = temp/s[:, None]
d1 = temp
temp = data2[LR_flag, :]
m = np.mean(temp, 1)
temp = temp - m[:, None]
s = sp.std(temp, axis=1)+1e-116
temp = temp/s[:, None]
d2 = temp


full_corr = sp.sum(d1 * d2, axis=1)/d2.shape[1]

for nb, iWinL in itertools.product(nbootiter, sp.arange(len(win_lengths))):

    WinL = win_lengths[iWinL]
    startpt = randint(0, data.shape[1])
    t = sp.arange(startpt, startpt + WinL)
    t = sp.mod(t, data.shape[1])
    temp = data[LR_flag, :]
    temp = temp[:, t]
    m = np.mean(temp, 1)
    temp = temp - m[:, None]
    s = sp.std(temp, axis=1)+1e-116
    temp = temp/s[:, None]
    d1 = temp

    temp = data2[LR_flag, :]
    temp = temp[:, t]
    m = np.mean(temp, 1)
    temp = temp - m[:, None]
    s = sp.std(temp, axis=1)+1e-116
    temp = temp/s[:, None]
    d2 = temp

    #    if t == 0:
#        d_ref = d
#
    drot, _ = rot_sub_data(ref=d2, sub=d1)
    dist[iWinL, nb] = sp.linalg.norm((drot - d2)/sp.sqrt(WinL))
    print nb, WinL, dist[iWinL, nb],

    corr_diff[iWinL, nb] = sp.median(sp.sum(drot * d2, axis=1)/d2.shape[1])

    print corr_diff[iWinL, nb]

    corr_mtx = sp.sum(d1 * d2, axis=1)/d2.shape[1]

    corr_mtx_diff[iWinL, nb] = sp.linalg.norm(corr_mtx - full_corr)

sp.savez_compressed('Corr_dist_nsamples.npz', corr_mtx_diff=corr_mtx_diff,
                    corr_diff=corr_diff, dist=dist)
