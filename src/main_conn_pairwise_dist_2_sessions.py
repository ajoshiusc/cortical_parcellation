# ||AUM||
import scipy.io
import scipy as sp
import numpy as np
from fmri_methods_sipi import rot_sub_data
from surfproc import view_patch_vtk, patch_color_attrib
from dfsio import readdfs
import os

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
count1 = 0
rho_rho = []
rho_all = []
cc_msk = (dfs_left.labels > 0)

for sub in lst:
    data = scipy.io.loadmat(os.path.join(p_dir, sub, sub + '.rfMRI_REST1_LR.\
reduce3.ftdata.NLM_11N_hvar_25.mat'))
    LR_flag = msk['LR_flag']
    LR_flag = np.squeeze(LR_flag) != 0
    data = data['ftdata_NLM']
    temp = data[LR_flag, :]
    m = np.mean(temp, 1)
    temp = temp - m[:, None]
    s = np.std(temp, 1)+1e-16
    temp = temp/s[:, None]
    d1 = temp[cc_msk, :]
    data = scipy.io.loadmat(os.path.join(p_dir, sub, sub + '.rfMRI_REST2_LR.\
reduce3.ftdata.NLM_11N_hvar_25.mat'))
    LR_flag = msk['LR_flag']
    LR_flag = np.squeeze(LR_flag) != 0
    data = data['ftdata_NLM']
    temp = data[LR_flag, :]
    m = np.mean(temp, 1)
    temp = temp - m[:, None]
    s = np.std(temp, 1)+1e-16
    temp = temp/s[:, None]
    d2 = temp[cc_msk, :]

    if count1 == 0:
        sub_data1 = sp.zeros((d1.shape[0], d1.shape[1], len(lst)))
        sub_data2 = sp.zeros((d2.shape[0], d2.shape[1], len(lst)))

    sub_data1[:, :, count1] = d1
    sub_data2[:, :, count1] = d2

    count1 += 1
    print count1,

nSub = sub_data1.shape[2]
dist_all_conn = sp.zeros(len(dfs_left_sm.vertices))

for ind in range(nSub):
    sub_conn1 = sp.corrcoef(sub_data1[:, :, ind]+1e-16)
    sub_conn1 = sub_conn1 - sp.mean(sub_conn1, axis=1)[:, None]
    sub_conn1 = sub_conn1 / (np.std(sub_conn1, axis=1) + 1e-16)[:, None]
    sub_conn2 = sp.corrcoef(sub_data2[:, :, ind]+1e-16)
    sub_conn2 = sub_conn2 - sp.mean(sub_conn2, axis=1)[:, None]
    sub_conn2 = sub_conn2 / (np.std(sub_conn2, axis=1) + 1e-16)[:, None]

    dist_all_conn[cc_msk] += sp.mean((sub_conn1-sub_conn2)**2.0, axis=(1))
    print ind,

dist_all_conn = dist_all_conn/(nSub)

var_all = sp.zeros((sub_data1.shape[0], sub_data2.shape[1]))

avg_sub_data = sp.mean(sub_data1, axis=2)


dfs_left_sm = patch_color_attrib(dfs_left_sm, dist_all_conn, clim=[0, 1])
view_patch_vtk(dfs_left_sm, azimuth=-90, elevation=-180, roll=-90,
               outfile='dist_sess_conn_view1_1sub_left.png', show=0)
view_patch_vtk(dfs_left_sm, azimuth=90, elevation=180, roll=90,
               outfile='dist_sess_conn_view2_1sub_left.png', show=0)

sp.savez('conn_sessions_pairwise_dist.npz', dist_all_conn)
