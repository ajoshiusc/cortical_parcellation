# ||AUM||
from dfsio import readdfs, writedfs
import scipy.io
import numpy as np
import csv
from dfsio import readdfs, writedfs
from mayavi import mlab
import h5py
import os

p_dir = 'E:\\HCP-fMRI-NLM';
lst = os.listdir(p_dir)
r_factor = 3;
ref_dir = os.path.join(p_dir, 'reference');


ref = '100307';
print(ref + '.reduce' + str(r_factor) + '.LR_mask.mat')
fn1 = ref + '.reduce' + str(r_factor) + '.LR_mask.mat'
fname1 = os.path.join(ref_dir, fn1)
msk = scipy.io.loadmat(fname1)  # h5py.File(fname1);


for sub in lst:


    data = scipy.io.loadmat(os.path.join(p_dir, sub, sub + '.rfMRI_REST1_RL.reduce3.ftdata.NLM_11N_hvar_25.mat'));

    dfs_left = readdfs(os.path.join(p_dir, 'reference', sub + '.aparc.a2009s.32k_fs.reduce3.left.dfs'));
    dfs_left_sm = readdfs(os.path.join(p_dir, 'reference', sub + '.aparc.a2009s.32k_fs.reduce3.very_smooth.left.dfs'));

    LR_flag = msk['LR_flag']
    LR_flag = np.squeeze(LR_flag) > 0
    data = data['ftdata_NLM']
    temp = data[LR_flag, :]

    msk_small_region = (dfs_left.labels == 46) | (dfs_left.labels == 28) | (dfs_left.labels == 29)  # % motor
    # temp = data.ftdata_NLM;%(msk.LR_flag, :);
    d = temp[msk_small_region, :];
    d_corr = temp[~msk_small_region, :];
