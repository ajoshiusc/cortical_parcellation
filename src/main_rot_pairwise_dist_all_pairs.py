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

ref = '196750'
print(ref + '.reduce' + str(r_factor) + '.LR_mask.mat')
fn1 = ref + '.reduce' + str(r_factor) + '.LR_mask.mat'
fname1 = os.path.join(ref_dir, fn1)
msk = scipy.io.loadmat(fname1)  # h5py.File(fname1);
dfs_right = readdfs(os.path.join(p_dir_ref, 'reference', ref + '.aparc.\
a2009s.32k_fs.reduce3.right.dfs'))
dfs_right_sm = readdfs(os.path.join(p_dir_ref, 'reference', ref + '.aparc.\
a2009s.32k_fs.reduce3.very_smooth.right.dfs'))
count1 = 0
rho_rho = []
rho_all = []
#lst=lst[:2]
labs_all = sp.zeros((len(dfs_right.labels), len(lst)))

for sub in lst:
    data = scipy.io.loadmat(os.path.join(p_dir, sub, sub + '.rfMRI_REST1_LR.\
reduce3.ftdata.NLM_11N_hvar_25.mat'))
    LR_flag = msk['LR_flag']
    LR_flag = np.squeeze(LR_flag) == 0
    data = data['ftdata_NLM']
    temp = data[LR_flag, :]
    m = np.mean(temp, 1)
    temp = temp - m[:, None]
    s = np.std(temp, 1)+1e-16
    temp = temp/s[:, None]
    d = temp
    if count1 == 0:
        sub_data = sp.zeros((d.shape[0], d.shape[1], len(lst)))

    sub_data[:, :, count1] = d
    count1 += 1
    print count1,

nSub = sub_data.shape[2]
rperm = sp.random.permutation(dfs_right_sm.vertices.shape[0])
#rperm=range(dfs_right_sm.vertices.shape[0])
dist_all_orig = sp.zeros(len(dfs_right_sm.vertices))
dist_all_rot = dist_all_orig.copy()
#sub_data[:,:,1]=sub_data[rperm,:,1]
sub_data_orig = sub_data.copy()

for ind in range(1, nSub):
    dist_all_orig += sp.mean((sub_data_orig[:, :, 0] - sub_data_orig[:, :, ind])**2.0,
                             axis=(1))
    sub_data[:, :, ind],_ = rot_sub_data(ref=sub_data[:, :, 0],
                                       sub=sub_data[:, :, ind])
    dist_all_rot += sp.mean((sub_data[:, :, 0] - sub_data[:, :, ind])**2.0, axis=(1))
    print ind,

dist_all_rot = dist_all_rot/nSub
dist_all_orig = dist_all_orig/nSub

var_all = sp.zeros((sub_data.shape[0], sub_data.shape[1]))

avg_sub_data = sp.mean(sub_data, axis=2)

# azimuth=-90,elevation=-180, roll=-90,
dfs_right_sm = patch_color_attrib(dfs_right_sm, (2-dist_all_orig)/2.0, clim=[0, 1])
view_patch_vtk(dfs_right_sm, azimuth=-90, elevation=-180,
               roll=-90, outfile='corr_orig_view1_1sub_right.png', show=0)
view_patch_vtk(dfs_right_sm, azimuth=90, elevation=180, roll=90,
               outfile='corr_orig_view2_1sub_right.png', show=0)

dfs_right_sm = patch_color_attrib(dfs_right_sm, (2-dist_all_rot)/2.0, clim=[0.75, 1])
ind = (dist_all_rot < 1e-6)
dfs_right_sm.vColor[ind, :] = 0.5
view_patch_vtk(dfs_right_sm, azimuth=-90, elevation=-180, roll=-90,
               outfile='corr_rot_view1_1sub_right.png', show=0)
view_patch_vtk(dfs_right_sm, azimuth=90, elevation=180, roll=90,
               outfile='corr_rot_view2_1sub_right.png', show=0)

sp.savez('rot_pairwise_dist.npz', dist_all_rot)