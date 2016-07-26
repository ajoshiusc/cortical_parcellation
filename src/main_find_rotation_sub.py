# ||AUM||
import scipy.io
import scipy as sp
import numpy as np
from dfsio import readdfs, writedfs
from mayavi import mlab
from fmri_methods_sipi import rot_sub_data
#import h5py
import os
from surfproc import view_patch, view_patch_vtk, get_cmap, smooth_patch
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import DictionaryLearning
from scipy.stats import trim_mean
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.metrics import silhouette_score
p_dir = 'E:\\HCP-fMRI-NLM'
p_dir_ref='E:\\'
lst = os.listdir(p_dir)
r_factor = 3
ref_dir = os.path.join(p_dir_ref, 'reference')
nClusters=2

ref = '100307'
sub = lst[15]
print sub, ref
print(ref + '.reduce' + str(r_factor) + '.LR_mask.mat')
fn1 = ref + '.reduce' + str(r_factor) + '.LR_mask.mat'
fname1 = os.path.join(ref_dir, fn1)
msk = scipy.io.loadmat(fname1)  # h5py.File(fname1);
dfs_left = readdfs(os.path.join(p_dir_ref, 'reference', ref + '.aparc.a2009s.32k_fs.reduce3.left.dfs'))
dfs_left_sm = readdfs(os.path.join(p_dir_ref, 'reference', ref + '.aparc.a2009s.32k_fs.reduce3.very_smooth.left.dfs'))
count1 = 0
roilist=[30, 72, 9, 47] #pc
ref=lst[11]
datasub = scipy.io.loadmat(os.path.join(p_dir, sub, sub + '.rfMRI_REST1_RL.reduce3.ftdata.NLM_11N_hvar_25.mat'))
dataref = scipy.io.loadmat(os.path.join(p_dir, ref, ref + '.rfMRI_REST1_RL.reduce3.ftdata.NLM_11N_hvar_25.mat'))

LR_flag = msk['LR_flag']
LR_flag = np.squeeze(LR_flag) > 0
data = dataref['ftdata_NLM']
temp = data[LR_flag, :]
m = np.mean(temp, 1)
temp = temp - m[:,None]
s = np.std(temp, 1)+1e-16
temp = temp/s[:,None]

data = datasub['ftdata_NLM']
tempsub = data[LR_flag, :]
m = np.mean(tempsub, 1)
tempsub = tempsub - m[:,None]
s = np.std(tempsub, 1)+1e-16
tempsub = tempsub/s[:,None]

msk_small_region = np.in1d(dfs_left.labels,roilist)
#    msk_small_region = (dfs_left.labels == 30) | (dfs_left.labels == 72) | (dfs_left.labels == 9) |  (dfs_left.labels == 47)  # % motor
d = temp[msk_small_region, :]

ref_mean_pc = sp.mean(d,axis=0)
ref_mean_pc=ref_mean_pc-sp.mean(ref_mean_pc)
ref_mean_pc=ref_mean_pc/(sp.std(ref_mean_pc))

rho = np.dot(ref_mean_pc,temp.T)
rho[~np.isfinite(rho)] = 0

simil_mtx=sp.pi/2.0 + sp.arcsin(rho)
#    simil_mtx=0.3*sp.ones(rho.shape)
SC = SpectralClustering(n_clusters=nClusters, affinity='precomputed')
labs_all = SC.fit_predict(simil_mtx)+1


dfs_left_sm.attributes = rho
view_patch(dfs_left_sm,rho)

rho = np.dot(ref_mean_pc,tempsub.T)
rho[~np.isfinite(rho)] = 0
dfs_left.attributes = rho
view_patch(dfs_left_sm,rho)

#sm=smooth_patch(dfs_left,iter=1000)
#view_patch(sm)
#view_patch(dfs_left,rho)

sub_rot = rot_sub_data(temp, tempsub)

rho = sp.dot(ref_mean_pc,sub_rot.T)
#rho=rho[0,1:]
rho[~np.isfinite(rho)] = 0
dfs_left.attributes = rho
view_patch(dfs_left_sm,rho)
