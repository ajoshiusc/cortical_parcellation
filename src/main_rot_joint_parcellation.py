# ||AUM||
import scipy.io
import scipy as sp
import numpy as np
from fmri_methods_sipi import rot_sub_data
from dfsio import readdfs, writedfs
#import h5py
import os
from surfproc import view_patch, view_patch_vtk, get_cmap
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.metrics import silhouette_score
p_dir = '/home/ajoshi/HCP_data'
p_dir_ref='/home/ajoshi/HCP_data'
lst = os.listdir(p_dir)
lst=lst[0:2]
r_factor = 3
ref_dir = os.path.join(p_dir_ref, 'reference')
nClusters=60

ref = '100307'
print(ref + '.reduce' + str(r_factor) + '.LR_mask.mat')
fn1 = ref + '.reduce' + str(r_factor) + '.LR_mask.mat'
fname1 = os.path.join(ref_dir, fn1)
msk = scipy.io.loadmat(fname1)  # h5py.File(fname1);
dfs_left = readdfs(os.path.join(p_dir_ref, 'reference', ref + '.aparc.a2009s.32k_fs.reduce3.left.dfs'))
dfs_left_sm = readdfs(os.path.join(p_dir_ref, 'reference', ref + '.aparc.a2009s.32k_fs.reduce3.very_smooth.left.dfs'))
count1 = 0
rho_rho=[];rho_all=[]

labs_all=sp.zeros((len(dfs_left.labels),len(lst)))
#roilist=[30,72,9,47] #prec
roilist=[6,7,8,9,10] #cing
#roilist=[2,22,11,58,59,20,43,19,45] #vis

for sub in lst:
#sub ='751348' #; lst:
    data = scipy.io.loadmat(os.path.join(p_dir, sub, sub + '.rfMRI_REST1_RL.reduce3.ftdata.NLM_11N_hvar_25.mat'))
    LR_flag = msk['LR_flag']
    LR_flag = np.squeeze(LR_flag) > 0
    data = data['ftdata_NLM']
    temp = data[LR_flag, :]
    m = np.mean(temp, 1)
    temp = temp - m[:,None]
    s = np.std(temp, 1)+1e-16
    temp = temp/s[:,None]
    msk_small_region = np.in1d(dfs_left.labels,roilist)
#    msk_small_region = (dfs_left.labels == 30) | (dfs_left.labels == 72) | (dfs_left.labels == 9) |  (dfs_left.labels == 47)  # % motor
    d = temp#[msk_small_region, :]
    
    if count1==0:        
        sub_data = sp.zeros((d.shape[0],d.shape[1],len(lst)))

    sub_data[:,:,count1] = d

    count1+=1
    print count1,
    
nSub=sub_data.shape[2]

cat_data=sp.zeros((nSub*sub_data.shape[0],sub_data.shape[1]))

for ind in range(nSub):
    sub_data[:,:,ind] = rot_sub_data(ref=sub_data[:,:,0],sub=sub_data[:,:,ind])
    cat_data[sub_data.shape[0]*ind:sub_data.shape[0]*(ind+1),:] = sub_data[:,:,ind]    
    print ind, sub_data.shape, cat_data.shape

 
rho=sp.corrcoef(cat_data)
rho[~np.isfinite(rho)] = 1

simil_mtx=sp.pi/2.0 + sp.arcsin(rho)
SC = SpectralClustering(n_clusters=nClusters, affinity='precomputed')
labs_all = SC.fit_predict(simil_mtx)+1

lab_sub=labs_all.reshape((sub_data.shape[0],nSub),order='F')
lab1=sp.zeros(dfs_left_sm.vertices.shape[0])
for ind in range(nSub):
#    lab1[msk_small_region]=lab_sub[:,ind]
    lab1=lab_sub[:,ind]
    view_patch(dfs_left_sm,lab1)
    