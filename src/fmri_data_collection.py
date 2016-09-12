import time
import scipy as sp
from centroid import search, find_location_smallmask, spatial_map, change_labels, change_order
from dfsio import readdfs
import scipy.io
import numpy as np
# import h5py
import os
# from scipy.stats import trim_mean
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
import networkx as nx
from sklearn.mixture import GMM


def ALL_parcellate_region(roilist, sub,R_all,scan_type):
    p_dir = '/home/ajoshi/data/HCP_data'
    r_factor = 3
    ref_dir = os.path.join(p_dir, 'reference')
    ref = '100307'
    fn1 = ref + '.reduce' + str(r_factor) + '.LR_mask.mat'
    fname1 = os.path.join(ref_dir, fn1)
    msk = scipy.io.loadmat(fname1)  # h5py.File(fname1);
    '''dfs_left = readdfs(os.path.join(p_dir, 'reference', ref + '.aparc.\
a2009s.32k_fs.reduce3.left.dfs'))
    dfs_left_sm = readdfs(os.path.join(p_dir, 'reference', ref + '.aparc.\
a2009s.32k_fs.reduce3.very_smooth.left.dfs'))'''

    dfs_left_sm = readdfs(os.path.join('/home/ajoshi/for_gaurav', '100307.BCI2reduce3.very_smooth.'+scan_type+'.dfs'))
    dfs_left = readdfs(os.path.join('/home/ajoshi/for_gaurav', '100307.BCI2reduce3.very_smooth.'+scan_type+'.dfs'))

    data = scipy.io.loadmat(os.path.join(p_dir,'data', sub, sub + '.rfMRI_REST' + str(1) + '_RL.\
reduce3.ftdata.NLM_11N_hvar_25.mat'))

    LR_flag = msk['LR_flag']

    # 0= right hemisphere && 1== left hemisphere
    if scan_type == 'right' :
        LR_flag = np.squeeze(LR_flag) == 0
    else :
        LR_flag = np.squeeze(LR_flag) == 1
    data = data['ftdata_NLM']
    temp = data[LR_flag, :]
    m = np.mean(temp, 1)
    temp = temp - m[:, None]
    s = np.std(temp, 1) + 1e-16
    temp = temp / s[:, None]
    msk_small_region = np.in1d(dfs_left.labels, roilist)
    if R_all.size == 0:
        R_all=temp
    else:
        R_all=np.concatenate((R_all,temp),axis=1)
    print R_all.shape[1]

    return (R_all,msk_small_region,dfs_left_sm.vertices,dfs_left_sm.faces)
