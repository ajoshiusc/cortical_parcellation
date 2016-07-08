from mayavi import mlab
import scipy as sp
from dfsio import readdfs
import scipy.io
import numpy as np
import os
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
import networkx as nx
from sklearn.mixture import GMM


def own_def(roilist, sub, nClusters, scan, scan_type, savepng=0, session=1, algo=0, type_cor=0):
    p_dir = '/home/ajoshi/HCP_data'
    r_factor = 3
    ref_dir = os.path.join(p_dir, 'reference')
    ref = '100307'
    fn1 = ref + '.reduce' + str(r_factor) + '.LR_mask.mat'
    fname1 = os.path.join(ref_dir, fn1)
    msk = scipy.io.loadmat(fname1)  # h5py.File(fname1);
    dfs_left = readdfs(os.path.join(p_dir, 'reference', ref + '.aparc.a2009s.32k_fs.reduce3.' + 'left' + '.dfs'))
    dfs_left_sm = readdfs(os.path.join(p_dir, 'reference', ref + '.aparc.\
a2009s.32k_fs.reduce3.very_smooth.' + 'left' + '.dfs'))
    data = scipy.io.loadmat(
        os.path.join(p_dir, sub, sub + '.rfMRI_REST' + str(session) + scan + '.reduce3.ftdata.NLM_11N_hvar_25.mat'))

    LR_flag = msk['LR_flag']
    LR_flag = np.squeeze(LR_flag) > 0
    data = data['ftdata_NLM']
    temp = data[LR_flag, :]
    m = np.mean(temp, 1)
    temp = temp - m[:, None]
    s = np.std(temp, 1) + 1e-16
    temp = temp / s[:, None]
    msk_small_region = np.in1d(dfs_left.labels, roilist)
    d = temp[msk_small_region, :]
    rho=np.corrcoef(d)
    rho[~np.isfinite(rho)]=0
    if algo == 0:
        SC = SpectralClustering(n_clusters=nClusters, affinity='precomputed')
        labels = SC.fit_predict(rho)

    if savepng > 0:
        r = dfs_left_sm
        r.labels = np.zeros([r.vertices.shape[0]])
        r.labels[msk_small_region] = labels + 1

        mlab.triangular_mesh(r.vertices[:, 0], r.vertices[:, 1], r.vertices[:,
                                                                 2], r.faces, representation='surface',
                             opacity=1, scalars=np.float64(r.labels))

        mlab.gcf().scene.parallel_projection = True
        mlab.view(azimuth=0, elevation=90)
        mlab.colorbar(orientation='horizontal')
        mlab.show()





p_dir = '/home/ajoshi/HCP_data'
lst = os.listdir(p_dir) #{'100307'}
#%%
mean_R = sp.zeros(6)
std_R = sp.zeros(6)
sdir=['_LR','_RL']
scan_type=['left','right']
session_type=[1,2]
fadd_1='.rfMRI_REST'
fadd_2='.reduce3.ftdata.NLM_11N_hvar_25.mat'

roilist = []
'''for i in range(19,20):
    roilist.append(i)'''
roilist=np.array([6,7,8,9,10,68])
#nClusters=num_of_cluster()
nClusters=1
count_break=0

for sub in lst:
    if os.path.isfile(os.path.join(p_dir, sub, sub + fadd_1 + str(session_type[0]) + sdir[1] + fadd_2)):
            # (46,28,29) motor 243 is precuneus
        own_def(roilist, sub, nClusters, sdir[1], scan_type[0],1, session_type[0], 0, 0)
        if count_break == 0:
            break