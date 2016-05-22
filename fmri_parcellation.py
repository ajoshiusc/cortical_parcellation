# -*- coding: utf-8 -*-
"""
Created on Tue Mar 01 14:51:12 2016

@author: ajoshi
"""
from dfsio import readdfs
import scipy.io
import numpy as np
from mayavi import mlab
# import h5py
import os
# from scipy.stats import trim_mean
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
import networkx as nx
from sklearn.mixture import GMM


def parcellate_motor(sub, nClusters, savepng=0, session=1, algo=0):
    p_dir = 'E:\\HCP-fMRI-NLM'
    r_factor = 3
    ref_dir = os.path.join(p_dir, 'reference')
    ref = '100307'
    fn1 = ref + '.reduce' + str(r_factor) + '.LR_mask.mat'
    fname1 = os.path.join(ref_dir, fn1)
    msk = scipy.io.loadmat(fname1)  # h5py.File(fname1);
    dfs_left = readdfs(os.path.join(p_dir, 'reference', ref + '.aparc.\
a2009s.32k_fs.reduce3.left.dfs'))
    dfs_left_sm = readdfs(os.path.join(p_dir, 'reference', ref + '.aparc.\
a2009s.32k_fs.reduce3.very_smooth.left.dfs'))
    data = scipy.io.loadmat(os.path.join(p_dir, sub, sub + '.rfMRI_REST'+str(session)+'_RL.\
reduce3.ftdata.NLM_11N_hvar_25.mat'))

    LR_flag = msk['LR_flag']
    LR_flag = np.squeeze(LR_flag) > 0
    data = data['ftdata_NLM']
    temp = data[LR_flag, :]
    m = np.mean(temp, 1)
    temp = temp - m[:, None]
    s = np.std(temp, 1) + 1e-16
    temp = temp / s[:, None]
    msk_small_region = (dfs_left.labels == 46) | (dfs_left.labels == 28) \
        | (dfs_left.labels == 29)  # % motor
    d = temp[msk_small_region, :]
    d_corr = temp[~msk_small_region, :]
    rho = np.corrcoef(d, d_corr)
    rho = rho[range(d.shape[0]), d.shape[0] + 1:]
    rho[~np.isfinite(rho)] = 0
    B = np.corrcoef(rho)
    B[~np.isfinite(B)] = 0
    B = np.abs(B)

    # SC = DBSCAN()
    if algo == 0:
        SC = SpectralClustering(n_clusters=nClusters, affinity='precomputed')
        labels = SC.fit_predict(B)
    elif algo == 1:
        g = nx.Graph()
        g.add_edges_from(dfs_left.faces[:, (0, 1)])
        g.add_edges_from(dfs_left.faces[:, (1, 2)])
        g.add_edges_from(dfs_left.faces[:, (2, 0)])
        Adj = nx.adjacency_matrix(g)
        AdjS = Adj[(msk_small_region), :]
        AdjS = AdjS[:, (msk_small_region)]
        AdjS = AdjS.todense()
        np.fill_diagonal(AdjS, 1)
        SC = AgglomerativeClustering(n_clusters=nClusters,
                                     connectivity=AdjS)
        labels = SC.fit_predict(B)
    elif algo == 2:
        GM = GMM(n_components=nClusters,covariance_type='full',n_iter=100)
        GM.fit(rho)        
        labels = GM.predict(rho)
        
        

    if savepng > 0:
        r = dfs_left_sm
        r.labels = np.zeros([r.vertices.shape[0]])
        r.labels[msk_small_region] = labels
        mlab.triangular_mesh(r.vertices[:, 0], r.vertices[:, 1], r.vertices[:,
                             2], r.faces, representation='surface',
                             opacity=1, scalars=np.float64(r.labels))
        mlab.gcf().scene.parallel_projection = True
        mlab.view(azimuth=0, elevation=-90)
        mlab.savefig(filename='clusters_' + str(nClusters) + 'subject_' +
                     sub + 'session' + str(session) + '_labels.png')
        mlab.close()

    return labels
