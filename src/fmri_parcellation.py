import scipy as sp
from separate_cluster import separate
from centroid import  search, find_location_smallmask, affinity_mat, change_labels, change_order, neighbour_correlation
from dfsio import readdfs
import scipy.io
import numpy as np
#from mayavi import mlab
# import h5py
import numpy as np
from sklearn.decomposition import PCA
import os
# from scipy.stats import trim_mean
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
import networkx as nx
from sklearn.mixture import GMM


def parcellate_region(roilist, sub, nClusters, scan,scan_type,savepng=0, session=1, algo=0,type_cor=0):
    p_dir = '/home/ajoshi/data/HCP_data'
    r_factor = 3
    ref_dir = os.path.join(p_dir, 'reference')
    ref = '100307'
    fn1 = ref + '.reduce' + str(r_factor) + '.LR_mask.mat'
    fname1 = os.path.join(ref_dir, fn1)
    msk = scipy.io.loadmat(fname1)  # h5py.File(fname1);
    #dfs_left = readdfs(os.path.join(p_dir, 'reference', ref + '.aparc.a2009s.32k_fs.reduce3.'+scan_type+'.dfs'))
    #dfs_left_sm = readdfs(os.path.join(p_dir, 'reference', ref + '.aparc.\
#a2009s.32k_fs.reduce3.very_smooth.'+scan_type+'.dfs'))


    dfs_left_sm = readdfs(os.path.join('/home/ajoshi/for_gaurav','100307.BCI2reduce3.very_smooth.'+scan_type+'.dfs'))
    dfs_left = readdfs(os.path.join('/home/ajoshi/for_gaurav','100307.BCI2reduce3.very_smooth.'+scan_type+'.dfs'))


    data = scipy.io.loadmat(os.path.join(p_dir, 'data',sub, sub + '.rfMRI_REST'+str(session)+scan+'.reduce3.ftdata.NLM_11N_hvar_25.mat'))

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
    msk_small_region = np.in1d(dfs_left.labels,roilist)
#    (dfs_left.labels == 46) | (dfs_left.labels == 28) \
 #       | (dfs_left.labels == 29)  # % motor
    d = temp[msk_small_region, :]
    rho=np.corrcoef(d)
    rho[~np.isfinite(rho)]=0
    #rho=np.abs(rho)
    d_corr = temp[~msk_small_region, :]
    rho_1 = np.corrcoef(d, d_corr)
    rho_1 = rho_1[range(d.shape[0]), d.shape[0] :]
    rho_1[~np.isfinite(rho_1)] = 0
    if type_cor==1:
        f_rho=np.arctanh(rho_1)
        f_rho[~np.isfinite(f_rho)]=0
        f_rho = np.corrcoef(f_rho)
        f_rho[~np.isfinite(f_rho)] = 0
        #B = np.abs(B)



    # SC = DBSCAN()
    if algo == 0:
        SC = SpectralClustering(n_clusters=nClusters,affinity='precomputed')
        #SC=SpectralClustering(n_clusters=nClusters,gamma=0.025)
        if type_cor==0 and rho.size>0:
            #affinity_matrix = affinity_mat(rho)
            affinity_matrix=np.arcsin(rho)
            labels = SC.fit_predict(np.abs(affinity_matrix))
        if type_cor ==1 and rho.size>0 :
            labels = SC.fit_predict(affinity_mat(B))
        #affinity_matrix=SC.fit(np.abs(d))
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
        labels = SC.fit_predict(rho)
    elif algo == 2:
        GM = GMM(n_components=nClusters,covariance_type='full',n_iter=100)
        GM.fit(rho)
        labels = GM.predict(rho)

    elif algo == 3:
        neighbour_correlation(rho, dfs_left_sm.faces, dfs_left_sm.vertices,msk_small_region)
        

    if savepng > 0 and labels.size>0:
        r = dfs_left_sm
        r.labels = np.zeros([r.vertices.shape[0]])
        r.labels[msk_small_region] = labels+1

        cent=separate(labels,r,r.vertices,nClusters)

        manual_order=np.array([0 for x in range(nClusters)])
        save=np.array([0 for x in range(nClusters)])

        for i in range(0,nClusters):
            if nClusters > 1:
                choose_vector=np.argmax(cent.transpose(),axis=1)
                save[i]=cent[choose_vector[1]][1]
                correspondence_point=find_location_smallmask(r.vertices,cent[choose_vector[1]],msk_small_region)
                cent[choose_vector[1]][1]=-np.Inf
                manual_order[i]=choose_vector[1]
                if i == 0:
                    #change
                    correlation_within_precuneus_vector=sp.array(rho[correspondence_point])
                    correlation_with_rest_vector=sp.array(rho_1[correspondence_point])
                else:
                    correlation_within_precuneus_vector=sp.vstack([correlation_within_precuneus_vector,[rho[correspondence_point]]])
                    correlation_with_rest_vector=sp.vstack([correlation_with_rest_vector,[rho_1[correspondence_point]]])
            else:
                choose_vector = 0
                correspondence_point = find_location_smallmask(r.vertices, cent, msk_small_region)
                manual_order[i] = choose_vector
                if i == 0:
                    # change
                    correlation_within_precuneus_vector = sp.array(rho[correspondence_point])
                    correlation_with_rest_vector = sp.array(rho_1[correspondence_point])

        manual_order=change_order(manual_order,nClusters)
        r.labels = change_labels(r.labels,manual_order,nClusters)

        new_cent=separate(r.labels,r,temp,nClusters)

        if nClusters > 1:
            for i in range(0,nClusters):
                cent[manual_order[i]][1]=save[i]

        '''mlab.triangular_mesh(r.vertices[:, 0], r.vertices[:, 1], r.vertices[:,
                                                                 2], r.faces, representation='surface',
                             opacity=1, scalars=np.float64(r.labels))

        for i in range(nClusters):
            mlab.points3d(new_cent[i][0], new_cent[i][1], new_cent[i][2])

        mlab.gcf().scene.parallel_projection = True
        mlab.view(azimuth=0, elevation=90)
        mlab.colorbar(orientation='horizontal')
        mlab.draw()
        mlab.savefig(filename='clusters_' + str(nClusters) + '_rois_' + str(roilist) + 'subject_' +
                              sub + 'session' + str(session) + '_labels.png')

        mlab.close()'''

    #return (r,correspondence_vector,msk_small_region)
    return (r, correlation_within_precuneus_vector, correlation_with_rest_vector, msk_small_region,new_cent)
