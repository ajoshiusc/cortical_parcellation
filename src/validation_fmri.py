import os
import scipy as sp
import numpy as np
from separate_cluster import separate
from centroid import search, find_location_smallmask, affinity_mat, change_labels, change_order, neighbour_correlation
from dfsio import readdfs
import scipy.io
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
import networkx as nx
from sklearn.mixture import GMM

from surfproc import patch_color_labels, view_patch


def parcellate_region(roilist, sub, nClusters, scan, scan_type, savepng=0, session=1, algo=0, type_cor=0):
    p_dir = '/home/ajoshi/data/HCP_data'
    r_factor = 3
    ref_dir = os.path.join(p_dir, 'reference')
    ref = '100307'
    fn1 = ref + '.reduce' + str(r_factor) + '.LR_mask.mat'
    fname1 = os.path.join(ref_dir, fn1)
    msk = scipy.io.loadmat(fname1)

    dfs_left_sm = readdfs(
        os.path.join('/home/ajoshi/for_gaurav', '100307.BCI2reduce3.very_smooth.' + scan_type + '.dfs'))
    dfs_left = readdfs(os.path.join('/home/ajoshi/for_gaurav', '100307.BCI2reduce3.very_smooth.' + scan_type + '.dfs'))

    data = scipy.io.loadmat(os.path.join(p_dir, 'data', sub, sub + '.rfMRI_REST' + str(
        session) + scan + '.reduce3.ftdata.NLM_11N_hvar_25.mat'))

    LR_flag = msk['LR_flag']
    # 0= right hemisphere && 1== left hemisphere
    if scan_type == 'right':
        LR_flag = np.squeeze(LR_flag) == 0
    else:
        LR_flag = np.squeeze(LR_flag) == 1
    data = data['ftdata_NLM']
    temp = data[LR_flag, :]
    m = np.mean(temp, 1)
    temp = temp - m[:, None]
    s = np.std(temp, 1) + 1e-16
    temp = temp / s[:, None]
    msk_small_region = np.in1d(dfs_left.labels, roilist)
    #    (dfs_left.labels == 46) | (dfs_left.labels == 28) \
    #       | (dfs_left.labels == 29)  # % motor
    d = temp[msk_small_region, :]
    rho = np.corrcoef(d)
    rho[~np.isfinite(rho)] = 0
    # rho=np.abs(rho)
    d_corr = temp[~msk_small_region, :]
    rho_1 = np.corrcoef(d, d_corr)
    rho_1 = rho_1[range(d.shape[0]), d.shape[0]:]
    rho_1[~np.isfinite(rho_1)] = 0
    if type_cor == 1:
        f_rho = np.arctanh(rho_1)
        f_rho[~np.isfinite(f_rho)] = 0
        B = np.corrcoef(f_rho)
        B[~np.isfinite(B)] = 0
        # B = np.abs(B)

    # SC = DBSCAN()
    if algo == 0:
        SC = SpectralClustering(n_clusters=nClusters, affinity='precomputed')
        # SC=SpectralClustering(n_clusters=nClusters,gamma=0.025)
        if type_cor == 0 and rho.size > 0:
            # affinity_matrix = affinity_mat(rho)
            affinity_matrix = np.arcsin(rho)
            labels = SC.fit_predict(np.abs(affinity_matrix))
        if type_cor == 1 and rho.size > 0:
            labels = SC.fit_predict(affinity_mat(B))
            # affinity_matrix=SC.fit(np.abs(d))
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
        GM = GMM(n_components=nClusters, covariance_type='full', n_iter=100)
        GM.fit(rho)
        labels = GM.predict(rho)

    elif algo == 3:
        neighbour_correlation(rho, dfs_left_sm.faces, dfs_left_sm.vertices, msk_small_region)

    if savepng > 0 :
        r = dfs_left_sm
        r.labels = np.zeros([r.vertices.shape[0]])
        r.labels[msk_small_region] = labels + 1

        cent = separate(labels, r, r.vertices, nClusters)

        manual_order = np.array([0 for x in range(nClusters)])
        save = np.array([0 for x in range(nClusters)])

        for i in range(0, nClusters):
            if nClusters > 1:
                choose_vector = np.argmax(cent.transpose(), axis=1)
                save[i] = cent[choose_vector[1]][1]
                correspondence_point = find_location_smallmask(r.vertices, cent[choose_vector[1]], msk_small_region)
                cent[choose_vector[1]][1] = -np.Inf
                manual_order[i] = choose_vector[1]
                if i == 0:
                    # change
                    correlation_within_precuneus_vector = sp.array(rho[correspondence_point])
                    correlation_with_rest_vector = sp.array(rho_1[correspondence_point])
                else:
                    correlation_within_precuneus_vector = sp.vstack(
                        [correlation_within_precuneus_vector, [rho[correspondence_point]]])
                    correlation_with_rest_vector = sp.vstack(
                        [correlation_with_rest_vector, [rho_1[correspondence_point]]])
            else:
                choose_vector = 0
                correspondence_point = find_location_smallmask(r.vertices, cent, msk_small_region)
                manual_order[i] = choose_vector
                if i == 0:
                    # change
                    correlation_within_precuneus_vector = sp.array(rho[correspondence_point])
                    correlation_with_rest_vector = sp.array(rho_1[correspondence_point])

        manual_order = change_order(manual_order, nClusters)
        r.labels = change_labels(r.labels, manual_order, nClusters)

        new_cent = separate(r.labels, r, temp, nClusters)

        if nClusters > 1:
            for i in range(0, nClusters):
                cent[manual_order[i]][1] = save[i]
    return (r, correlation_within_precuneus_vector, correlation_with_rest_vector, msk_small_region, new_cent)

class sc:
    pass

right_hemisphere=np.array([226,168,184,446,330,164,442,328,172,444,130,424,166,326,342,142,146,144,222,170,
150,242,186,120,422,228,224,322,310,162,324,500])

left_hemisphere=np.array([227,169,185,447,331,165,443,329,173,445,131,425,167,327,343,143,147,145,223,171,
151,243,187,121,423,229,225,323,311,163,325,501])

nClusters=np.array([3,1,3,2,2,2,3,3,2,2,2,3,1,4,1,2,1,3,2,1,4,2,1,2,2,2,2,3,1,2,1,2])

p_dir = '/big_disk/HCP100-fMRI-NLM/HCP100-fMRI-NLM'
lst = os.listdir(p_dir) #{'100307'}
save_dir= '/home/sgaurav/Documents/git_sandbox/cortical_parcellation/src/validation'

sdir=['_RL','_LR']
scan_type=['left','right']
session_type=[1,2]
fadd_1='.rfMRI_REST'
fadd_2='.reduce3.ftdata.NLM_11N_hvar_25.mat'

# %% Across session study
for sub in lst:
    for scan in range(0,4):
        if (sub is not 'reference' or 'zip1') and (os.path.isfile(os.path.join(p_dir, sub, sub + fadd_1 + str(session_type[scan%2]) + sdir[scan/2] + fadd_2))):
            for i in range(0,2):
                labs_all  = np.zeros([10832])
                count1 = 0
                all_centroid = []
                centroid = []
                label_count=0
                for n in range(nClusters.shape[0]):

                    print n
                    roiregion=left_hemisphere[n]
                    if i==1 :
                        roiregion=right_hemisphere[n]
                    labs1, correlation_within_roi_vector, correlation_with_rest_vector, mask, centroid = parcellate_region(
                        roiregion, sub, nClusters[n], sdir[scan/2], scan_type[i],
                        1, session_type[scan%2], 0, 0)

                    labs_all[mask]=labs1.labels[mask] +label_count
                    label_count += nClusters[n]

                sc.labels = labs_all
                sc.vertices = labs1.vertices
                sc.faces = labs1.faces
                sc.vColor = np.zeros([labs1.vertices.shape[0]])
                #sc = patch_color_labels(sc, cmap='Paired', shuffle=True)
                #view_patch(sc, show=1, colormap='Paired', colorbar=0)


                sp.savez(os.path.join(save_dir, str(sub) + '_' + scan_type[i]  + sdir[scan/2] + '_' + str(session_type[scan%2]) + '.npz'),
                    labels=sc.labels, vertices=sc.vertices,
                    faces=sc.faces)

