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
    p_dir = '/home/ajoshi/data/HCP_data'
    r_factor = 3
    ref_dir = os.path.join(p_dir, 'reference')
    ref = '100307'
    fn1 = ref + '.reduce' + str(r_factor) + '.LR_mask.mat'
    fname1 = os.path.join(ref_dir, fn1)
    msk = scipy.io.loadmat(fname1)  # h5py.File(fname1);


    #dfs_left = readdfs(os.path.join(p_dir, 'reference', ref + '.aparc.a2009s.32k_fs.reduce3.' + 'left' + '.dfs'))
    #dfs_left_sm = readdfs(os.path.join(p_dir, 'reference', ref + '.aparc.\
#a2009s.32k_fs.reduce3.very_smooth.' + 'left' + '.dfs'))

    dfs_left_sm = readdfs(os.path.join('/home/ajoshi/for_gaurav', '100307.BCI2reduce3.very_smooth.'+scan_type+'.dfs'))
    dfs_left = readdfs(os.path.join('/home/ajoshi/for_gaurav', '100307.BCI2reduce3.very_smooth.'+scan_type+'.dfs'))

    data = scipy.io.loadmat(
        os.path.join(p_dir, 'data',sub, sub + '.rfMRI_REST' + str(session) + scan + '.reduce3.ftdata.NLM_11N_hvar_25.mat'))

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
    d = temp[msk_small_region, :]
    rho=np.corrcoef(d)
    rho[~np.isfinite(rho)]=0
    if algo == 0:
        SC = SpectralClustering(n_clusters=nClusters, affinity='precomputed')
        labels = SC.fit_predict(np.abs(rho))

    if savepng > 0:
        r = dfs_left_sm
        r.labels = np.zeros([r.vertices.shape[0]])
        r.labels[msk_small_region] = labels+1

        mlab.triangular_mesh(r.vertices[:, 0], r.vertices[:, 1], r.vertices[:,
                                                                 2], r.faces, representation='surface',
                             opacity=1, scalars=np.float64(r.labels))

        mlab.gcf().scene.parallel_projection = True
        mlab.view(azimuth=0, elevation=-90)
        mlab.colorbar(orientation='horizontal')
        #mlab.show()
        mlab.savefig(filename='clusters_' + str(nClusters) + '_rois_' + str(roilist) + 'subject_' +
                     sub + 'session' + str(session) + '_labels.png')
        mlab.show()

p_dir = '/home/ajoshi/data/HCP_data/data'
lst = os.listdir(p_dir) #{'100307'}
#%%
mean_R = sp.zeros(6)
std_R = sp.zeros(6)
sdir=['_LR','_RL']
scan_type=['left','right']
session_type=[1,2]
fadd_1='.rfMRI_REST'
fadd_2='.reduce3.ftdata.NLM_11N_hvar_25.mat'


roilist =  [443,442]#np.array([[143],[142],[145],[144],[147],[146],[151],[150],[161],[160],[163],[162]])
roiregion=['parahippocampal gyrus','pars triangularis','pars orbitalis','pre-central gyrus','pole|orbital frontal lobe','transvers frontal gyrus']
nClusters=1
count_break=0
for j in range(0,2):
    for sub in lst:
        if os.path.isfile(os.path.join(p_dir, sub, sub + fadd_1 + str(session_type[0]) + sdir[0] + fadd_2)):
                    # (46,28,29) motor 243 is precuneus
            own_def(roilist[j], sub, nClusters, sdir[0], scan_type[j], 1, session_type[0], 0, 0)
            if count_break == 0:
                break