# ||AUM||
from dfsio import readdfs
import scipy.io
import numpy as np
from mayavi import mlab
#import h5py
import os
#from scipy.stats import trim_mean
import scipy.sparse as sp
from sklearn.cluster import SpectralClustering
import matplotlib.pylab as plt
p_dir = 'E:\\HCP-fMRI-NLM'
lst = os.listdir(p_dir)
r_factor = 3
ref_dir = os.path.join(p_dir, 'reference')


ref = '100307'
print(ref + '.reduce' + str(r_factor) + '.LR_mask.mat')
fn1 = ref + '.reduce' + str(r_factor) + '.LR_mask.mat'
fname1 = os.path.join(ref_dir, fn1)
msk = scipy.io.loadmat(fname1)  # h5py.File(fname1);
dfs_left = readdfs(os.path.join(p_dir, 'reference', ref + '.aparc.a2009s.32k_fs.reduce3.left.dfs'))
dfs_left_sm = readdfs(os.path.join(p_dir, 'reference', ref + '.aparc.a2009s.32k_fs.reduce3.very_smooth.left.dfs'))
count1 = 0
rho_rho=[];rho_all=np.zeros([0,0])
for sub in lst:
    try:
        data = scipy.io.loadmat(os.path.join(p_dir, sub, sub + '.rfMRI_REST1_RL.reduce3.ftdata.NLM_11N_hvar_25.mat'))
    except:
        continue

    LR_flag = msk['LR_flag']
    LR_flag = np.squeeze(LR_flag) > 0
    data = data['ftdata_NLM']
    temp = data[LR_flag, :]
    m = np.mean(temp, 1)
    temp = temp - m[:,None]
    s = np.std(temp, 1)+1e-16
    temp = temp/s[:,None]
    msk_small_region = (dfs_left.labels == 46) | (dfs_left.labels == 28) | (dfs_left.labels == 29)  # % motor
    d = temp[msk_small_region, :]
    d_corr = temp[~msk_small_region, :]
    rho = np.corrcoef(d, d_corr)
    rho=rho[range(d.shape[0]),d.shape[0]+1:]
    rho[~np.isfinite(rho)] = 0
    if count1==0:
        rho_all=rho
    else:
        rho_all=np.vstack([rho_all,rho])
   
    rho_rho.append(np.corrcoef(rho))
    count1+=1
    print(count1),
    if count1 >4 :
        break

B=np.corrcoef(rho_all)
B[~np.isfinite(B)] = 0
B=np.abs(B)
plt.figure()
plt.imshow(np.abs(rho_all))
plt.show()
plt.figure()
plt.imshow(np.abs(B))
plt.show()

A = sp.block_diag(rho_rho)
print(A)

A.shape
nNodes=rho.shape[0]
nSub=len(rho_rho)
A=A.tolil()
for jj in range(nSub):
    for kk in range(nSub):
        if jj == kk:
            continue
        A[(jj*nNodes+np.arange(nNodes)),(kk*nNodes+np.arange(nNodes))]=1
 #       B[(jj*nNodes+np.arange(nNodes)),(kk*nNodes+np.arange(nNodes))]=B[(jj*nNodes+np.arange(nNodes)),(kk*nNodes+np.arange(nNodes))]+1

plt.spy(A,precision=0.01, markersize=1)
plt.show()

for nClusters in [6]:#[2,4,6,8,10,12]:
    #SC=SpectralClustering(n_clusters=nClusters,affinity='precomputed')
    SC=SpectralClustering(n_clusters=nClusters,assign_labels='discretize')
    labs=SC.fit_predict(rho_all)

    for kk in range(nSub):
        r=dfs_left_sm;r.labels=r.labels*0;r.labels[msk_small_region]=labs[kk*nNodes+np.arange(nNodes)]+1
        mesh = mlab.triangular_mesh(r.vertices[:,0], r.vertices[:,1], r.vertices[:,2], r.faces, representation='surface',
                                opacity=1,scalars=np.float64(r.labels))
        #mlab.pipeline.surface(mesh)

        mlab.gcf().scene.parallel_projection = True
        mlab.view(azimuth=0, elevation=-90)
        mlab.savefig(filename = 'c'+str(nClusters)+'s'+str(kk)+'labels1_1.png')
 #       mlab.view(azimuth=0, elevation=90)
 #       mlab.savefig(filename = str(nClusters)+str(kk)+'labels2_1.png')
        mlab.close()

