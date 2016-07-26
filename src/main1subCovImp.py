# ||AUM||
import scipy.io
import scipy as sp
import numpy as np
from dfsio import readdfs, writedfs
from mayavi import mlab
#import h5py
import os
from surfproc import view_patch, view_patch_vtk, get_cmap
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
nClusters=3

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

silhouette_avg=sp.zeros(len(lst))
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
    roilist=sp.unique(dfs_left.labels)
    msk_small_region = np.in1d(dfs_left.labels,roilist)
#    msk_small_region = (dfs_left.labels == 30) | (dfs_left.labels == 72) | (dfs_left.labels == 9) |  (dfs_left.labels == 47)  # % motor
    d = temp[msk_small_region, :]
    rho = np.corrcoef(d)
    rho[~np.isfinite(rho)] = 0
    
#    simil_mtx=sp.exp((rho-1)/0.7)
    simil_mtx=sp.pi/2.0 + sp.arcsin(rho)
#    simil_mtx=0.3*sp.ones(rho.shape)
    SC = SpectralClustering(n_clusters=nClusters, affinity='precomputed')
    labs = SC.fit_predict(simil_mtx)+1
    silhouette_avg[count1] = silhouette_score(sp.pi-simil_mtx, labs-1, metric='precomputed')
    
    print(count1),    
    
    r = dfs_left_sm;r.labels = r.labels*0;r.labels[msk_small_region] = labs
    labs_all[:,count1]=r.labels
    count1+=1


print("Avg. Silhoutte Score is:%f"%sp.mean(silhouette_avg))
 
labs_all0_vec = sp.zeros((labs_all.shape[0],nClusters+1),'bool')
labs_alli_vec = labs_all0_vec.copy()
for i in range(nClusters+1):
    labs_all0_vec[:,i] = (labs_all[:,0]==i)
    
    
for i in range(1,40):
    for j in range(nClusters+1):
        labs_alli_vec[:,j] = (labs_all[:,i]==j)

    D=pairwise_distances(labs_alli_vec.T, labs_all0_vec.T, metric='dice')
    ind1 = linear_assignment(D)
    labs_all[:,i] = ind1[sp.int16(labs_all[:,i]),1]

s=r
s.vColor=sp.zeros(s.vertices.shape)
label_vert,lab_count=sp.stats.mode(labs_all.T)
colr=get_cmap(nClusters+1)
lab_count=sp.float32(lab_count.squeeze())
s.vColor=s.vColor+1

for i in range(len(s.vertices)):
#        print i, (lab_count[i]/sp.amax(lab_count)), colr(label_vert[0,i])[:3], (lab_count[i]/sp.amax(lab_count)), 0.55*sp.array(colr(label_vert[0,i])[:3])
    if label_vert[0,i]>0 :
        freq=((lab_count[i]/sp.amax(lab_count)) - 1.0/nClusters)*(sp.float32(nClusters)/(nClusters-1.0))
        s.vColor[i,]=(1-freq) + freq*sp.array(colr(label_vert[0,i])[:3])
        
view_patch(s)
view_patch_vtk(s)    
writedfs('outclustering_pc.dfs',s)
#    view_patch(r,r.labels)
    #mesh = mlab.triangular_mesh(r.vertices[:,0], r.vertices[:,1], r.vertices[:,2], r.faces, representation='surface',
#                            opacity=1,scalars=np.float64(r.labels))
#    #mlab.pipeline.surface(mesh)
#mlab.gcf().scene.parallel_projection = True
#mlab.view(azimuth=0, elevation=-90)
#mlab.show()
#mlab.savefig(filename = 'dict_learning_1.png')
#mlab.view(azimuth=0, elevation=90)
#mlab.savefig(filename = 'dict_learning_2.png')
#mlab.close()

