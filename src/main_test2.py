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
import random
from surfproc import patch_color_labels, view_patch_vtk, patch_color_attrib
from fmri_methods_sipi import region_growing_fmri


class sc:
    pass

#rlist = [21]  # precuneus
#rlist = [10]  # middle frontal gyrus
rlist = [13]  # middle temporal gyrus

right_hemisphere=np.array([226,168,184,446,330,164,442,328,172,444,130,424,166,326,342,142,146,144,222,170,
150,242,186,120,422,228,224,322,310,162,324,500])

left_hemisphere=np.array([227,169,185,447,331,165,443,329,173,445,131,425,167,327,343,143,147,145,223,171,
151,243,187,121,423,229,225,323,311,163,325,501])

nClusters=np.array([3,1,3,2,2,2,3,3,2,2,2,3,1,4,1,2,1,3,2,1,4,2,1,2,2,2,2,3,1,2,1,2])
right_hemisphere=right_hemisphere[rlist]
left_hemisphere=left_hemisphere[rlist]
nClusters=nClusters[rlist]

p_dir = '/big_disk/ajoshi/HCP100-fMRI-NLM/HCP100-fMRI-NLM'
lst = os.listdir(p_dir) #{'100307'}
old_lst = [] #os.listdir('/home/ajoshi/data/HCP_data/data')
old_lst+=['reference','zip1'] #,'106016','366446']
save_dir= '/big_disk/ajoshi/fmri_validation'

sdir=['_RL','_LR']
scan_type=['left','right']
session_type=[1,2]
fadd_1='.rfMRI_REST'
fadd_2='.reduce3.ftdata.NLM_11N_hvar_25.mat'

#%%Across session study

# for sub in lst:
sub = lst[1]
scan = 0
i = 0
#    for scan in range(0,4):
#        if (sub not in old_lst) and (os.path.isfile(os.path.join(p_dir, sub, sub + fadd_1 + str(session_type[scan%2]) + sdir[scan/2] + fadd_2))):
#            for i in range(0,2):
labs_all  = np.zeros([10832])
count1 = 0
n_rep = 100
all_centroid = []
centroid = []
label_count = 0
# for n in range(nClusters.shape[0]):
#    print n
n = 0
roiregion = left_hemisphere[n]

s_a = readdfs('100307.reduce3.very_smooth.left.refined.dfs')

seeds = sp.zeros(nClusters)
# seeds_roi = sp.zeros(nClusters)


col = sp.zeros(s_a.vertices.shape[0])
col[sp.int16(seeds)] = sp.arange(len(seeds))+1
s_a = patch_color_attrib(s_a, col)
ref_dir = os.path.join(p_dir, 'reference')
ref = '100307'
fn1 = ref + '.reduce' + str(3) + '.LR_mask.mat'
fname1 = os.path.join(ref_dir, fn1)
msk = scipy.io.loadmat(fname1)

data = scipy.io.loadmat(os.path.join(p_dir,  sub, sub + '.rfMRI_REST' + str(
        1) + '_LR' + '.reduce3.ftdata.NLM_11N_hvar_25.mat'))
LR_flag = 0
data = data['ftdata_NLM']
LR_flag = msk['LR_flag'].squeeze()
LR_flag = np.squeeze(LR_flag) == 1

dfs_left = readdfs(os.path.join('/home/ajoshi/for_gaurav',
                                '100307.BCI2reduce3.very_smooth.' +
                                'left' + '.dfs'))

temp = data[LR_flag, :]
m = np.mean(temp, 1)
temp = temp - m[:, None]
s = np.std(temp, 1) + 1e-16
temp = temp / s[:, None]
msk_small_region = np.in1d(dfs_left.labels, roiregion)

for ind in range(nClusters):
    lind = s_a.labels[msk_small_region] == roiregion * 10 + ind + 1
    print lind.sum()
    lind = sp.where(lind)[0]
    vert = s_a.vertices[msk_small_region, ]
    m = sp.mean(vert[lind, ], axis=0)
    dist = vert[lind, ] - m
    diff = sp.sum(dist**2, axis=1)
    indc = sp.argmin(diff)
    print lind[indc]
    seeds[ind] = lind[indc]
    #  seeds_roi[ind] = indc


n_samples = temp.shape[1]
ind = random.sample(range(temp.shape[1]), n_samples)
temp = temp[:, ind]
d = temp[msk_small_region, :]
rho = np.corrcoef(d)
rho[~np.isfinite(rho)] = 0
d_corr = temp[~msk_small_region, :]
rho_1 = np.corrcoef(d, d_corr)
rho_1 = rho_1[range(d.shape[0]), d.shape[0]:]
rho_1[~np.isfinite(rho_1)] = 0
f_rho = np.arctanh(rho_1)
f_rho[~np.isfinite(f_rho)] = 0
B = np.corrcoef(f_rho)
B[~np.isfinite(B)] = 0
affinity_matrix = np.arcsin(rho)

# conn = sp.sparse.lil_matrix((dfs_left.vertices.shape[0],
#                             dfs_left.vertices.shape[0]))
conn = sp.eye(dfs_left.vertices.shape[0])

conn[dfs_left.faces[:, 0], dfs_left.faces[:, 1]] = 1
conn[dfs_left.faces[:, 1], dfs_left.faces[:, 2]] = 1
conn[dfs_left.faces[:, 0], dfs_left.faces[:, 2]] = 1
conn = conn + conn.T
conn = conn > 0
conn = conn[msk_small_region, ]
conn = conn[:, msk_small_region]
labs = region_growing_fmri(seeds, affinity_matrix, conn)

col = 0*col
col[msk_small_region] = labs + 1
ms = sp.where(msk_small_region)
ms = sp.array(ms).squeeze()
col[ms[sp.int16(seeds)]] = 10
s_a = patch_color_attrib(s_a, col)
view_patch_vtk(s_a)
s_a.labels[~msk_small_region] = 0
s_a = patch_color_labels(s_a)
view_patch_vtk(s_a)
print sp.sum(labs <0)
