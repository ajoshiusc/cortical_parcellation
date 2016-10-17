# -*- coding: utf-8 -*-
# %%
"""
Created on Tue Jul 26 23:31:10 2016

@author: ajoshi
"""
# ||AUM||
from scipy.stats import itemfreq
import scipy.io
import scipy as sp
import numpy as np
from fmri_methods_sipi import rot_sub_data, reorder_labels
from dfsio import readdfs, writedfs
#import h5py
import os
from surfproc import view_patch, view_patch_vtk, get_cmap, patch_color_labels
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.metrics import adjusted_rand_score


def mean_std_rand(labels_all):
    # labels_all is nvert x nsub matrix
    # delete subjects for which parcellation is not done
    labs1 = labels_all
    ind = (sp.sum(labs1, axis=0) != 0)
    labs1 = labs1[:, ind]

    labs = reorder_labels(labs1)

    labs_mode, freq = sp.stats.mode(labs, axis=1)
    freq1 = sp.double(freq.squeeze())
    freq1 /= labs.shape[1]

    ars = sp.zeros(labs.shape[1])
    for ind in range(labs.shape[1]):
        ars[ind] = adjusted_rand_score(labs_mode.squeeze(), labs[:, ind])

    return ars.mean(), ars.std(), freq1, labs_mode


p_dir_ref = '/home/ajoshi/data/HCP_data'
ref = '100307'
dfs_left_sm = readdfs(os.path.join(p_dir_ref, 'reference',
                                   ref + '.aparc.a2009s.32k_fs.reduce3.\
very_smooth.left.dfs'))


#rlist = [21]  # Precuneus
#msksize = [507]

#rlist = [10]
#msksize = [545]
rlist = [13]
msksize = [577]

# right_hemisphere=np.array([226,168,184,446,330,164,442,328,172,444,130,424,166,326,342,142,146,144,222,170,
# 150,242,186,120,422,228,224,322,310,162,324,500])

left_hemisphere=np.array([227,169,185,447,331,165,443,329,173,445,131,425,167,327,343,143,147,145,223,171,
151,243,187,121,423,229,225,323,311,163,325,501])

nClusters = np.array([3,1,3,2,2,2,3,3,2,2,2,3,1,4,1,2,1,3,2,1,4,2,1,2,2,2,2,3,1,2,1,2])
# right_hemisphere = right_hemisphere[rlist]
left_hemisphere = left_hemisphere[rlist]
nClusters = nClusters[rlist]

out_dir = '/big_disk/ajoshi/out_dir'
p_dir = '/big_disk/ajoshi/HCP100-fMRI-NLM/HCP100-fMRI-NLM'
lst = os.listdir(p_dir)
old_lst = []  # os.listdir('/home/ajoshi/data/HCP_data/data')
old_lst += ['reference', 'zip1']  # , '106016', '366446']
save_dir = '/big_disk/ajoshi/fmri_validation'

sdir = ['_RL', '_LR']
scan_type = ['left', 'right']
session_type = [1, 2]
fadd_1 = '.rfMRI_REST'
fadd_2 = '.reduce3.ftdata.NLM_11N_hvar_25.mat'

# %% Across session study
labels_corr_sininv_all = sp.zeros((len(lst), 4, msksize[0]))
labels_corr_corr_exp_all = sp.zeros((len(lst), 4, msksize[0]))
labels_corr_dist_all = sp.zeros((len(lst), 4, msksize[0]))
labels_corr_exp_all = sp.zeros((len(lst), 4, msksize[0]))
subno = 0
for sub in lst:
    for scan in range(0, 4):
        if (sub not in old_lst) and \
            (os.path.isfile(os.path.join(p_dir, sub, sub + fadd_1 +
             str(session_type[scan % 2]) + sdir[scan / 2] + fadd_2))):
            count1 = 0
            all_centroid = []
            centroid = []
            label_count = 0
            roiregion = left_hemisphere[0]
            session = session_type[scan % 2]
            scan1 = sdir[scan/2]
            l = sp.load(os.path.join(out_dir, sub + '.rfMRI_REST' +
                        str(session) + scan1 + str(roiregion) + '.labs.npz'))

            labels_corr_sininv_all[subno, scan, :] = l['labels_corr_sininv']
            labels_corr_corr_exp_all[subno, scan, :] = \
                l['labels_corr_corr_exp']
            labels_corr_dist_all[subno, scan, :] = l['labels_corr_dist']
            labels_corr_exp_all[subno, scan, :] = l['labels_corr_exp']
    subno += 1

ind = (sp.sum(labels_corr_sininv_all, axis=(1, 2)) != 0)
labels_corr_sininv_all = labels_corr_sininv_all[ind, :, :]
labels_corr_corr_exp_all = labels_corr_corr_exp_all[ind, :, :]
labels_corr_dist_all = labels_corr_dist_all[ind, :, :]
labels_corr_exp_all = labels_corr_exp_all[ind, :, :]

labels = sp.zeros([10832])
ind = l['msk_small_region']
freq = sp.zeros([10832])
mean_rnd = sp.zeros((4, 4))
std_rnd = sp.zeros((4, 4))

for scanid in range(4):
    # labels_corr_sininv_all is nsub x nsession x nvert
    m, s, freq1, labs_mode = mean_std_rand(labels_corr_sininv_all
                                           [:, scanid, :].T)
    mean_rnd[scanid, 0] = m
    std_rnd[scanid, 0] = s
    print('scanid = ' + str(scanid) + ' roi = ' + str(roiregion) +
          ' labels_corr_sininv_all mean =' + str(m) + ' std = ' + str(s))
    labels[ind] = labs_mode.squeeze()

    freq[ind] = freq1

    dfs_left_sm.labels = labels
#    dfs_left_sm = patch_color_labels(dfs_left_sm, freq=freq, cmap='jet')
#    view_patch_vtk(dfs_left_sm)

    # labels_corr_corr_exp_all is nsub x nsession x nvert
    m, s, freq1, labs_mode = mean_std_rand(labels_corr_corr_exp_all[:, 0, :].T)
    mean_rnd[scanid, 1] = m
    std_rnd[scanid, 1] = s
    print('scanid = ' + str(scanid) + ' roi = ' + str(roiregion) +
          ' labels_corr_corr_exp_all mean =' + str(m) + ' std = ' + str(s))
    labels[ind] = labs_mode.squeeze()

    freq[ind] = freq1

    dfs_left_sm.labels = labels
#    dfs_left_sm = patch_color_labels(dfs_left_sm, freq=freq, cmap='jet')
#    view_patch_vtk(dfs_left_sm)

    # labels_corr_dist_all is nsub x nsession x nvert
    m, s, freq1, labs_mode = mean_std_rand(labels_corr_dist_all[:, 0, :].T)
    mean_rnd[scanid, 2] = m
    std_rnd[scanid, 2] = s
    print('scanid = ' + str(scanid) + ' roi = ' + str(roiregion) +
          ' labels_corr_dist_all mean =' + str(m) + ' std = ' + str(s))
    labels[ind] = labs_mode.squeeze()

    freq[ind] = freq1

    dfs_left_sm.labels = labels
#    dfs_left_sm = patch_color_labels(dfs_left_sm, freq=freq, cmap='jet')
#    view_patch_vtk(dfs_left_sm)

    # labels_corr_corr_exp_all is nsub x nsession x nvert
    m, s, freq1, labs_mode = mean_std_rand(labels_corr_exp_all[:, 0, :].T)
    mean_rnd[scanid, 3] = m
    std_rnd[scanid, 3] = s
    print('scanid = ' + str(scanid) + ' roi = ' + str(roiregion) +
          ' labels_corr_exp_all mean =' + str(m) + ' std = ' + str(s))

    labels[ind] = labs_mode.squeeze()

    freq[ind] = freq1

    dfs_left_sm.labels = labels
#    dfs_left_sm = patch_color_labels(dfs_left_sm, freq=freq, cmap='jet')
#    view_patch_vtk(dfs_left_sm)

print 'ACROSS SUBJECT COMPARISON of RAND SCORES'
print 'corr_sininv mean :' +\
    str(sp.mean(mean_rnd[:, 0])) + 'std:' + str(sp.mean(std_rnd[:, 0]))
print 'corr_corr_exp mean :' +\
    str(sp.mean(mean_rnd[:, 1])) + 'std:' + str(sp.mean(std_rnd[:, 1]))
print 'corr_dist mean :' +\
    str(sp.mean(mean_rnd[:, 2])) + 'std:' + str(sp.mean(std_rnd[:, 2]))
print 'corr_exp mean :' +\
    str(sp.mean(mean_rnd[:, 3])) + 'std:' + str(sp.mean(std_rnd[:, 3]))

# %% Across session comparison
nsub = labels_corr_sininv_all.shape[0]
mean_rnd = sp.zeros((nsub, 4))
std_rnd = sp.zeros((nsub, 4))

for subno in range(nsub):
    # labels_corr_sininv_all is nsub x nsession x nvert
    m, s, freq1, labs_mode = mean_std_rand(labels_corr_sininv_all
                                           [subno, :, :].T)
    mean_rnd[subno, 0] = m
    std_rnd[subno, 0] = s
    print('subno = ' + str(subno) + ' roi = ' + str(roiregion) +
          ' labels_corr_sininv_all mean =' + str(m) + ' std = ' + str(s))
    labels[ind] = labs_mode.squeeze()

    freq[ind] = freq1

#    dfs_left_sm.labels = labels
#    dfs_left_sm = patch_color_labels(dfs_left_sm, freq=freq, cmap='jet')
#    view_patch_vtk(dfs_left_sm)

    # labels_corr_corr_exp_all is nsub x nsession x nvert
    m, s, freq1, labs_mode = mean_std_rand(labels_corr_corr_exp_all
                                           [subno, :, :].T)
    mean_rnd[subno, 1] = m
    std_rnd[subno, 1] = s
    print('subno = ' + str(subno) + ' roi = ' + str(roiregion) +
          ' labels_corr_corr_exp_all mean =' + str(m) + ' std = ' + str(s))
    labels[ind] = labs_mode.squeeze()

    freq[ind] = freq1

#    dfs_left_sm.labels = labels
#    dfs_left_sm = patch_color_labels(dfs_left_sm, freq=freq, cmap='jet')
#    view_patch_vtk(dfs_left_sm)

    # labels_corr_dist_all is nsub x nsession x nvert
    m, s, freq1, labs_mode = mean_std_rand(labels_corr_dist_all[subno, :, :].T)
    mean_rnd[subno, 2] = m
    std_rnd[subno, 2] = s
    print('subno = ' + str(subno) + ' roi = ' + str(roiregion) +
          ' labels_corr_dist_all mean =' + str(m) + ' std = ' + str(s))
    labels[ind] = labs_mode.squeeze()

    freq[ind] = freq1

#    dfs_left_sm.labels = labels
#    dfs_left_sm = patch_color_labels(dfs_left_sm, freq=freq, cmap='jet')
#    view_patch_vtk(dfs_left_sm)

    # labels_corr_corr_exp_all is nsub x nsession x nvert
    m, s, freq1, labs_mode = mean_std_rand(labels_corr_exp_all[subno, :, :].T)
    mean_rnd[subno, 3] = m
    std_rnd[subno, 3] = s
    print('subno = ' + str(subno) + ' roi = ' + str(roiregion) +
          ' labels_corr_exp_all mean =' + str(m) + ' std = ' + str(s))

    labels[ind] = labs_mode.squeeze()

    freq[ind] = freq1

#    dfs_left_sm.labels = labels
#    dfs_left_sm = patch_color_labels(dfs_left_sm, freq=freq, cmap='jet')
#    view_patch_vtk(dfs_left_sm)

print 'ACROSS SCANS COMPARISON of RAND SCORES'
print 'corr_sininv mean :' +\
    str(sp.mean(mean_rnd[:, 0])) + 'std:' + str(sp.mean(std_rnd[:, 0]))
print 'corr_corr_exp mean :' +\
    str(sp.mean(mean_rnd[:, 1])) + 'std:' + str(sp.mean(std_rnd[:, 1]))
print 'corr_dist mean :' +\
    str(sp.mean(mean_rnd[:, 2])) + 'std:' + str(sp.mean(std_rnd[:, 2]))
print 'corr_exp mean :' +\
    str(sp.mean(mean_rnd[:, 3])) + 'std:' + str(sp.mean(std_rnd[:, 3]))
