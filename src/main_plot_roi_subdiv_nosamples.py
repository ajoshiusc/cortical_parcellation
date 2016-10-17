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
import seaborn as sns
import matplotlib.pyplot as plt


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

l = sp.load('temp100.npz')

labels_corr_sininv_all = l['labels_corr_sininv_all']
labels_corr_corr_exp_all = l['labels_corr_corr_exp_all']
labels_corr_dist_all = l['labels_corr_dist_all']
labels_corr_exp_all = l['labels_corr_exp_all']


rnd_ind = sp.zeros((labels_corr_sininv_all.shape[0],
                    labels_corr_sininv_all.shape[1],4))

for ind in range(labels_corr_sininv_all.shape[0]):
    for nosamples in range(labels_corr_sininv_all.shape[1]):
        # labels_corr_sininv_all is nsub x nsession x nvert
        rnd_ind[ind, nosamples, 0] = adjusted_rand_score(labels_corr_sininv_all[ind, -1 , :],
                                 labels_corr_sininv_all[ind, nosamples, :])
        rnd_ind[ind, nosamples, 1] = adjusted_rand_score(labels_corr_corr_exp_all[ind, -1 , :],
                                 labels_corr_corr_exp_all[ind, nosamples, :])
        rnd_ind[ind, nosamples, 2] = adjusted_rand_score(labels_corr_dist_all[ind, -1 , :],
                                 labels_corr_dist_all[ind, nosamples, :])
        rnd_ind[ind, nosamples,3] = adjusted_rand_score(labels_corr_exp_all[ind, -1 , :],
                                 labels_corr_exp_all[ind, nosamples, :])

#rnd_ind_mean = sp.stats.trim_mean(rnd_ind, axis=0)



sns_plot = sns.tsplot(data=rnd_ind,  value = "adj rand score", condition = ['$\sin^{-1}$','conn','$\exp$', '$L^2$'], ci = 95, err_style='ci_band')

plt.savefig('perf_samples.pdf')

