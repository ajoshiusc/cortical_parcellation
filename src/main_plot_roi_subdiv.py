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
from surfproc import view_patch, view_patch_vtk, get_cmap
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.metrics import silhouette_score
out_dir= '/big_disk/ajoshi/out_dir'
p_dir = '/big_disk/ajoshi/HCP100-fMRI-NLM/HCP100-fMRI-NLM'
lst = os.listdir(p_dir) #{'100307'}
old_lst = os.listdir('/home/ajoshi/data/HCP_data/data')
old_lst+=['reference','zip1','106016','366446']
save_dir= '/big_disk/ajoshi/fmri_validation'

sdir=['_RL','_LR']
scan_type=['left','right']
session_type=[1,2]
fadd_1='.rfMRI_REST'
fadd_2='.reduce3.ftdata.NLM_11N_hvar_25.mat'

#%%Across session study
for sub in lst:
    for scan in range(0,4):
        if (sub not in old_lst) and (os.path.isfile(os.path.join(p_dir, sub, sub + fadd_1 + str(session_type[scan%2]) + sdir[scan/2] + fadd_2))):
            for i in range(0,2):
                labs_all  = np.zeros([10832])
                count1 = 0
                all_centroid = []
                centroid = []
                label_count = 0
                for n in range(nClusters.shape[0]):
                    print n
                    roiregion = left_hemisphere[n]
                    if i == 1:
                        roiregion = right_hemisphere[n]
                        
                    l=sp.load(os.path.join(out_dir, sub + '.rfMRI_REST' + str(session) +
                          scan + str(roilist) + '.labs.npz'))

