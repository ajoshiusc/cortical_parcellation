# -*- coding: utf-8 -*-
"""
Created on Tue Mar 01 16:24:42 2016

@author: ajoshi
"""

from fmri_parcellation import parcellate_region
import os
import scipy as sp
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt


p_dir = 'E:\\HCP-fMRI-NLM'
lst = os.listdir(p_dir) #{'100307'}
#%%
mean_R = sp.zeros(6)
std_R = sp.zeros(6)

#%% Across session study
R_all = []
for nClusters in range(6):
    labs_all_1 = []
    labs_all_2 = []
    count1 = 0
    for sub in lst:
        if os.path.isfile(os.path.join(p_dir, sub, sub +
                                       '.rfMRI_REST1_RL.reduce3.ftdata.NLM_11N\
_hvar_25.mat')):

            # (46,28,29) motor 243 is precuneus
            labs1 = parcellate_region((30,72,9,47), sub, nClusters+1, 1, 1, 0)
        if os.path.isfile(os.path.join(p_dir, sub, sub +
                                       '.rfMRI_REST2_RL.reduce3.ftdata.NLM_11N\
_hvar_25.mat')):
            labs2 = parcellate_region((30,72,9,47), sub, nClusters+1, 1, 2, 0)
            count1 += 1
            if count1 == 1:
                labs_all_1 = sp.array(labs1)
                labs_all_2 = sp.array(labs2)
            else:
                labs_all_1 = sp.vstack([labs_all_1, labs1])
                labs_all_2 = sp.vstack([labs_all_2, labs2])

    R = sp.zeros(count1)
    for a in range(count1):
        R[a] = adjusted_rand_score(labs_all_1[a,:], labs_all_2[a,:])

    R_all.append(R)
    print('Clusters=', nClusters)

sp.savez_compressed('clustering_results_sessions_region_pc', R_all=R_all)

#%%
fig = plt.figure()
plt.boxplot(R_all)
fig.savefig('across_subjects_adj_rand_sessions_region_pc.pdf')


#%%