# -*- coding: utf-8 -*-
"""
Created on Tue Mar 01 16:24:42 2016

@author: ajoshi
"""

from fmri_parcellation import parcellate_motor
import os
import scipy as sp
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

p_dir = 'E:\\HCP-fMRI-NLM'
lst = os.listdir(p_dir)

mean_R = sp.zeros(30)
std_R = sp.zeros(30)
R_all = []
for nClusters in range(30):
    labs_all = []
    count1 = 0
    for sub in lst:
        if os.path.isfile(os.path.join(p_dir, sub, sub +
                                       '.rfMRI_REST1_RL.reduce3.ftdata.NLM_11N\
_hvar_25.mat')):
            labs = parcellate_motor(sub, nClusters+1)
            count1 += 1
            if count1 == 1:
                labs_all = sp.array(labs)
            else:
                labs_all = sp.vstack([labs_all, labs])
    R = sp.zeros(((labs_all.shape[0]-1)*labs_all.shape[0])/2)
    cnt = 0
    for a in range(labs_all.shape[0]):
        for b in range(a):
            R[cnt] = adjusted_rand_score(labs_all[a, :], labs_all[b, :])
            cnt += 1
    R_all.append(R)
    mean_R = sp.mean(R)
    std_R = sp.std(R)
    print('Clusters=', nClusters)

sp.savez_compressed('clustering_results', R_all=R_all, mean_R=mean_R,
                    std_R=std_R)
fig = plt.figure()
plt.boxplot(R_all.T)
fig.savefig('across_subjects_adj_rand_score.pdf')
