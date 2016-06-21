# -*- coding: utf-8 -*-
"""
Created on Tue Mar 01 16:24:42 2016

@author: ajoshi
"""
from centroid import merge, choose_best, replot, avgplot
from fmri_parcellation import parcellate_region
import os
import scipy as sp
import numpy as np
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

from separate_cluster import separate

p_dir = '/home/ajoshi/HCP_data'
lst = os.listdir(p_dir) #{'100307'}
#%%
mean_R = sp.zeros(6)
std_R = sp.zeros(6)

#%% Across session study
R_all = []
for nClusters in [3]: #range(6):
    labs_all_1 = []
    labs_all_2 = []
    vert_all_1 = []
    vert_all_2 = []
    faces_all_1 = []
    faces_all_2 = []
    all_subjects=sp.array([])
    count1 = 0
    count_break=0
    session=[]
    for sub in lst:
        count_break +=1
        print count_break
        if os.path.isfile(os.path.join(p_dir, sub, sub +
                                       '.rfMRI_REST1_RL.reduce3.ftdata.NLM_11N\
_hvar_25.mat')):

            # (46,28,29) motor 243 is precuneus
            labs1 , session ,mask = parcellate_region((30,72,9,47), sub, nClusters, 1, 1, 0)
            count1 += 1
            if count1 == 1:
                labs_all_1 = sp.array(labs1.labels)
                vert_all_1 = sp.array(labs1.vertices)
                faces_all_1 = sp.array(labs1.faces)
                all_subjects=sp.array(session)
            else:
                labs_all_1 = sp.vstack([labs_all_1, labs1.labels])
                vert_all_1 = sp.array([labs1.vertices])
                faces_all_1 = sp.array([labs1.faces])
                all_subjects=sp.vstack([all_subjects,session])

#sp.savez_compressed('clustering_results_sessions_region_pc', R_all=R_all)
sp.savez('data_file.npz',corr_vec=all_subjects,labels=labs_all_1,vertices=labs1.vertices,faces=labs1.faces,mask=mask)


#%%
'''fig = plt.figure()
plt.boxplot(R_all)
fig.savefig('across_subjects_adj_rand_sessions_region_pc.pdf')'''


#%%