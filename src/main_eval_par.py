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
for nClusters in [2]: #range(6):
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
        count_break+=1
        if os.path.isfile(os.path.join(p_dir, sub, sub +
                                       '.rfMRI_REST1_RL.reduce3.ftdata.NLM_11N\
_hvar_25.mat')):

            # (46,28,29) motor 243 is precuneus
            labs1 , session1  = parcellate_region((30,72,9,47), sub, nClusters, 1, 1, 0)
        if os.path.isfile(os.path.join(p_dir, sub, sub +
                                       '.rfMRI_REST2_RL.reduce3.ftdata.NLM_11N\
_hvar_25.mat')):
            labs2, session2 = parcellate_region((30,72,9,47), sub, nClusters, 1, 2, 0)
            session=merge(session1,session2)
            count1 += 1
            if count1 == 1:
                labs_all_1 = sp.array(labs1.labels)
                labs_all_2 = sp.array(labs2.labels)
                vert_all_1 = sp.array(labs1.vertices)
                vert_all_2 = sp.array(labs2.vertices)
                faces_all_1 = sp.array(labs1.faces)
                faces_all_2 = sp.array(labs2.faces)
                all_subjects=sp.array(session)
            else:
                labs_all_1 = sp.vstack([labs_all_1, labs1.labels])
                labs_all_2 = sp.vstack([labs_all_2, labs2.labels])
                vert_all_1 = sp.array([labs1.vertices])
                vert_all_2 = sp.array([labs2.vertices])
                faces_all_1 = sp.array([labs1.faces])
                faces_all_2 = sp.array([labs2.faces])
                all_subjects=sp.vstack([all_subjects,session])






    for i in range(0,all_subjects.shape[0]/4):
        label_matrix=choose_best(all_subjects[i*4:i*4+2],all_subjects[0:2])
        labs_all_1[i]=replot(labs_all_1[i],labs2.vertices,labs2.faces,label_matrix,labs_all_1[0])





    avgplot(labs_all_1.transpose(),all_subjects.shape[0]/4,labs2.vertices,labs2.faces)


    R = sp.zeros(count1)
    for a in range(count1):
        R[a] = adjusted_rand_score(labs_all_1.labels[a,:], labs_all_2[a,:])

    R_all.append(R)
    print('Clusters=', nClusters)


sp.savez_compressed('clustering_results_sessions_region_pc', R_all=R_all)

#%%
fig = plt.figure()
plt.boxplot(R_all)
fig.savefig('across_subjects_adj_rand_sessions_region_pc.pdf')


#%%