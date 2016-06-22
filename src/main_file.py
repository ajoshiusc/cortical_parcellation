from centroid import merge, choose_best, replot, avgplot, spatial_map, change_corr_vector
from fmri_parcellation import parcellate_region
import os
import scipy as sp
import numpy as np
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

from separate_cluster import separate
nSubjects = 40
nCluster=3
labs_all_1 = []
vert_all_1 = np.array([])
faces = np.array([])
all_subjects=sp.array([])
all_centroid=sp.array([])
all_subjects=np.load('data_file.npz')['corr_vec']
labs_all_1=np.load('data_file.npz')['labels']
vertices=np.load('data_file.npz')['vertices']
faces=np.load('data_file.npz')['faces']
mask=np.load('data_file.npz')['mask']
all_centroid=np.load('data_file.npz')['centroid']
labs_all_2=np.array( [[0 for x in range(labs_all_1.shape[1])] for y in range(0,nSubjects)] ,dtype=float)

#print all_centroid
for j in range(0,nSubjects):
    for i in range(0, nSubjects):
        label_matrix = choose_best(all_subjects[i * nCluster:i * nCluster + nCluster], all_subjects[j*nCluster:j*nCluster+nCluster])
        c_all_subjects=np.array(all_subjects[i*nCluster:i*nCluster+nCluster])
        all_subjects[i * nCluster] = c_all_subjects[label_matrix[0]]
        all_subjects[(i * nCluster)+1] = c_all_subjects[label_matrix[1]]
        all_subjects[(i * nCluster)+2] = c_all_subjects[label_matrix[2]]
        labs_all_2[i] = replot(labs_all_1[i], vertices, faces, label_matrix, labs_all_1[0],all_centroid[i*nCluster:i*nCluster+nCluster])


#sp.savez('data_file.npz',corr_vec=all_subjects,labels=labs_all_2,vertices=vertices,faces=faces,mask=mask)
    #sp.io.savemat('transfer.mat',dict(corr_vec=all_subjects,labels=labs_all_2,vertices=vertices,faces=faces,mask=mask))

    store= labs_all_2==labs_all_1
    vector= np.array( [[0 for x in range(all_subjects.shape[1])] for y in range(0,nCluster)] ,dtype=float)

    for i in range(nSubjects*nCluster):
        if i%nCluster == 0:
            #print np.add(vector[i%3],all_subjects[i])
            vector[i % nCluster]=np.add(vector[i%nCluster],all_subjects[i])
        elif i % nCluster == 1:
            vector[i % nCluster] = np.add(vector[i % nCluster], all_subjects[i])
        elif i % nCluster == 2:
            vector[i % nCluster] = np.add(vector[i % nCluster], all_subjects[i])
        elif i % nCluster == 3:
            vector[i % nCluster] = np.add(vector[i % nCluster], all_subjects[i])

    vector=vector/nSubjects

    #for i in range(nCluster):
        #spatial_map(vector[i],vertices,faces,mask)


    avgplot(labs_all_1.transpose(), nSubjects, vertices, faces,nCluster)
