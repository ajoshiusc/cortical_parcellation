from matplotlib import cm
from mayavi import mlab
from centroid import merge, choose_best, replot, avgplot, spatial_map, change_corr_vector, num_of_cluster
from fmri_parcellation import parcellate_region
import os
import scipy as sp
import numpy as np

nSubjects = 40
nCluster=3
labs_all_1 = []
vert_all_1 = np.array([])
faces = np.array([])
all_centroid=sp.array([])
correlation_within_precuneus=np.load('data_filecingulate.npz')['correlation_within_precuneus']
correlation_with_rest=np.load('data_filecingulate.npz')['correlation_with_rest']
labs_all_1=np.load('data_filecingulate.npz')['labels']
vertices=np.load('data_filecingulate.npz')['vertices']
faces=np.load('data_filecingulate.npz')['faces']
mask=np.load('data_filecingulate.npz')['mask']
all_centroid=np.load('data_filecingulate.npz')['centroid']
labs_all_2=np.array( [[0 for x in range(labs_all_1.shape[1])] for y in range(0,nSubjects)] ,dtype=float)

#print all_centroid
for j in range(0,nSubjects):
    for i in range(0, nSubjects):
        label_matrix = choose_best(correlation_within_precuneus[i * nCluster:i * nCluster + nCluster], correlation_within_precuneus[j*nCluster:j*nCluster+nCluster],nCluster)
        c_all_subjects=np.array(correlation_within_precuneus[i*nCluster:i*nCluster+nCluster])
        for k in range(0,nCluster):
            correlation_within_precuneus[i * nCluster + k] = correlation_within_precuneus[label_matrix[k]]
        labs_all_2[i] = replot(labs_all_1[i], vertices, faces, label_matrix, labs_all_1[0],nCluster)


#sp.savez('data_file.npz',corr_vec=all_subjects,labels=labs_all_2,vertices=vertices,faces=faces,mask=mask)
    #sp.io.savemat('final_result.mat',dict(correlation_within_motor=correlation_within_precuneus,correlation_with_rest=correlation_with_rest,labels=labs_all_2,vertices=vertices,faces=faces,mask=mask))

    vector= np.array( [[0 for x in range(correlation_with_rest.shape[1])] for y in range(0,nCluster)] ,dtype=float)

    for i in range(nSubjects*nCluster):
        vector[i % nCluster] = np.add(vector[i % nCluster], correlation_with_rest[i])

    vector=vector/nSubjects

    for i in range(nCluster):
        mlab.points3d(all_centroid[i+4][0], all_centroid[i+4][1], all_centroid[i+4][2])
        spatial_map(vector[i],vertices,faces,mask)


    avgplot(labs_all_2.transpose(), nSubjects, vertices, faces,nCluster)
