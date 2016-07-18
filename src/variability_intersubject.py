from centroid import rand_indices_within_subjects, rand_indices_across_subjects, choose_best, replot, avgplot, \
    intersubjects, intrasubjects, display_intrasubjects
import scipy as sp
import numpy as np

session_num=4
nSubjects = 40
for nCluster in [3]:
    data_file='data_file'+str(nCluster)
    all_subjects = np.array([])
    labs_all = np.array([])
    all_centroid = np.array([])
    all_temp=np.array([])
    for i in range(session_num):
        if i == 0:
            all_subjects=sp.array([np.load(data_file+str(i)+'precuneus_sine.npz')['correlation_within_precuneus']])
            labs_all=sp.array([np.load(data_file+str(i)+'precuneus_sine.npz')['labels']])
            all_centroid = sp.array([np.load(data_file + str(i) + 'precuneus_sine.npz')['centroid']])
            #all_temp=sp.array([np.load(data_file + str(i) + 'logistic_labels.npz')['labels']])
        else:
            all_subjects = sp.vstack([all_subjects,[np.load(data_file + str(i) + 'precuneus_sine.npz')['correlation_within_precuneus']]])
            labs_all = sp.vstack([labs_all,[np.load(data_file + str(i) + 'precuneus_sine.npz')['labels']]])
            all_centroid = sp.vstack([all_centroid,[np.load(data_file + str(i) + 'precuneus_sine.npz')['centroid']]])
            #all_temp = sp.vstack([all_temp, [np.load(data_file + str(i) + 'logistic_labels.npz')['labels']]])

    vertices=np.load(data_file+str(0)+'precuneus_sine.npz')['vertices']
    faces=np.load(data_file+str(0)+'precuneus_sine.npz')['faces']
    mask=np.load(data_file+str(0)+'precuneus_sine.npz')['mask']

    #print all_centroid
    sum=0
    for j in [0]:
        for k in range(session_num):
            for i in range(0, nSubjects):
                label_matrix = choose_best(all_subjects[k][i * nCluster:i * nCluster + nCluster], all_subjects[k][j*nCluster:j*nCluster+nCluster],nCluster)
                c_all_subjects=np.array(all_subjects[k][i*nCluster:i*nCluster+nCluster])
                for l in range(0,nCluster):
                    all_subjects[k][i * nCluster + l] = c_all_subjects[label_matrix[l]]
                labs_all[k][i] = replot(labs_all[k][i], vertices, faces, label_matrix, labs_all[k][0],nCluster)

        rand_indices_within_sub=rand_indices_within_subjects(session_num,labs_all,nSubjects,mask)
        rand_indices_across_sub = rand_indices_across_subjects(session_num, labs_all, nSubjects,mask)
        sum = rand_indices_within_sub/rand_indices_across_sub

    print rand_indices_within_sub,rand_indices_across_sub,sum
