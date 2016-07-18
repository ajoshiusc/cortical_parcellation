from mayavi import mlab
from centroid import merge, choose_best, replot, avgplot, spatial_map

import scipy as sp
import numpy as np
from scipy import io

nSubjects = 40
roiregion=['motor','precuneus','temporal','cingulate','semato','visual']

roilist = np.array([[29,69,70],[(30, 72, 9, 47)],[33,34,35,36,74],[6,7,8,9,10],[28],[(2,22,11,58,59,20,43,19,45)]])

nCluster=np.array([3,3,7,3,2,4])
for n in range(5,6):
    data_file='data_file'+roiregion[n] +str(0) +'overall.npz'
    labs_all_1 = []
    vert_all_1 = np.array([])
    faces = np.array([])
    all_centroid=sp.array([])
    correlation_within_precuneus=np.load(data_file)['correlation_within_precuneus']
    correlation_with_rest=np.load(data_file)['correlation_with_rest']
    labs_all_1=np.load(data_file)['labels']
    vertices=np.load(data_file)['vertices']
    faces=np.load(data_file)['faces']
    mask=np.load(data_file)['mask']
    all_centroid=np.load(data_file)['centroid']
    labs_all_2=np.array( [[0 for x in range(labs_all_1.shape[1])] for y in range(0,nSubjects)] ,dtype=float)
    
    #print all_centroid
    for j in range(0,1):
        for i in range(0, nSubjects):
            label_matrix = choose_best(correlation_within_precuneus[i * nCluster[n]:i * nCluster[n] + nCluster[n]], correlation_within_precuneus[j*nCluster[n]:j*nCluster[n]+nCluster[n]],nCluster[n])
            c_all_subjects=np.array(correlation_within_precuneus[i*nCluster[n]:i*nCluster[n]+nCluster[n]])
            c_all_subjects_rest = np.array(correlation_with_rest[i * nCluster[n]:i * nCluster[n] + nCluster[n]])
            for k in range(0,nCluster[n]):
                correlation_within_precuneus[i * nCluster[n] + k] = c_all_subjects[label_matrix[k]]
                correlation_with_rest[i * nCluster[n] + k] = c_all_subjects_rest[label_matrix[k]]
            labs_all_2[i] = replot(labs_all_1[i], vertices, faces, label_matrix, labs_all_1[0],nCluster[n])
    
    
    #sp.savez('data_file.npz',corr_vec=all_subjects,labels=labs_all_2,vertices=vertices,faces=faces,mask=mask)
        sp.io.savemat('intensity_file_'+roiregion[n]+'.mat',dict(correlation_within_motor=correlation_within_precuneus,correlation_with_rest=correlation_with_rest,labels=labs_all_2,vertices=vertices,faces=faces,mask=mask))
    
        vector= np.array( [[0 for x in range(correlation_with_rest.shape[1])] for y in range(0,nCluster[n])] ,dtype=float)
        val = np.array([[0 for x in range(correlation_within_precuneus.shape[1])] for y in range(0, nCluster[n])], dtype=float)
    
        for i in range(nSubjects*nCluster[n]):
            vector[i % nCluster[n]] = np.add(vector[i % nCluster[n]], correlation_with_rest[i])
            val[i % nCluster[n]] = np.add(val[i%nCluster[n]],correlation_within_precuneus[i])
        val=val/nSubjects
        vector=vector/nSubjects
    
    
        labs=avgplot(labs_all_2.transpose(), nSubjects, vertices, faces,nCluster[n])
    
        for i in range(nCluster[n]):
            spatial_map(vector[i], vertices, faces, mask, val[i], i + 1)
print