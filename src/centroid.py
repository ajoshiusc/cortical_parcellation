import sys
import itertools
import numpy as np
from mayavi import mlab
import scipy as sp
from scipy import stats
def modified_find_centroid(c):
    sx = sy = sL = sz = 0
    index=-1
    min=np.Inf
    for i in range(c.shape[0]):  # counts from 0 to len(points)-1
       x0, y0, z0 = c[i]  # in Python points[-1] is last element of points
       sL=0
       for j in range(c.shape[0]):
           x1, y1, z1 = c[j]
           L = ((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2) ** 0.5
           sL += L
       if min>sL :
           min=sL
           index=i
    return c[index]

def search(vertices,centroid):
    for i in range(vertices.shape[0]):
        if vertices[i][0] == centroid[0]:
            if vertices[i][1] == centroid[1]:
                if vertices[i][2] == centroid[2]:
                    return i
    return -1

def find_location_smallmask(vertices,centroid,mask):
    index = search(vertices, centroid)
    #remember to reconsider here about the range
    count=0
    for i in range(index):
        if mask[i]==True:
            count+=1;
    return count

def merge(s1,s2):
    s=[]
    s=sp.array(s1)
    s=sp.vstack([s,s2])
    return s

def choose_best(subject,reference_subject):
    #print subject,reference_subject
    min=np.Inf
    save=np.array([])
    #change
    for j in list(itertools.permutations([0, 1,2], 3)):
        sum=0
        #change
        for k in range(0, 3):
            sum += np.sum(np.abs(subject[k]-reference_subject[j[k]])**2)
            #sum += np.sum(subject[k] - reference_subject[j[k]])
            #print  sum, j
        if min > sum:
            min = sum
            save = np.array(j)
    #print save
    return save

def replot(r_labels,r_vertices,r_faces,label_matrix,reference_label,centroid):
    for i in range(r_labels.shape[0]):
        if r_labels[i] == 1:
            r_labels[i]=label_matrix[0]+1
        elif r_labels[i] == 2:
            r_labels[i] = label_matrix[1]+1
        elif r_labels[i] == 3:
            r_labels[i] = label_matrix[2]+1
        elif r_labels[i] == 4:
            r_labels[i] = label_matrix[3] + 1
    return r_labels

def avgplot(r_labels,nSubjects,r_vertices,r_faces,nCluster):
    labels = np.zeros(r_labels.shape[0],dtype=float)
    for i in range(r_labels.shape[0]):
        labels[i] = np.sum(r_labels[i])/nSubjects
    mlab.triangular_mesh(r_vertices[:, 0], r_vertices[:, 1], r_vertices[:, 2], r_faces, representation='surface',
                         opacity=1, scalars=np.float64(labels))
    mlab.gcf().scene.parallel_projection = True
    mlab.view(azimuth=0, elevation=90)
    mlab.colorbar(orientation='horizontal')
    mlab.close()
    labels = np.zeros(r_labels.shape[0], dtype=float)
    for i in range(r_labels.shape[0]):
        mode,count=stats.mode(r_labels[i])
        labels[i] = mode[0]
    mlab.triangular_mesh(r_vertices[:, 0], r_vertices[:, 1], r_vertices[:, 2], r_faces, representation='surface',
                             opacity=1, scalars=np.float64(labels))
    mlab.gcf().scene.parallel_projection = True
    mlab.view(azimuth=0, elevation=90)
    mlab.colorbar(orientation='horizontal')
    mlab.close()

def spatial_map(vector,r_vertices,r_faces,msk_small_region):
    r_labels = np.zeros([r_vertices.shape[0]])
    r_labels[~msk_small_region] = vector
    mlab.triangular_mesh(r_vertices[:, 0], r_vertices[:, 1], r_vertices[:,
                                                             2], r_faces, representation='surface',
                         opacity=1, scalars=np.float64(r_labels))
    #mlab.points3d(cent[0], cent[1], cent[2])
    mlab.gcf().scene.parallel_projection = True
    mlab.view(azimuth=0, elevation=90)
    mlab.draw()
    #mlab.show()
    mlab.colorbar(orientation='horizontal')
    mlab.close()




def change_labels(labels,order):
    print labels
    for i in range(labels.shape[0]):
        if labels[i] == 1:
            labels[i] = order[0]+1
        elif labels[i] == 2:
            labels[i] = order[1] + 1
        elif labels[i] == 3:
            labels[i] = order[2] + 1
        elif labels[i] == 4:
            labels[i] = order[3] + 1
    return labels


def change_order(order,nCluster):
    save=np.array([0,0,0,0])
    for i in range(0,nCluster):
        save[order[i]]=i
    return save

def change_corr_vector(subject,label_matrix):
    return subject[label_matrix[0]],subject[label_matrix[1]],subject[label_matrix[2]]