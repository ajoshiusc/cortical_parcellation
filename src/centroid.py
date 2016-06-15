import sys
import itertools
import numpy as np
from mayavi import mlab
import scipy as sp
from scipy import stats
def modified_find_centroid(c):
    sx = sy = sL = sz = 0
    index=-1
    min=100000000
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
    min=np.Inf
    save=np.array([])
    #change
    for j in list(itertools.permutations([0, 1,2], 3)):
        sum=0
        #change
        for k in range(0,3):
            sum += np.sum(np.abs(subject[k]-reference_subject[j[k]])**2)
            #sum += np.sum(subject[k] - reference_subject[j[k]])
        if min > sum:
            min = sum
            save = j
    print sum , save
    for i in range(save.shape[0]):
        for j in range(0,i+1):
            sum += np.sum(np.abs(subject[i]-reference_subject[j])**2)
            sys.stdout.write(sum)
            sys.stdout.flush()
            sys.stdout.write('     ')
            sys.stdout.flush()
        print()
    return save

def replot(r_labels,r_vertices,r_faces,label_matrix,reference_label):
    mlab.triangular_mesh(r_vertices[:, 0], r_vertices[:, 1], r_vertices[:, 2], r_faces, representation='surface',
                         opacity=1, scalars=np.float64(r_labels))
    mlab.gcf().scene.parallel_projection = True
    mlab.view(azimuth=0, elevation=90)
    mlab.colorbar(orientation='horizontal')
    for i in range(r_labels.shape[0]):
        if r_labels[i] == 1:
            r_labels[i]=label_matrix[0]+1
        elif r_labels[i] == 2:
            r_labels[i] = label_matrix[1]+1
        elif r_labels[i] == 3:
            r_labels[i] = label_matrix[2]+1
    mlab.triangular_mesh(r_vertices[:, 0], r_vertices[:, 1], r_vertices[:,2], r_faces, representation='surface',opacity=1, scalars=np.float64(r_labels))
    mlab.gcf().scene.parallel_projection = True
    mlab.view(azimuth=0, elevation=90)
    mlab.colorbar(orientation='horizontal')
    return r_labels

def avgplot(r_labels,nSubjects,r_vertices,r_faces):
    labels = np.zeros(r_labels.shape[0],dtype=float)
    for i in range(r_labels.shape[0]):
        labels[i] = np.sum(r_labels[i])/nSubjects
        #print(r_labels[i],labels[i])
    mlab.triangular_mesh(r_vertices[:, 0], r_vertices[:, 1], r_vertices[:, 2], r_faces, representation='surface',
                         opacity=1, scalars=np.float64(labels))
    mlab.gcf().scene.parallel_projection = True
    mlab.view(azimuth=0, elevation=90)
    mlab.colorbar(orientation='horizontal')
    labels = np.zeros(r_labels.shape[0], dtype=float)
    for i in range(r_labels.shape[0]):
        mode,count=stats.mode(r_labels[i])
        labels[i] = mode[0]
        #print(r_labels[i], labels[i])
    mlab.triangular_mesh(r_vertices[:, 0], r_vertices[:, 1], r_vertices[:, 2], r_faces, representation='surface',
                         opacity=1, scalars=np.float64(labels))
    mlab.gcf().scene.parallel_projection = True
    mlab.view(azimuth=0, elevation=90)
    mlab.colorbar(orientation='horizontal')
    #mlab.close()

def spatial_map(vector,r,msk_small_region,cent):
    r.labels = np.zeros([r.vertices.shape[0]])
    r.labels[~msk_small_region] = vector
    mlab.triangular_mesh(r.vertices[:, 0], r.vertices[:, 1], r.vertices[:,
                                                             2], r.faces, representation='surface',
                         opacity=1, scalars=np.float64(r.labels))
    mlab.points3d(cent[0], cent[1], cent[2])
    mlab.gcf().scene.parallel_projection = True
    mlab.view(azimuth=0, elevation=90)
    mlab.draw()
    #mlab.show()
    mlab.colorbar(orientation='horizontal')
    mlab.close()