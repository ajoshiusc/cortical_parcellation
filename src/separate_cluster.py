import scipy as sp
import numpy as np
from centroid import  modified_find_centroid


def separate(labels, r,temp,nCluster):
    c=[]
    for i in range(nCluster):
        c.append([])
    all_centroid=[]
    for i in range(r.labels.shape[0]):
        for j in range(nCluster):
            if r.labels[i] == (j+1):
                c[j].append(r.vertices[i])
    for i in range(nCluster):
        c[i]=np.array(c[i])
        if i == 0:
            all_centroid = sp.array(modified_find_centroid(c[i]))
        else:
            all_centroid = sp.vstack([all_centroid, modified_find_centroid(c[i])])
    return (all_centroid)