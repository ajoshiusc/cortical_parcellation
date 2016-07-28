# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 14:59:29 2016

@author: ajoshi
"""
import scipy as sp
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.linear_assignment_ import linear_assignment


def rot_sub_data(ref,sub):
    xcorr=sp.dot((sub.T),ref)
    u,s,v=sp.linalg.svd(xcorr)
    R=sp.dot(v.T,u.T)
    return sp.dot(sub,R.T)


def reorder_labels(labels):
    
    nClusters=sp.int32(sp.amax(labels.flatten())+1)
    labels0_vec = sp.zeros((labels.shape[0],nClusters),'bool')
    labelsi_vec = labels0_vec.copy()
    for i in range(nClusters):
        labels0_vec[:,i] = (labels[:,0]==i)
        
        
    for i in range(labels.shape[1]):
        for j in range(nClusters):
            labelsi_vec[:,j] = (labels[:,i]==j)
    
        D=pairwise_distances(labelsi_vec.T, labels0_vec.T, metric='dice')
        D[~sp.isfinite(D)]=1
        ind1 = linear_assignment(D)
        labels[:,i] = ind1[sp.int16(labels[:,i]),1]
        
    return labels
        
        