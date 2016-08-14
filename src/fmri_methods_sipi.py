# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 14:59:29 2016

@author: ajoshi
"""
import scipy as sp
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.linear_assignment_ import linear_assignment
import matplotlib.pyplot as plt


def rot_sub_data(ref,sub):
    """ref and sub matrices are of the form (vertices x time) """
    xcorr=sp.dot((sub.T),ref)
    u,s,v=sp.linalg.svd(xcorr)
    R=sp.dot(v.T,u.T)
    return sp.dot(sub,R.T)


def show_slices(slices,vmax=None,vmin=None):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        a = axes[i].imshow(slice.T, cmap="gray", origin="lower",vmax=vmax,vmin=vmin)
    fig.colorbar(a)


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
        
import numpy as np
from scipy.stats import f as f_distrib

def hotelling_t2(X,Y):
    """X and Y are n_features x n_subjects x n_voxels"""
    nx=X.shape[1];ny=Y.shape[1];p=X.shape[0];
    Xbar=X.mean(1);Ybar=Y.mean(1);
    Xbar=Xbar.reshape(Xbar.shape[0],1,Xbar.shape[1])
    Ybar=Ybar.reshape(Ybar.shape[0],1,Ybar.shape[1])

    X_Xbar=X-Xbar
    Y_Ybar=Y-Ybar
    Wx=np.einsum('ijk,ljk->ilk',X_Xbar,X_Xbar)
    Wy=np.einsum('ijk,ljk->ilk',Y_Ybar,Y_Ybar)
    W=(Wx+Wy)/float(nx+ny-2)
    Xbar_minus_Ybar=Xbar-Ybar
    x = np.linalg.solve(W.transpose(2,0,1), Xbar_minus_Ybar.transpose(2,0,1))
    x=x.transpose(1,2,0);

    t2=np.sum(Xbar_minus_Ybar*x,0)
    t2=t2*float(nx*ny)/float(nx+ny);
    stat=(t2*float(nx+ny-1-p)/(float(nx+ny-2)*p));

    pval=1-np.squeeze(f_distrib.cdf(stat,p,nx+ny-1-p));
    return pval,t2
        