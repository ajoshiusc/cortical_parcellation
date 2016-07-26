# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 14:59:29 2016

@author: ajoshi
"""
import scipy as sp

def rot_sub_data(ref,sub):
    xcorr=sp.dot((sub.T),ref)
    u,s,v=sp.linalg.svd(xcorr)
    R=sp.dot(v.T,u.T)
    return sp.dot(sub,R.T)
    