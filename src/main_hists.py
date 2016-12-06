#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 16:55:30 2016

@author: ajoshi
"""
import seaborn as sns
import scipy.io
import scipy as sp
import numpy as np
from pylab import figure, axes, pie, title, show, xlim
import matplotlib
matplotlib.rcParams.update({'font.size': 44})

a=sp.load('conn_sessions_pairwise_dist.npz')
conn_sessions=a['arr_0']
conn_sessions=a['arr_0']
conn_sessions=conn_sessions[conn_sessions!=0]

a=sp.load('rot_sessions_pairwise_dist.npz')
rot_sessions=a['arr_0']
rot_sessions=rot_sessions[rot_sessions!=0]

fig=figure()
xlim([0,2])

sns.kdeplot(rot_sessions,legend=True,color='b')
sns.kdeplot(conn_sessions,legend=True,color='r')
fig.savefig('kde_hist_sessions.pdf')


a=sp.load('conn_pairwise_dist.npz')
conn_sessions=a['arr_0']
conn_sessions=a['arr_0']
conn_sessions=conn_sessions[conn_sessions!=0]

a=sp.load('rot_pairwise_dist.npz')
rot_sessions=a['arr_0']
rot_sessions=rot_sessions[rot_sessions!=0]

fig=figure()
xlim([0,2])
sns.kdeplot(rot_sessions,legend=True,color='b')
sns.kdeplot(conn_sessions,legend=True,color='r')
fig.savefig('kde_hist_subjects.pdf')
