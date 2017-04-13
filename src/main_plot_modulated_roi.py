#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 01:25:01 2017

@author: ajoshi
"""

# ||AUM||
from scipy.stats import itemfreq
import scipy.io
import scipy as sp
import numpy as np
from fmri_methods_sipi import rot_sub_data, reorder_labels
from dfsio import readdfs, writedfs
#import h5py
import os
from surfproc import view_patch, view_patch_vtk, get_cmap, patch_color_labels, patch_color_attrib
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.metrics import adjusted_rand_score

s = readdfs('/big_disk/ajoshi/coding_ground/svreg-matlab/BCI-DNI_brain_atlas_refined/BCI-DNI_brain.left.mid.cortex.mod.dfs')
sl = readdfs('/big_disk/ajoshi/coding_ground/svreg-matlab/BCI-DNI_brain_atlas_refined/BCI-DNI_brain.left.mid.cortex.dfs')
sm = readdfs('/big_disk/ajoshi/coding_ground/svreg-matlab/BCI-DNI_brain_atlas_refined/BCI-DNI_brain.left.mid.cortex_smooth10.dfs')
s.labels = sl.labels.copy()
flg = (s.labels != 187) & (s.labels != 189) & (s.labels != 191)
#131 middle frontal
s.attributes[flg] = 0
s.labels[flg] = 0

s.attributes = 2.0*sp.maximum(s.attributes-0.5,0)

s = patch_color_labels(s, freq=s.attributes, cmap='hsv')
#s = patch_color_attrib(s,cmap='gray',clim=[0,1])

s.vertices = sm.vertices
view_patch_vtk(s,outfile='mod_map_cingulate2.png',show=0, azimuth=90, elevation=0, roll=-90)

