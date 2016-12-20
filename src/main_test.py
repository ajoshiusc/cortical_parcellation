# -*- coding: utf-8 -*-
import matlab.engine as meng
import scipy.io
import os
from surfproc import patch_color_labels, view_patch_vtk, smooth_patch, patch_color_attrib
# smooth_patch
import scipy as sp
from fmri_methods_sipi import reduce3_to_bci_rh, interpolate_labels
from dfsio import readdfs, writedfs

"""
Created on Wed Sep 14 00:34:30 2016

@author: ajoshi
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 00:08:31 2016

@author: ajoshi
"""


surfname = '/home/ajoshi/data/BCI-DNI_brain_atlas/BCI-DNI_brain.right.\
mid.cortex_refined_labs.dfs'
sub_out = '/home/ajoshi/data/BCI-DNI_brain_atlas/BCI-DNI_brain.right.\
mid.cortex_refined_labs_out.dfs'

eng = meng.start_matlab()
eng.addpath(eng.genpath('/home/ajoshi/coding_ground/svreg-matlab/MEX_Files'))
eng.addpath(eng.genpath('/home/ajoshi/coding_ground/svreg-matlab/3rdParty'))
eng.addpath(eng.genpath('/home/ajoshi/coding_ground/svreg-matlab/src'))

eng.corr_topology_labels(surfname, sub_out)



