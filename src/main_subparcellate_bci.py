# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 00:08:31 2016

@author: ajoshi
"""
from fmri_methods_sipi import reduce3_to_bci_lh
from dfsio import readdfs, writedfs
import scipy.io
import os 
from surfproc import view_patch, patch_color_labels, smooth_patch
import scipy as sp

outputfile='gaurav_bci.dfs'

p_dir='/home/sgaurav/Documents/git_sandbox/cortical_parcellation/src/intensity_mode_map'

lst = os.listdir(p_dir)
bci_bst = readdfs('/home/ajoshi/data/BCI-DNI_brain_atlas/BCI-DNI_brain.left.\
mid.cortex.dfs')
for fname in lst:
 #fname='intensity_file_cingulate_184_nCluster=3_BCI.mat'
    s=fname.find('_nCluster')
    
    roino=int(fname[s-3:s])
    print roino
    if sp.mod(roino,2) == 0:
        continue
    
    data1 = scipy.io.loadmat(os.path.join(p_dir, fname))
    
    labs = data1['labs_all']
    bci_labs=reduce3_to_bci_lh(labs)
    

    bci_bst.labels += bci_labs
    
    #bci_bst = smooth_patch(bci_bst, iterations=1000, relaxation=0.8)
    
    #writedfs(outputfile, bci_bst)
bci_bst=patch_color_labels(bci_bst,cmap='Set1')
bci_bst=smooth_patch(bci_bst,iterations=1000,relaxation=0.8)
view_patch(bci_bst,show=1)
writedfs('/home/ajoshi/data/BCI-DNI_brain_atlas/BCI-DNI_brain.left.\
mid.cortex_refined_labs_smooth.dfs',bci_bst)

