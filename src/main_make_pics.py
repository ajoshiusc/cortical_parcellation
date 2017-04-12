# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 15:51:16 2016

@author: ajoshi
"""
from dfsio import readdfs
from surfproc import view_patch_vtk

subbasename='/big_disk/ajoshi/fs_dir/co20050723_090747MPRAGET1Coronals002a001'
hemi = 'left'

s = readdfs(subbasename + '/' + hemi + '.mid.dfs')

view_patch_vtk(s, outfile=subbasename+'/mri/BST/fs_' + hemi + '1.png', show=0)
view_patch_vtk(s, outfile=subbasename+'/mri/BST/fs_' + hemi + '2.png', azimuth=-90,
               roll=90, show=0)

s = readdfs(subbasename+'/mri/BST/orig.' + hemi + '.mid.cortex.svreg.dfs')

view_patch_vtk(s, outfile=subbasename+'/mri/BST/bst_' + hemi + '1.png', show=0)
view_patch_vtk(s, outfile=subbasename+'/mri/BST/bst_' + hemi + '2.png', azimuth=-90,
               roll=90, show=0)

s = readdfs(subbasename+'/mri/BST/atlas.' + hemi + '.mid.cortex.svreg.dfs')

view_patch_vtk(s, outfile=subbasename+'/mri/BST/atlas_' + hemi + '1.png', show=0)
view_patch_vtk(s, outfile=subbasename+'/mri/BST/atlas_' + hemi + '2.png', azimuth=-90,
               roll=90, show=0)

