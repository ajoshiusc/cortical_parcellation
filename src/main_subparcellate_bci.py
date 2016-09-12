# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 00:08:31 2016

@author: ajoshi
"""
from fmri_methods_sipi import reduce3_to_bci_rh, interpolate_labels
from dfsio import readdfs, writedfs
import scipy.io
import os
from surfproc import patch_color_labels, view_patch_vtk, smooth_patch, patch_color_attrib
# smooth_patch
import scipy as sp

outputfile = 'gaurav_bci.dfs'
# initialize the structures


class ts:
    pass


class fs:
    pass

p_dir = '/home/sgaurav/Documents/git_sandbox/cortical_parcellation/src/\
intensity_mode_map/'
nSub = 40
lst = os.listdir(p_dir)
bci_bst = readdfs('/home/ajoshi/data/BCI-DNI_brain_atlas/BCI-DNI_brain.right.\
mid.cortex.dfs')
bci_bst_sm = readdfs('/home/ajoshi/data/BCI-DNI_brain_atlas/BCI-DNI_brain.\
right.mid.cortex_smooth10.dfs')
bci_bst.vertices = bci_bst_sm.vertices

freq = sp.zeros(bci_bst.vertices.shape[0])
bci_bst.labels *= 10
for fname in lst:
    if not (fname.endswith('.mat')):
        continue

    s = fname.find('_nCluster')

    roino = int(fname[s-3:s])
    print roino
    if sp.mod(roino, 2) != 0:
        continue

    data1 = scipy.io.loadmat(os.path.join(p_dir, fname))

    labs = data1['labs_all']
    freq1 = data1['freq']
    print 'max freq1 = %f' % sp.amax(freq1)
    freq1 = freq1/nSub
    freq1[labs == 0] = 0
    bci_labs = reduce3_to_bci_rh(labs)
    freq1 = reduce3_to_bci_rh(freq1)
    bci_labs_orig = bci_labs

    if sp.amax(bci_labs) > 0:
        bci_labs[bci_bst.labels != roino*10] = 0
        indt = (bci_bst.labels == roino*10) & (bci_labs == 0)
        ts.vertices = bci_bst.vertices[indt, :]
        ind = (bci_bst.labels == roino*10) & (bci_labs != 0)
        fs.vertices = bci_bst.vertices[ind, :]
        fs.labels = bci_labs[ind]
        ts = interpolate_labels(fs, ts)
        bci_labs[indt] = ts.labels

    freq1[(bci_labs > 0) & (freq1 == 0)] = \
        sp.amin(freq1[(bci_labs_orig > 0) & (freq1 != 0)])

    freq[(bci_bst.labels == roino*10)] += freq1[(bci_bst.labels == roino*10)]
    print 'max freq = %f' % sp.amax(freq)
    bci_bst.labels += bci_labs


freq[freq == 0] = 1
bci_bst.attributes = freq
bci_bst = patch_color_labels(bci_bst, cmap='Paired')
# bci_bst = smooth_patch(bci_bst, iterations=90, relaxation=10.8)
view_patch_vtk(bci_bst, show=1)
writedfs('/home/ajoshi/data/BCI-DNI_brain_atlas/BCI-DNI_brain.right.\
mid.cortex_refined_labs.dfs', bci_bst)
bci_bst = patch_color_attrib(bci_bst, bci_bst.labels)
view_patch_vtk(bci_bst, show=1)

bci_bst = patch_color_labels(bci_bst, freq=freq, cmap='Paired')
# bci_bst = smooth_patch(bci_bst, iterations=90, relaxation=10.8)
view_patch_vtk(bci_bst, show=1)
writedfs('/home/ajoshi/data/BCI-DNI_brain_atlas/BCI-DNI_brain.right.\
mid.cortex_refined_labs_mod_freq.dfs', bci_bst)
