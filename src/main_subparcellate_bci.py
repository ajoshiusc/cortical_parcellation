# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 00:08:31 2016

@author: ajoshi
"""
from fmri_methods_sipi import reduce3_to_bci_rh, interpolate_labels
from dfsio import readdfs, writedfs
import scipy.io
import matlab.engine as meng
import os
from surfproc import patch_color_labels, view_patch_vtk, patch_color_attrib
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
bci_bst = readdfs('100307.BCI2reduce3.very_smooth.right.dfs')
bci_bst_sm = readdfs('100307.BCI2reduce3.very_smooth.right.dfs')

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

    labs = data1['labs_all'].squeeze()
    freq1 = data1['freq'].squeeze()
    freq1 = freq1/nSub
    freq1[labs == 0] = 0
#    bci_labs = reduce3_to_bci_rh(labs)
#    freq1 = reduce3_to_bci_rh(freq1)
    bci_labs = labs.squeeze().copy()
    bci_labs_orig = bci_labs.copy()

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
    bci_bst.labels += sp.uint16(bci_labs)


freq[freq == 0] = 1
bci_bst.attributes = freq
bci_bst = patch_color_labels(bci_bst, cmap='Paired')
# bci_bst = smooth_patch(bci_bst, iterations=90, relaxation=10.8)
view_patch_vtk(bci_bst, show=1)
####writedfs('/home/ajoshi/data/BCI-DNI_brain_atlas/BCI-DNI_brain.right.\
####mid.cortex_refined_labs_uncorr.dfs', bci_bst)
bci_bst = patch_color_attrib(bci_bst, bci_bst.attributes)
view_patch_vtk(bci_bst, show=1)

bci_bst = patch_color_labels(bci_bst, freq=freq, cmap='Paired')
# bci_bst = smooth_patch(bci_bst, iterations=90, relaxation=10.8)
view_patch_vtk(bci_bst, show=1)
####writedfs('/home/ajoshi/data/BCI-DNI_brain_atlas/BCI-DNI_brain.right.\
####mid.cortex_refined_labs_mod_freq_uncorr.dfs', bci_bst)
bci_bst = patch_color_attrib(bci_bst, bci_bst.labels)
view_patch_vtk(bci_bst, show=1)

bci_labs = reduce3_to_bci_rh(bci_bst.labels)
bci_freq = reduce3_to_bci_rh(bci_bst.attributes)

bci_bst = readdfs('/big_disk/ajoshi/data/BCI-DNI_brain_atlas/BCI-DNI_brain.right.\
mid.cortex.dfs')
bci_bst_sm = readdfs('/big_disk/ajoshi/data/BCI-DNI_brain_atlas/BCI-DNI_brain.\
right.mid.cortex_smooth10.dfs')
bci_bst.vertices = bci_bst_sm.vertices

bci_bst.labels = bci_labs
bci_bst.attributes = bci_freq

bci_bst = patch_color_labels(bci_bst, cmap='Paired')
view_patch_vtk(bci_bst, show=1)
writedfs('/big_disk/ajoshi/data/BCI-DNI_brain_atlas/BCI-DNI_brain.right.\
mid.cortex_refined_labs_uncorr.dfs', bci_bst)

bci_bst = patch_color_labels(bci_bst, freq=bci_bst.attributes, cmap='Paired')
# bci_bst = smooth_patch(bci_bst, iterations=90, relaxation=10.8)
view_patch_vtk(bci_bst, show=1)
writedfs('/big_disk/ajoshi/data/BCI-DNI_brain_atlas/BCI-DNI_brain.right.\
mid.cortex_refined_labs_mod_freq_uncorr.dfs', bci_bst)




surfname = '/big_disk/ajoshi/data/BCI-DNI_brain_atlas/BCI-DNI_brain.right.\
mid.cortex_refined_labs_uncorr.dfs'
sub_out = '/big_disk/ajoshi/data/BCI-DNI_brain_atlas/BCI-DNI_brain.right.\
mid.cortex_refined_labs_out.dfs'

eng = meng.start_matlab()
eng.addpath(eng.genpath('/big_disk/ajoshi/coding_ground/svreg-matlab/MEX_Files'))
eng.addpath(eng.genpath('/big_disk/ajoshi/coding_ground/svreg-matlab/3rdParty'))
eng.addpath(eng.genpath('/big_disk/ajoshi/coding_ground/svreg-matlab/src'))

eng.corr_topology_labels(surfname, sub_out)

bci_bst = readdfs(sub_out)
bci_bst = patch_color_labels(bci_bst, freq=bci_bst.attributes, cmap='Paired')
# bci_bst = smooth_patch(bci_bst, iterations=90, relaxation=10.8)
view_patch_vtk(bci_bst, show=1)
writedfs('/big_disk/ajoshi/data/BCI-DNI_brain_atlas/BCI-DNI_brain.right.\
mid.cortex_refined_labs_mod_freq.dfs', bci_bst)

bci_bst = patch_color_labels(bci_bst, cmap='Paired')
# bci_bst = smooth_patch(bci_bst, iterations=90, relaxation=10.8)
view_patch_vtk(bci_bst, show=1)
writedfs('/big_disk/ajoshi/data/BCI-DNI_brain_atlas/BCI-DNI_brain.right.\
mid.cortex_refined_labs.dfs', bci_bst)
