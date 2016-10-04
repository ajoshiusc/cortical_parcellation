import numpy as np
import os
import xml.etree.ElementTree
from centroid import mapping
from dfsio import readdfs
import scipy.io

class s:
    pass

from surfproc import patch_color_labels, view_patch

scan_type=['left','right']
p_dir='/home/sgaurav/Documents/git_sandbox/cortical_parcellation'
e = xml.etree.ElementTree.parse('/home/ajoshi/for_gaurav/brainsuite_labeldescription.xml').getroot()
for hemi in range(1,2):
    left_mid = np.load('very_smooth_data_'+scan_type[hemi]+'.npz')
    lab=left_mid['labels']
    vertices=left_mid['vertices']
    faces=left_mid['faces']
    refined_roilists = {}
    refined_roilists[lab[0]] = 1
    roilist_count = 0
    for label_id in range(lab.shape[0]):
        roilist_count = mapping(refined_roilists, roilist_count, lab[label_id], 1)
    sorted(refined_roilists)
    T1 = np.array([int(atype.get('id')) for atype in e.findall('label') ])
    label_count=1
    labs_all=np.zeros([vertices.shape[0]])
    for i in xrange(T1.__len__()):
        flag = 0
        for j in xrange(5):
            if (T1[i] * 10 + j + 1) in refined_roilists.viewkeys():
                flag = 1
                msk_small_region = np.in1d(lab, T1[i]*10 +j+1)
                labs_all[msk_small_region]=label_count
                label_count+=1
    s.labels=labs_all
    s.vertices=vertices
    s.faces=faces
    s.vColor=left_mid['vColor']
    s=patch_color_labels(s,cmap='Paired',shuffle=True)
    view_patch(s,show=1,colormap='Paired',colorbar=0)