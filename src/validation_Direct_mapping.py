import numpy as np
import os
import xml.etree.ElementTree
from centroid import mapping

class s:
    pass

from surfproc import patch_color_labels, view_patch
right_hemisphere=[226,168,184,446,330,164,442,328,172,444,130,424,166,326,342,142,146,144,222,170,
150,242,186,120,422,228,224,322,310,162,324,500]

left_hemisphere=[227,169,185,447,331,165,443,329,173,445,131,425,167,327,343,143,147,145,223,171,
151,243,187,121,423,229,225,323,311,163,325,501]

roilists=right_hemisphere

scan_type=['right','left']
p_dir='/home/sgaurav/Documents/git_sandbox/cortical_parcellation'
e = xml.etree.ElementTree.parse('/home/ajoshi/for_gaurav/brainsuite_labeldescription.xml').getroot()
for hemi in range(0,2):
    refined_list=[]
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
    if hemi==0:
        roilists=right_hemisphere
    else:
        roilists=left_hemisphere
    T1=np.array(roilists)
    label_count=1
    labs_all=np.zeros([vertices.shape[0]])
    for i in xrange(T1.__len__()):
        flag = 0
        for j in xrange(5):
            if (T1[i] * 10 + j + 1) in refined_roilists.viewkeys():
                refined_list.append(T1[i]*10+j+1)
                flag = 1
                msk_small_region = np.in1d(lab, T1[i]*10 +j+1)
                labs_all[msk_small_region]=label_count
                label_count+=1
    s.labels=labs_all
    s.vertices=vertices
    s.faces=faces
    s.vColor=left_mid['vColor']
    #s=patch_color_labels(s,cmap='Paired',shuffle=True)
    #view_patch(s,show=1,colormap='Paired',colorbar=0)
    save_dir = '/home/sgaurav/Documents/git_sandbox/cortical_parcellation/src/validation'
    import scipy as sp
    sorted(refined_list)
    sp.savez(os.path.join(save_dir, 'direct_mapping'+scan_type[hemi] + '.npz'),
             labels=s.labels, vertices=s.vertices,
             faces=s.faces,roilists=lab,label=refined_list)