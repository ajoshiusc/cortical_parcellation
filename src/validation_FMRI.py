import numpy as np
import xml.etree.ElementTree
import xml.etree.cElementTree as ET
from centroid import mapping
from dfsio import readdfs
from xml.etree import ElementTree
from xml.dom import minidom

e = xml.etree.ElementTree.parse('/home/ajoshi/for_gaurav/brainsuite_labeldescription.xml').getroot()
left_mid = readdfs('/home/ajoshi/data/BCI-DNI_brain_atlas/BCI-DNI_brain.left.mid.cortex.refined.dfs')
right_mid = readdfs('/home/ajoshi/data/BCI-DNI_brain_atlas/BCI-DNI_brain.right.mid.cortex.refined.dfs')
refined_roilists = {}
refined_roilists[left_mid.labels[0]] = left_mid.vColor[0]
roilist_count = 0
for label_id in range(left_mid.labels.shape[0]):
    roilist_count = mapping(refined_roilists, roilist_count, left_mid.labels[label_id], left_mid.vColor[label_id])
    #roilist_count = mapping(refined_roilists, roilist_count, right_mid.labels[label_id], right_mid.vColor[label_id])
sorted(refined_roilists)
T1 = np.array([int(atype.get('id')) for atype in e.findall('label') ])
label_count=0
labs_all=np.zeros([left_mid.vertices.shape[0]])
for i in xrange(T1.__len__()):
    flag = 0
    for j in xrange(5):
        if (T1[i] * 10 + j + 1) in refined_roilists.viewkeys():
            flag = 1
            msk_small_region = np.in1d(left_mid.labels, T1[i]*10 +j+1)
            labs_all[msk_small_region]=label_count
            label_count+=1
    if flag == 0 and T1[i] in refined_roilists:
        msk_small_region = np.in1d(left_mid.labels, T1[i])
        labs_all[msk_small_region] = label_count
        label_count += 1
from mayavi import mlab

mlab.figure(size=(1024, 768), \
            bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))
mlab.triangular_mesh(left_mid.vertices[:, 0], left_mid.vertices[:, 1], left_mid.vertices[:, 2], left_mid.faces,
                     representation='surface',
                     opacity=1, scalars=np.float64(labs_all.transpose()))
mlab.gcf().scene.parallel_projection = True
mlab.view(azimuth=0, elevation=90)
mlab.colorbar(orientation='vertical')
mlab.show()
