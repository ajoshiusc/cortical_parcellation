import numpy as np
import xml.etree.ElementTree
import xml.etree.cElementTree as ET
from centroid import mapping
from dfsio import readdfs
from xml.etree import ElementTree
from xml.dom import minidom

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

e = xml.etree.ElementTree.parse('/home/ajoshi/for_gaurav/brainsuite_labeldescription.xml').getroot()
root = ET.Element("labelset")
left_mid = readdfs('/home/ajoshi/data/BCI-DNI_brain_atlas/BCI-DNI_brain.left.mid.cortex.refined.dfs')
refined_roilists={}
refined_roilists[left_mid.labels[0]]=left_mid.vColor[0]
roilist_count=0
label_list=left_mid.labels.tolist()
vColor_list=left_mid.vColor.tolist()
for label_id in range(left_mid.labels.shape[0]):
    roilist_count=mapping(refined_roilists,roilist_count,left_mid.labels[label_id],left_mid.vColor[label_id])
left_mid = readdfs('/home/ajoshi/data/BCI-DNI_brain_atlas/BCI-DNI_brain.right.mid.cortex.refined.dfs')
for label_id in range(left_mid.labels.shape[0]):
    roilist_count=mapping(refined_roilists,roilist_count,left_mid.labels[label_id],left_mid.vColor[label_id])
sorted(refined_roilists)
T1 = np.array([int(atype.get('id')) for atype in e.findall('label') ])
T2 = [atype.get('tag') for atype in e.findall('label') ]
T3 = [atype.get('fullname') for atype in e.findall('label') ]
T4 = [atype.get('color') for atype in e.findall('label') ]
label_list=label_list+left_mid.labels.tolist()
vColor_list=vColor_list+left_mid.vColor.tolist()
for i in xrange(T2.__len__()):
    flag=0
    for j in xrange(5):
        if (T1[i]*10 + j+1) in refined_roilists.viewkeys():
            flag=1
            ET.SubElement(root,"label\t",id=str(T1[i]*10+j+1)+"\t",tag = T2[i]+"_"+str(j+1)+"\t", color=str(hex(255*np.array(refined_roilists[T1[i]*10+j+1][0])))[2:-1]+str(hex(255*np.array(refined_roilists[T1[i]*10+j+1][1])))[2:-1]+str(hex(255*np.array(refined_roilists[T1[i]*10+j+1][2])))[2:-1],  vfullname ="\t"+ T3[i]+"_"+str(j+1)+"\t")
    if flag == 0:
        ET.SubElement(root, "label\t", id=str(T1[i])+"\t", tag=T2[i]+"\t", color=T4[i][2:],vfullname="\t"+T3[i]+"\t" )
with open("Brainsuite_refined_label_description.xml","w") as f:
    f.write(prettify(root))