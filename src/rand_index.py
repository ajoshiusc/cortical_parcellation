import numpy as np
import os

from matplotlib.colors import LogNorm

nsub=1
from sklearn.metrics import adjusted_rand_score

sdir=['_RL','_LR']
scan_type=['left','right']
session_type=[1,2]
temp=[]
for hemi in range(0,2):
    direct= np.load('very_smooth_data_'+scan_type[hemi]+'.npz')
    dir_labels = direct['labels']
    prev=np.zeros([direct['vertices'].shape[0]])
    for scan in range(0,2):
         fmri= np.load('validation151526'+'_'+scan_type[hemi]+sdir[scan/2]+'_'+str(session_type[scan%2])+'.npz')
         fmri_labels = fmri['labels']
         temp.append(adjusted_rand_score(dir_labels,fmri_labels))
         if adjusted_rand_score(prev,fmri_labels) >0:
            temp.append(adjusted_rand_score(prev,fmri_labels))
         prev=fmri_labels
    temp.append(0)
temp.append(0)
temp=np.array(temp)
import matplotlib.pyplot as plt
labels=["D-to-S1","D-to-S2","S1-to-S2",""]
labels=np.tile(labels,nsub*2)
import pandas as pd
temp=pd.Series.from_array(temp)
ax = temp.plot(kind='bar')
rects = ax.patches
plt.figure(figsize=(12, 8))
ax = temp.plot(kind='bar',stacked=True)
ax.set_title("VALIDATION PLOT",fontsize=53, fontweight='bold')
plt.subplots_adjust(left=0.06, right=0.96, top=0.90, bottom=0.09)
ax.set_xlabel("SUBJECT_151526 ( LEFT HEMISPHERE FOLLOWED BY RIGHT HEMISPHERE )",fontsize=30, fontweight='bold')
ax.set_ylabel("RAND INDEX",fontsize=20, fontweight='bold')

for rect, label in zip(rects, labels):
    height = rect.get_height()
    #if height > 0:
    ax.text(rect.get_x() + rect.get_width()/2, 1.007*height, label, ha='center', va='bottom')
    prev=rect.get_x() + rect.get_width()/2
font = {'family': 'serif',
            'color': 'green',
            'weight': 'normal',
            'size': 11,
            }
labels="D-to-Si= Direct_Mapping_"
plt.text(prev-.47, 0.96, labels ,fontdict=font)
labels="               _to_ith_Session"
plt.text(prev-.50, 0.93, labels ,fontdict=font)
labels="Si-to-sj= ith_Session_to_"
plt.text(prev-.50, 0.88, labels ,fontdict=font)
labels="                  _jth_Session"
plt.text(prev-.50, 0.85, labels ,fontdict=font)
plt.show()