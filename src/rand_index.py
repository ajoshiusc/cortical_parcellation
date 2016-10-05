import numpy as np
import os

'''0.870582369886
    0.871266789163
    0.848544510842
    0.877074771985'''
from sklearn.metrics import adjusted_rand_score

sdir=['_RL','_LR']
scan_type=['left','right']
session_type=[1,2]
for hemi in range(0,2):
    temp=[]
    direct= np.load('very_smooth_data_'+scan_type[hemi]+'.npz')
    dir_labels = direct['labels']
    prev=np.zeros([direct['vertices']])
    for scan in range(0,2):
         fmri= np.load('validation151526'+'_'+scan_type[hemi]+'_'+sdir[scan/2]+'_'+session_type[scan%2]+'.npz')
         fmri_labels = fmri['labels']
         temp.append(adjusted_rand_score(dir_labels,fmri_labels))
         temp.append(adjusted_rand_score(prev,fmri_labels))
         prev=fmri_labels
    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt

    # the histogram of the data
    n, bins, patches = plt.hist(temp, 50, normed=1, facecolor='green', alpha=0.75)

    # add a 'best fit' line
    #y = mlab.normpdf(bins, mu, sigma)
    #l = plt.plot(bins, y, 'r--', linewidth=1)

    plt.xlabel('subject')
    plt.ylabel('RAND INDEX')
    plt.title(r'Rand indeices')
    plt.axis([40, 160, 0, 0.03])
    plt.grid(True)

    plt.show()

