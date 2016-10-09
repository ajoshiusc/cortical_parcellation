import numpy as np
import os
from surfproc import patch_color_labels, view_patch

def plot_histogram(nsub,temp):
    import matplotlib.pyplot as plt
    import pandas as pd
    Color = ['r', 'g', 'b', 'y', 'k', 'c', 'm', '#A78F1E', '#F78F1E', '#BE3224', 'w', 'r', 'g', 'b', 'y', 'k', 'c', 'm', '#A78F1E',
             '#F78F1E', '#BE3224', 'w', 'w']
    Color = np.tile(Color, nsub)
    temp = pd.Series.from_array(temp)
    plt.figure(figsize=(12, 8))
    ax = temp.plot(kind='bar', stacked=True, rot=0, color=Color)
    ax.set_title("VALIDATION PLOT", fontsize=53, fontweight='bold')
    plt.subplots_adjust(left=0.06, right=0.96, top=0.90, bottom=0.09)
    ax.set_xlabel("SUBJECTS ( LEFT HEMISPHERE FOLLOWED BY RIGHT HEMISPHERE )", fontsize=30, fontweight='bold')
    ax.set_ylabel("RAND INDEX", fontsize=20, fontweight='bold')
    ax.set_ylim(0, 1.75)
    import matplotlib.patches as mpatches

    NA = mpatches.Patch(color='r', label='Direct Mapping to Session 1')
    EU = mpatches.Patch(color='g', label='Direct Mapping to Session 2')
    AP = mpatches.Patch(color='b', label='Direct Mapping to Session 3')
    SA = mpatches.Patch(color='y', label='Direct Mapping to Session 4')
    NA1 = mpatches.Patch(color='k', label='Session 1 to Session 2')
    EU1 = mpatches.Patch(color='c', label='Session 1 to Session 3')
    AP1 = mpatches.Patch(color='m', label='Session 1 to Session 4')
    SA1 = mpatches.Patch(color='#A78F1E', label='Session 2 to Session 3')
    SA2 = mpatches.Patch(color='#F78F1E', label='Session 2 to Session 4')
    SA3 = mpatches.Patch(color='#BE3224', label='Session 3 to Session 4')
    plt.legend(handles=[NA, EU, AP, SA, NA1, EU1, AP1, SA1, SA2, SA3], loc=2)
    raw_data = {'first_name': ['nSub=118528', 'nSub=133928', 'nSub=196750', 'nSub=151526'],
                'pre_score': [4, 24, 31, 2],
                'mid_score': [25, 94, 57, 62],
                'post_score': [5, 43, 23, 23]}
    df = pd.DataFrame(raw_data, columns=['first_name', 'pre_score', 'mid_score', 'post_score'])

    # manually plotted
    ax.set_xticks([10, 32, 55.5, 79])

    ax.set_xticklabels(df['first_name'])
    ax.set_xticklabels(df['first_name'])
    plt.show()

def plot_hist_each_subject(nsub,temp):
    import matplotlib.pyplot as plt
    labels = ["D-to-S1", "D-to-S2", "S1-to-S2", ""]
    labels = np.tile(labels, nsub * 2)
    import pandas as pd
    temp = pd.Series.from_array(temp)
    ax = temp.plot(kind='bar')
    rects = ax.patches
    plt.figure(figsize=(12, 8))
    ax = temp.plot(kind='bar', stacked=True)
    ax.set_title("VALIDATION PLOT", fontsize=53, fontweight='bold')
    plt.subplots_adjust(left=0.06, right=0.96, top=0.90, bottom=0.09)
    ax.set_xlabel("SUBJECT_151526 ( LEFT HEMISPHERE FOLLOWED BY RIGHT HEMISPHERE )", fontsize=30, fontweight='bold')
    ax.set_ylabel("RAND INDEX", fontsize=20, fontweight='bold')

    for rect, label in zip(rects, labels):
        height = rect.get_height()
        # if height > 0:
        ax.text(rect.get_x() + rect.get_width() / 2, 1.007 * height, label, ha='center', va='bottom')
        prev = rect.get_x() + rect.get_width() / 2
    font = {'family': 'serif',
            'color': 'green',
            'weight': 'normal',
            'size': 11,
            }
    labels = "D-to-Si= Direct_Mapping_"
    plt.text(prev - .47, 0.96, labels, fontdict=font)
    labels = "               _to_ith_Session"
    plt.text(prev - .50, 0.93, labels, fontdict=font)
    labels = "Si-to-sj= ith_Session_to_"
    plt.text(prev - .50, 0.88, labels, fontdict=font)
    labels = "                  _jth_Session"
    plt.text(prev - .50, 0.85, labels, fontdict=font)
    plt.show()

def plot_fmri_subject(lst):
    sdir=['_RL','_LR']
    scan_type=['left','right']
    session_type=[1,2]
    data_file = 'validation'
    class sc:
        pass
    for sub in lst:
        for hemi in range(0,2):
            left=np.load(data_file + str(sub) + '_' + scan_type[hemi]  + sdir[0] + '_' + str(session_type[0]) + '.npz')
            sc.labels = left['labels']
            sc.vertices = left['vertices']
            sc.faces = left['faces']
            sc.vColor = np.zeros([sc.vertices.shape[0]])
            sc = patch_color_labels(sc, cmap='Paired', shuffle=True)
            view_patch(sc, show=1, colormap='Paired', colorbar=0)

from sklearn.metrics import adjusted_rand_score
p_dir = '/big_disk/HCP100-fMRI-NLM/HCP100-fMRI-NLM'
lst = os.listdir(p_dir)
#lst=['118528','133928','196750','151526']
save_dir= '/home/sgaurav/Documents/git_sandbox/cortical_parcellation/src/validation'

sdir=['_RL','_LR']
scan_type=['left','right']
session_type=[1,2]
temp=[]
for sub in lst:
    for hemi in range(0,2):
        direct= np.load(os.path.join(save_dir, 'very_smooth_data_'+scan_type[hemi] + '.npz'))
        dir_labels = direct['labels']
        for scan in range(0,4):
             fmri= np.load(os.path.join(save_dir, 'validation'+str(sub) + '_' + scan_type[hemi]  + sdir[scan/2] + '_' + str(session_type[scan%2]) + '.npz'))
             fmri_labels = fmri['labels']
             temp.append(adjusted_rand_score(dir_labels,fmri_labels))
        for scan1 in range(0, 4):
            fmri1 = np.load(os.path.join(save_dir,  'validation'+str(sub) + '_' + scan_type[hemi]  + sdir[scan1/2] + '_' + str(session_type[scan1%2]) + '.npz'))
            fmri1_labels = fmri1['labels']
            for scan2 in range(scan1 + 1, 4):
                fmri2 = np.load(os.path.join(save_dir, 'validation'+ str(sub) + '_' + scan_type[hemi]  + sdir[scan2/2] + '_' + str(session_type[scan2%2]) + '.npz'))
                fmri2_labels = fmri2['labels']
                if adjusted_rand_score(fmri1_labels, fmri2_labels) > 0:
                    temp.append(adjusted_rand_score(fmri1_labels, fmri2_labels))
        temp.append(0)
    temp.append(0)
temp.append(0)
temp=np.array(temp)
import scipy as sp
sp.savez(
    os.path.join(save_dir, 'sub_and_rand-index_data.npz'),
    rand_index=temp,subjects=lst)
nsub=lst.__len__()
plot_histogram(nsub,temp)