from fmri_data_collection import ALL_parcellate_region
import os
import scipy as sp
import numpy as np
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt


roilist = []
for i in range(1,76):
    roilist.append(i)


p_dir = '/home/ajoshi/HCP_data'
lst = os.listdir(p_dir) #{'100307'}
#%%
mean_R = sp.zeros(6)
std_R = sp.zeros(6)
sdir=['_LR','_RL']
session_type=[1,2]
fadd_1='.rfMRI_REST'
fadd_2='.reduce3.ftdata.NLM_11N_hvar_25.mat'
R_all = np.array([])  # range(6):
labs_all_1 = []
labs_all_2 = []
vert_all_1 = []
vert_all_2 = []
faces_all_1 = []
faces_all_2 = []
all_subjects = sp.array([])
all_centroid = sp.array([])
count1 = 0
count_break = 0
session = []
centroid = []
for sub in lst:
    count_break += 1
    print count_break
    if os.path.isfile(os.path.join(p_dir, sub, sub + fadd_1 + str(session_type[0]) + sdir[1] + fadd_2)):

            # (46,28,29) motor 243 is precuneus
            #(2,11,19,20,22,43,45,58,59,66) visual by gaurav
        R_all, mask,vertices,faces = ALL_parcellate_region((6,7,8,9,10), sub, R_all)

# sp.savez_compressed('clustering_results_sessions_region_pc', R_all=R_all)
data_file = 'all_data_file'
sp.savez(data_file  + '.npz', R_all=R_all,mask=mask,vertices=vertices,faces=faces)
print
#%%
'''fig = plt.figure()
plt.boxplot(R_all)
fig.savefig('across_subjects_adj_rand_sessions_region_pc.pdf')'''


#%%