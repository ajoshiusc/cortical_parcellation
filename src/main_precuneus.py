from centroid import num_of_cluster, neighbour_correlation
from fmri_parcellation import parcellate_region
import os
import scipy as sp

p_dir = '/home/ajoshi/HCP_data'
lst = os.listdir(p_dir) #{'100307'}
#%%
mean_R = sp.zeros(6)
std_R = sp.zeros(6)
sdir=['_LR','_RL']
scan_type=['left','right']
session_type=[1,2]
fadd_1='.rfMRI_REST'
fadd_2='.reduce3.ftdata.NLM_11N_hvar_25.mat'
# %% Across session study
R_all = []

roilist = []
for i in range(0,76):
    roilist.append(i)

#nClusters=num_of_cluster()
nClusters=3

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
        labs1, correlation_within_precuneus_vector, correlation_with_rest_vector, mask, centroid = parcellate_region((30, 72, 9, 47), sub, nClusters, sdir[1], scan_type[0],
                                                               1, session_type[0], 0,0)
        count1 += 1
        if count1 == 1:
            labs_all_1 = sp.array(labs1.labels)
            vert_all_1 = sp.array(labs1.vertices)
            faces_all_1 = sp.array(labs1.faces)
            correlation_within_precuneus = sp.array(correlation_within_precuneus_vector)
            correlation_with_rest=sp.array(correlation_with_rest_vector)
            all_centroid = sp.array(centroid)
        else:
            labs_all_1 = sp.vstack([labs_all_1, labs1.labels])
            vert_all_1 = sp.array([labs1.vertices])
            faces_all_1 = sp.array([labs1.faces])
            correlation_within_precuneus = sp.vstack([correlation_within_precuneus, correlation_within_precuneus_vector])
            correlation_with_rest= sp.vstack([correlation_with_rest, correlation_with_rest_vector])
            all_centroid = sp.vstack([all_centroid, centroid])


# sp.savez_compressed('clustering_results_sessions_region_pc', R_all=R_all)
data_file = 'data_file'
sp.savez(data_file  + '.npz', correlation_within_precuneus=correlation_within_precuneus,correlation_with_rest=correlation_with_rest, labels=labs_all_1, vertices=labs1.vertices,
         faces=labs1.faces, mask=mask, centroid=all_centroid)


#%%
'''fig = plt.figure()
plt.boxplot(R_all)
fig.savefig('across_subjects_adj_rand_sessions_region_pc.pdf')'''


#%%