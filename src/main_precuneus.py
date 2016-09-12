from fmri_parcellation import parcellate_region
import os
import scipy as sp
import numpy as np
roilist = []


#specify the roilist
#roiregion=['motor','precuneus','temporal','cingulate','semato','visual']

#roilist = np.array([[29,69,70],[(30, 72, 9, 47)],[33,34,35,36,74],[6,7,8,9,10],[28],[(2,22,11,58,59,20,43,19,45)]])



#nClusters=np.array([3,3,7,3,2,4])
#roilist =  np.array([[143],[142],[145],[144],[147],[146],[151],[150],[161],[160],[163],[162]])
#roiregion=['pars opercularis','pars triangularis','pars orbitalis','pre-central gyrus','pole|orbital frontal lobe','transvers frontal gyrus']
roilist =  np.array([[147],[146]])
roiregion=['pars orbitalis','motor','temporal','precuneus','semato','visual']
nClusters=np.array([1])

p_dir = '/home/ajoshi/data/HCP_data/data'
lst = os.listdir(p_dir) #{'100307'}

sdir=['_LR','_RL']
scan_type=['left','right']
session_type=[1,2]
fadd_1='.rfMRI_REST'
fadd_2='.reduce3.ftdata.NLM_11N_hvar_25.mat'
# %% Across session study
for n in range(nClusters.shape[0]):
    for i in range(0,2):
        R_all = []

        labs_all_1 = []
        vert_all_1 = []
        faces_all_1 = []
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
                labs1, correlation_within_precuneus_vector, correlation_with_rest_vector, mask, centroid = parcellate_region(
                    roilist[i], sub, nClusters[n], sdir[1], scan_type[i],
                    1, session_type[0], 0, 0)
                count1 += 1
                if count1 == 1:
                    labs_all_1 = sp.array(labs1.labels)
                    vert_all_1 = sp.array(labs1.vertices)
                    faces_all_1 = sp.array(labs1.faces)
                    correlation_within_precuneus = sp.array(correlation_within_precuneus_vector)
                    correlation_with_rest = sp.array(correlation_with_rest_vector)
                    all_centroid = sp.array(centroid)
                else:
                    labs_all_1 = sp.vstack([labs_all_1, labs1.labels])
                    vert_all_1 = sp.array([labs1.vertices])
                    faces_all_1 = sp.array([labs1.faces])
                    correlation_within_precuneus = sp.vstack(
                        [correlation_within_precuneus, correlation_within_precuneus_vector])
                    correlation_with_rest = sp.vstack([correlation_with_rest, correlation_with_rest_vector])
                    all_centroid = sp.vstack([all_centroid, centroid])

        data_file = 'data_file'
        sp.savez(data_file + roiregion[n] +str(i) +'BCI_overall.npz', correlation_within_precuneus=correlation_within_precuneus,
                 correlation_with_rest=correlation_with_rest, labels=labs_all_1, vertices=labs1.vertices,
                 faces=labs1.faces, mask=mask, centroid=all_centroid)

