from fmri_data_collection import ALL_parcellate_region
import os
import scipy as sp
import numpy as np
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
roilist = []


#specify the roilist

'''roiregion=['motor','precuneus','temporal','cingulate','semato','visual']

roilist = np.array([[29,69,70],[(30, 72, 9, 47)],[33,34,35,36,74],[6,7,8,9,10],[28],[(2,22,11,58,59,20,43,19,45)]])

nClusters=np.array([3,3,7,3,2,4])'''

roilist =  np.array([[445],[444],[447],[446],[501],[500],[423],[422],[425],[424],[443],[442]])
roiregion=['lingual gyrus','cuneus','Insula','superior occipital gyrus','middle occipital gyrus','inferior occipital gyrus']
nClusters=np.array([5])


p_dir = '/home/ajoshi/data/HCP_data/data'
lst = os.listdir(p_dir) #{'100307'}
#%%
mean_R = sp.zeros(6)
std_R = sp.zeros(6)
sdir=['_LR','_RL']
session_type=[1,2]
scan_type=['left','right']
fadd_1='.rfMRI_REST'
fadd_2='.reduce3.ftdata.NLM_11N_hvar_25.mat'
for n in range(0,6):
    for i in range(0,2):
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
                R_all, mask,vertices,faces = ALL_parcellate_region(roilist[n*2+i], sub, R_all,scan_type[i])

        d = R_all[mask, :]
        rho = np.corrcoef(d)
        rho[~np.isfinite(rho)] = 0
        # rho = np.abs(rho)
        d_corr = R_all[~mask, :]
        rho_1 = np.corrcoef(d, d_corr)
        rho_1 = rho_1[range(d.shape[0]), d.shape[0]:]
        rho_1[~np.isfinite(rho_1)] = 0
        # sp.savez_compressed('clustering_results_sessions_region_pc', R_all=R_all)
        data_file = 'all_data_file'
        sp.savez(data_file  + roiregion[n] +'BCI_'+scan_type[i]+'_overall.npz', rho=rho,rho_1=rho_1,mask=mask,vertices=vertices,faces=faces)