# ||AUM||
import scipy.io
import scipy as sp
import numpy as np
from fmri_methods_sipi import rot_sub_data
from dfsio import readdfs
import os
from sklearn.cluster import KMeans

p_dir = '/home/ajoshi/HCP_data/data'
p_dir_ref='/home/ajoshi/HCP_data'
lst = os.listdir(p_dir)
lst=lst[:5]
r_factor = 3
ref_dir = os.path.join(p_dir_ref, 'reference')
nClusters=60

ref = '100307'
print(ref + '.reduce' + str(r_factor) + '.LR_mask.mat')
fn1 = ref + '.reduce' + str(r_factor) + '.LR_mask.mat'
fname1 = os.path.join(ref_dir, fn1)
msk = scipy.io.loadmat(fname1)  # h5py.File(fname1);
dfs_left = readdfs(os.path.join(p_dir_ref, 'reference', ref + '.aparc.a2009s.32k_fs.reduce3.left.dfs'))
dfs_left_sm = readdfs(os.path.join(p_dir_ref, 'reference', ref + '.aparc.a2009s.32k_fs.reduce3.very_smooth.left.dfs'))
count1 = 0
rho_rho=[];rho_all=[]

labs_all=sp.zeros((len(dfs_left.labels),len(lst)))

for sub in lst:
    data = scipy.io.loadmat(os.path.join(p_dir, sub, sub + '.rfMRI_REST1_RL.reduce3.ftdata.NLM_11N_hvar_25.mat'))
    LR_flag = msk['LR_flag']
    LR_flag = np.squeeze(LR_flag) > 0
    data = data['ftdata_NLM']
    temp = data[LR_flag, :]
    m = np.mean(temp, 1)
    temp = temp - m[:,None]
    s = np.std(temp, 1)+1e-16
    temp = temp/s[:,None]
    d = temp
    
    if count1==0:        
        sub_data = sp.zeros((d.shape[0],d.shape[1],len(lst)))

    sub_data[:,:,count1] = d

    count1+=1
    print count1,
    
nSub=sub_data.shape[2]

cat_data=sp.zeros((nSub*sub_data.shape[0],sub_data.shape[1]))

for ind in range(nSub):
    sub_data[:,:,ind] = rot_sub_data(ref=sub_data[:,:,0],sub=sub_data[:,:,ind])
    cat_data[sub_data.shape[0]*ind:sub_data.shape[0]*(ind+1),:] = sub_data[:,:,ind]    
    print ind, sub_data.shape, cat_data.shape

 
SC = KMeans(n_clusters=nClusters,random_state=5324)
lab_sub=sp.zeros((sub_data.shape[0],nSub))
for ind in range(nSub):
    lab_sub[:,ind]=SC.fit_predict(sub_data[:,:,ind])    
#labs_all = SC.fit_predict(cat_data)

#lab_sub=labs_all.reshape((sub_data.shape[0],nSub),order='F')
sp.savez_compressed('labs_all_data1_rot_individual_nclusters60_sub5', lab_sub=lab_sub, cat_data=cat_data, lst=lst)
