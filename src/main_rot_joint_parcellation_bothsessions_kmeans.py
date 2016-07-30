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
r_factor = 3
ref_dir = os.path.join(p_dir_ref, 'reference')
nClusters=30

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
    data1 = scipy.io.loadmat(os.path.join(p_dir, sub, sub + '.rfMRI_REST1_RL.reduce3.ftdata.NLM_11N_hvar_25.mat'))
    data2 = scipy.io.loadmat(os.path.join(p_dir, sub, sub + '.rfMRI_REST2_RL.reduce3.ftdata.NLM_11N_hvar_25.mat'))

    LR_flag = msk['LR_flag']
    LR_flag = np.squeeze(LR_flag) > 0
    data1 = data1['ftdata_NLM']
    data2 = data2['ftdata_NLM']

    temp = data1[LR_flag, :]
    m = np.mean(temp, 1)
    temp = temp - m[:,None]
    s = np.std(temp, 1)+1e-16
    temp = temp/s[:,None]
    d1 = temp
    temp = data2[LR_flag, :]
    m = np.mean(temp, 1)
    temp = temp - m[:,None]
    s = np.std(temp, 1)+1e-16
    temp = temp/s[:,None]
    d2 = temp

    
    if count1==0:        
        sub_data1 = sp.zeros((d1.shape[0],d1.shape[1],len(lst)))
        sub_data2 = sp.zeros((d2.shape[0],d2.shape[1],len(lst)))

    sub_data1[:,:,count1] = d1
    sub_data2[:,:,count1] = d2
 
    count1+=1
    print count1,
    
nSub=sub_data1.shape[2]

cat_data=sp.zeros((2*nSub*sub_data1.shape[0],sub_data1.shape[1]))

for ind in range(nSub):
    sub_data1[:,:,ind] = rot_sub_data(ref=sub_data1[:,:,0],sub=sub_data1[:,:,ind])
    sub_data2[:,:,ind] = rot_sub_data(ref=sub_data1[:,:,0],sub=sub_data2[:,:,ind])
    
    cat_data[sub_data1.shape[0]*ind:sub_data1.shape[0]*(ind+1),:] = sub_data1[:,:,ind]
    ind1=nSub+ind    
    cat_data[sub_data1.shape[0]*ind1:sub_data1.shape[0]*(ind1+1),:] = sub_data2[:,:,ind]    

    print ind, sub_data1.shape, cat_data.shape
sp.savez_compressed('data_bothsessions', cat_data=cat_data)

 
SC = KMeans(n_clusters=nClusters,random_state=5324)
labs_all = SC.fit_predict(cat_data)

lab_sub=labs_all.reshape((sub_data1.shape[0],2*nSub),order='F')
sp.savez_compressed('labs_all_data_bothsessions', lab_sub=lab_sub, cat_data=cat_data, lst=lst)
