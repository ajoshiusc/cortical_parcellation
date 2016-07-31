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

cat_data1=sp.zeros((nSub*sub_data.shape[0]/2,sub_data.shape[1]))
cat_data2=sp.zeros((nSub*sub_data.shape[0]/2,sub_data.shape[1]))

for ind in range(nSub):
    sub_data[:,:,ind] = rot_sub_data(ref=sub_data[:,:,0],sub=sub_data[:,:,ind])
    if ind < nSub/2:
        cat_data1[sub_data.shape[0]*ind:sub_data.shape[0]*(ind+1),:] = sub_data[:,:,ind]    
    else:
        ind2=ind-nSub/2
        cat_data2[sub_data.shape[0]*ind2:sub_data.shape[0]*(ind2+1),:] = sub_data[:,:,ind]    
        

 
SC = KMeans(n_clusters=nClusters,random_state=5324)
labs_all1 = SC.fit_predict(cat_data1)
labs_all2 = SC.fit_predict(cat_data2)

lab_sub1=labs_all1.reshape((sub_data.shape[0],nSub/2),order='F')
lab_sub2=labs_all2.reshape((sub_data.shape[0],nSub/2),order='F')

sp.savez_compressed('labs_all_split2_data1', lst=lst, lab_sub1=lab_sub1, lab_sub2=lab_sub2, cat_data1=cat_data1, cat_data2=cat_data2)
