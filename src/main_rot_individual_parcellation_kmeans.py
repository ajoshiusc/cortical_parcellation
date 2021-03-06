# ||AUM||
import scipy.io
import scipy as sp
import numpy as np
from fmri_methods_sipi import rot_sub_data
from dfsio import readdfs
import os
from sklearn.cluster import KMeans

p_dir = '/big_disk/ajoshi/HCP_data/data'
p_dir_ref='/big_disk/ajoshi/HCP_data'
lst = os.listdir(p_dir)
#lst=lst[:5]
r_factor = 3
ref_dir = os.path.join(p_dir_ref, 'reference')
nClusters=100

ref = '196750'#'100307'
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
    data = scipy.io.loadmat(os.path.join(p_dir, sub, sub + '.rfMRI_REST1_LR.reduce3.ftdata.NLM_11N_hvar_25.mat'))
    LR_flag = msk['LR_flag']
    LR_flag = np.squeeze(LR_flag) > 0
    data = data['ftdata_NLM']
    temp = data[LR_flag, :]
    m = np.mean(temp, 1)
    temp = temp - m[:,None]
    s = np.std(temp, 1)+1e-16
    temp = temp/s[:,None]
    d1 = temp

    data = scipy.io.loadmat(os.path.join(p_dir, sub, sub + '.rfMRI_REST2_LR.reduce3.ftdata.NLM_11N_hvar_25.mat'))
    LR_flag = msk['LR_flag']
    LR_flag = np.squeeze(LR_flag) > 0
    data = data['ftdata_NLM']
    temp = data[LR_flag, :]
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
#
#cat_data=sp.zeros((nSub*sub_data.shape[0],sub_data.shape[1]))
#
#for ind in range(nSub):
#    sub_data[:,:,ind] = rot_sub_data(ref=sub_data[:,:,0],sub=sub_data[:,:,ind])
#    cat_data[sub_data.shape[0]*ind:sub_data.shape[0]*(ind+1),:] = sub_data[:,:,ind]    
#    print ind, sub_data.shape, cat_data.shape

 
SC = KMeans(n_clusters=nClusters,random_state=5324)
lab_sub1=sp.zeros((sub_data1.shape[0],nSub))
lab_sub2=sp.zeros((sub_data2.shape[0],nSub))
for ind in range(nSub):
    lab_sub1[:,ind]=SC.fit_predict(sub_data1[:,:,ind])    
    lab_sub2[:,ind]=SC.fit_predict(sub_data2[:,:,ind])    
    print ind
#labs_all = SC.fit_predict(cat_data)

#lab_sub=labs_all.reshape((sub_data.shape[0],nSub),order='F')
sp.savez_compressed('labs_all_data_rot_individual_nclusters100', lab_sub1=lab_sub1, lab_sub2=lab_sub2, lst=lst)
