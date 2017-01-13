# ||AUM||
import scipy.io
import scipy as sp
import numpy as np
from fmri_methods_sipi import rot_sub_data
from surfproc import view_patch_vtk, patch_color_attrib
from dfsio import readdfs
import os
import nibabel as nib
import matplotlib.pyplot as plt

p_dir = '/big_disk/ajoshi/HCP_data/data'
p_dir_ref = '/big_disk/ajoshi/HCP_data'
lst = os.listdir(p_dir)

r_factor = 3
ref_dir = os.path.join(p_dir_ref, 'reference')
nClusters = 30

ref = '100307'
print(ref + '.reduce' + str(r_factor) + '.LR_mask.mat')
fn1 = ref + '.reduce' + str(r_factor) + '.LR_mask.mat'
fname1 = os.path.join(ref_dir, fn1)
msk = scipy.io.loadmat(fname1)  # h5py.File(fname1);
dfs_left = readdfs(os.path.join(p_dir_ref, 'reference', ref + '.aparc.\
a2009s.32k_fs.reduce3.left.dfs'))
dfs_left_sm = readdfs(os.path.join(p_dir_ref, 'reference', ref + '.aparc.\
a2009s.32k_fs.reduce3.very_smooth.left.dfs'))
count1 = 0
rho_rho = []
rho_all = []
# lst=lst[:1]
labs_all = sp.zeros((len(dfs_left.labels), len(lst)))

sub = lst[0]
data = scipy.io.loadmat(os.path.join(p_dir, sub, sub + '.rfMRI_REST1_LR.\
reduce3.ftdata.NLM_11N_hvar_25.mat'))
LR_flag = msk['LR_flag']
LR_flag = np.squeeze(LR_flag) != 0
data = data['ftdata_NLM']

#vlang=nib.load('/big_disk/ajoshi/HCP5/' + '100307' + '/MNINonLinear/Results/tfMRI_LANGUAGE_LR/tfMRI_LANGUAGE_LR_Atlas.dtseries.nii')    
#LR_flag = msk['LR_flag']
#LR_flag = np.squeeze(LR_flag) > 0
#data = sp.squeeze(vlang.get_data()).T

# Length of window
WinT = 20

dist = sp.zeros(data.shape[1]+1-WinT)
for t in sp.arange(0, data.shape[1]+1-WinT):
    temp = data[LR_flag, t:(t+WinT)]
    m = np.mean(temp, 1)
    temp = temp - m[:, None]
    s = np.std(temp, 1)+1e-16
    temp = temp/s[:, None]
    d = temp
    if t == 0:
        d_ref = d

    drot = rot_sub_data(ref=d_ref, sub=d)
    dist[t] = sp.linalg.norm(drot-d_ref)
    d_ref = d
    print t, dist[t]

plt.plot(dist)
plt.ylabel('$L^2$ dist')
plt.xlabel('time samples')
plt.savefig('Rest_L1_windowed.png',dpi=200)
plt.show()
plt.draw()
