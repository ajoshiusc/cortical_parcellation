
####################################################################
# First we load the ADHD200 data
from nilearn import datasets
import scipy as sp
from fmri_methods_sipi import rot_sub_data, hotelling_t2
import matplotlib.pyplot as plt
from nilearn import input_data
from nilearn.plotting import plot_roi, show, plot_stat_map
from nilearn.image.image import mean_img
from nilearn.image import index_img
from sklearn.decomposition import PCA
from scipy.stats import levene
from statsmodels.sandbox.stats.multicomp import multipletests
from nilearn.image import resample_to_img

std_msk=datasets.load_mni152_brain_mask()
gm_msk=datasets.fetch_icbm152_brain_gm_mask(threshold=0.3)
gm_msk.set_data_dtype(sp.float32)
affine_tar=1.0*sp.eye(4)
affine_tar[3,3]=.3

adhd_dataset = datasets.fetch_adhd()

func_filenames = adhd_dataset.func  # list of 4D nifti files for each subject

mean_func_img = mean_img(func_filenames[0])

gm_msk=resample_to_img(source_img=gm_msk, target_img=mean_func_img, interpolation='nearest')
                       #affine_tar, 
                    #interpolation='nearest', target_shape=sp.array([61,73,61]))
print('First functional nifti image (4D) is at: %s' %
      func_filenames[0])  # 4D data
nifti_masker = input_data.NiftiMasker(standardize=True, mask_img=gm_msk, #  mask_strategy='epi', #
                           memory="nilearn_cache", memory_level=2,
                           smoothing_fwhm=4, detrend=True)
# Compute EPI based mask 
nifti_masker.fit(func_filenames[0])

mask_img = nifti_masker.mask_img_
plot_roi(mask_img, mean_func_img, display_mode='y', cut_coords=4, title="Mask")
plt.show()
# Load 75 time points per volume
nii_img = index_img(func_filenames[0], slice(5, 75))
ref_mskd = nifti_masker.transform(nii_img).T
all_data=sp.zeros((ref_mskd.shape[0],ref_mskd.shape[1],len(func_filenames)))
all_data[:,:,0]=ref_mskd
all_data_orig=all_data.copy()
# Load all the data and rotate it
adhd_flag=sp.zeros(len(func_filenames))
adhd_flag[0]=adhd_dataset.phenotypic[0][22]
for ind in range(1,len(func_filenames)):
    nii_img = index_img(func_filenames[ind], slice(5, 75))
    sub_mskd = nifti_masker.transform(nii_img).T ##check fwhm
    temp = rot_sub_data(ref_mskd, sub_mskd)
    all_data[:,:,ind] = temp
    all_data_orig[:,:,ind] = sub_mskd
    adhd_flag[ind]=adhd_dataset.phenotypic[ind][22]
    print ind,
    

# Variance before rotation
var_before = sp.average(sp.var(all_data_orig,axis=2),axis=1)
var_before=nifti_masker.inverse_transform(var_before)
plot_stat_map(var_before,title='Variance before rotation')


# variance after rotation
var_after = sp.average(sp.var(all_data,axis=2),axis=1)
var_after=nifti_masker.inverse_transform(var_after)
plot_stat_map(var_after,title='Variance after rotation')


var_after_adhd = sp.average(sp.var(all_data[:,:,adhd_flag>0],axis=2),axis=1) 
var_after_normal = sp.average(sp.var(all_data[:,:,adhd_flag==0],axis=2),axis=1) 
var_after=var_after_adhd-var_after_normal
var_after=nifti_masker.inverse_transform(var_after)
plot_stat_map(nifti_masker.inverse_transform(var_after_adhd), title='Variance of adhd after rotation') #,vmax=0.45,
plot_stat_map(nifti_masker.inverse_transform(var_after_normal), title='Variance normals after rotation')
plot_stat_map(var_after,title='Variance Diff between adhd and normals after rotation')

show()
''' Dimensionality reduction using PCA'''

n_components = 5
P = PCA(n_components=n_components, whiten=False, copy=True)
P.fit(all_data[:,:,0])
all_data_pca=sp.zeros((all_data.shape[0],n_components,all_data.shape[2]))
print("Doing PCA on Synced data")
for ind in range(all_data.shape[2]):
    all_data_pca[:,:,ind]=P.transform(all_data[:,:,ind])
    print ind,

P.fit(all_data_orig[:,:,0])
all_data_orig_pca=sp.zeros((all_data_orig.shape[0],n_components,all_data_orig.shape[2]))
print("Doing PCA on Synced data")
for ind in range(all_data.shape[2]):
    all_data_orig_pca[:,:,ind]=P.transform(all_data_orig[:,:,ind])
    print ind,

''' Now lets work on group difference'''
print("Doing Group Diff using Hotelling test")
#pval,t2=hotelling_t2(all_data[:,:,adhd_flag>0].transpose(1,2,0),all_data[:,:,adhd_flag==0].transpose(1,2,0))
pval,t2 = hotelling_t2(all_data[:,:,adhd_flag==0].transpose(1,2,0),all_data[:,:,adhd_flag>0].transpose(1,2,0))
_, pval, _, _ = multipletests(pval,alpha=0.15, method='fdr_bh')
pval1=nifti_masker.inverse_transform((0.15-pval)*(pval<0.15))
plot_stat_map(pval1,title='Hotelling test after rotation')

pval,t2 = hotelling_t2(all_data_orig[:,:,adhd_flag==0].transpose(1,2,0),all_data_orig[:,:,adhd_flag>0].transpose(1,2,0))
_, pval, _, _ = multipletests(pval,alpha=0.15, method='fdr_bh')
pval1=nifti_masker.inverse_transform((0.15-pval)*(pval<0.15))
plot_stat_map(pval1,title='Hotelling test before rotation')

pval,t2 = hotelling_t2(all_data_orig_pca[:,:,adhd_flag==0].transpose(1,2,0),all_data_orig_pca[:,:,adhd_flag>0].transpose(1,2,0))
_, pval, _, _ = multipletests(pval,alpha=0.15, method='fdr_bh')
pval1=nifti_masker.inverse_transform((0.15-pval)*(pval<0.15))
plot_stat_map(pval1,title='Hotelling test before rotation (PCA)')


pval,t2 = hotelling_t2(all_data_pca[:,:,adhd_flag==0].transpose(1,2,0),all_data_pca[:,:,adhd_flag>0].transpose(1,2,0))
_, pval, _, _ = multipletests(pval,alpha=0.15, method='fdr_bh')
pval1=nifti_masker.inverse_transform((0.15-pval)*(pval<0.15))
plot_stat_map(pval1,title='Hotelling test after rotation (PCA)')


''' Reduce to 1 component and do f test'''
n_components = 1
P = PCA(n_components=n_components, whiten=False, copy=True)
P.fit(all_data[:,:,0])
all_data_pca = sp.zeros((all_data.shape[0],n_components,all_data.shape[2]))
print("Doing PCA on Synced data")
for ind in range(all_data.shape[2]):
    all_data_pca[:,:,ind]=P.transform(all_data[:,:,ind])
    print ind,

P.fit(all_data_orig[:,:,0])
all_data_orig_pca = sp.zeros((all_data_orig.shape[0],n_components,all_data_orig.shape[2]))
print("Doing PCA on Synced data")
for ind in range(all_data.shape[2]):
    all_data_orig_pca[:,:,ind] = P.transform(all_data_orig[:,:,ind])
    print ind,

pval = sp.zeros((all_data_pca.shape[0]))
for ind in range(all_data_pca.shape[0]):
    f,pval[ind] = levene(all_data_orig_pca[ind,:,adhd_flag==0].squeeze(), all_data_orig_pca[ind,:,adhd_flag>0].squeeze())
_, pval, _, _ = multipletests(pval,alpha=0.15, method='fdr_bh')
pval1 = nifti_masker.inverse_transform((0.15-pval.T)*(pval.T<0.15))
plot_stat_map(pval1,title='leven test before rotation (PCA)')

for ind in range(all_data_pca.shape[0]):
    f,pval[ind] = levene(all_data_pca[ind,:,adhd_flag==0].squeeze(), all_data_pca[ind,:,adhd_flag>0].squeeze())
_, pval, _, alpha_adj = multipletests(pval,alpha=0.15, method='fdr_bh')
pval1 = nifti_masker.inverse_transform((0.15-pval.T)*(pval.T<0.15))
plot_stat_map(pval1,title='leven test after rotation (PCA)')
    
show()
