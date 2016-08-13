
####################################################################
# First we load the ADHD200 data
from nilearn import datasets
import scipy as sp
from fmri_methods_sipi import rot_sub_data, show_slices
import matplotlib.pyplot as plt
from nilearn import input_data
from nilearn.plotting import plot_roi, plot_epi, show
from nilearn.image.image import mean_img
from nilearn.image import index_img


adhd_dataset = datasets.fetch_adhd(n_subjects=30)
func_filenames = adhd_dataset.func  # list of 4D nifti files for each subject
print('First functional nifti image (4D) is at: %s' %
      func_filenames[0])  # 4D data
nifti_masker = input_data.NiftiMasker(standardize=True, mask_strategy='epi',
                           memory="nilearn_cache", memory_level=2,
                           smoothing_fwhm=6*4, detrend=True)
# Compute EPI based mask 
nifti_masker.fit(func_filenames[0])

mask_img = nifti_masker.mask_img_
mean_func_img = mean_img(func_filenames[0])
plot_roi(mask_img, mean_func_img, display_mode='y', cut_coords=4, title="Mask")
plt.show()
# Load 75 time points per volume
nii_img = index_img(func_filenames[0], slice(0, 75))
ref_mskd = nifti_masker.transform(nii_img).T
all_data=sp.zeros((ref_mskd.shape[0],ref_mskd.shape[1],len(func_filenames)))
all_data[:,:,0]=ref_mskd
all_data_orig=all_data.copy()
# Load all the data and rotate it
for ind in range(1,len(func_filenames)):
    nii_img = index_img(func_filenames[ind], slice(0, 75))
    sub_mskd = nifti_masker.transform(nii_img).T ##check fwhm
    temp = rot_sub_data(ref_mskd, sub_mskd)
    all_data[:,:,ind] = temp
    all_data_orig[:,:,ind] = sub_mskd
    print ind,
    

# Variance before rotation
var_before = sp.average(sp.var(all_data_orig,axis=2),axis=1)
var_before=nifti_masker.inverse_transform(var_before)
var_before=var_before.get_data()
show_slices([var_before[:,:,30],var_before[:,30,:],var_before[30,:,:]],vmax=1.0,vmin=0.0)


# variance after rotation
var_after = sp.average(sp.var(all_data,axis=2),axis=1)
var_after=nifti_masker.inverse_transform(var_after)
var_after=var_after.get_data()
show_slices([var_after[:,:,30],var_after[:,30,:],var_after[30,:,:]],vmax=1.0,vmin=0.0)

plt.show()

# Now lets work on group difference
    
