

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
#mask_img = load_mni152_brain_mask()
#msk_ind = (mni_template.get_data()>0)
# print basic information on the dataset
print('First functional nifti image (4D) is at: %s' %
      func_filenames[0])  # 4D data
#nifti_masker = input_data.NiftiMasker(mask_img = mask_img)
nifti_masker = input_data.NiftiMasker(standardize=True, mask_strategy='epi',
                           memory="nilearn_cache", memory_level=2,
                           smoothing_fwhm=6*4, detrend=True)
nifti_masker.fit(func_filenames[0])

mask_img = nifti_masker.mask_img_
mean_func_img = mean_img(func_filenames[0])
plot_roi(mask_img, mean_func_img, display_mode='y', cut_coords=4, title="Mask")
plt.show()
nii_img = index_img(func_filenames[0], slice(0, 75))
ref_mskd = nifti_masker.transform(nii_img).T
#ref_mskd[~sp.isfinite(ref_mskd)]=1e-12
#ref_mskd = (1e-12+ref_mskd)/(1e-12+sp.sqrt(sp.sum(ref_mskd*ref_mskd,axis=1))[:,None])
all_data=sp.zeros((ref_mskd.shape[0],ref_mskd.shape[1],len(func_filenames)))
all_data[:,:,0]=ref_mskd
all_data_orig=all_data.copy()

for ind in range(1,len(func_filenames)):
    nii_img = index_img(func_filenames[ind], slice(0, 75))
    sub_mskd = nifti_masker.transform(nii_img).T ##check fwhm

 #   sub_mskd[~sp.isfinite(sub_mskd)]=1e-12
 #   sub_mskd = (1e-12+sub_mskd)/(1e-12+sp.sqrt(sp.sum(sub_mskd*sub_mskd,axis=1))[:,None])
    print ind, sub_mskd.shape
#    all_data[:,:,ind] 
    print 'before', sp.linalg.norm(ref_mskd-sub_mskd)
    temp = rot_sub_data(ref_mskd, sub_mskd)
#    plot_epi(mean_img(nifti_masker.inverse_transform(temp.T)),[10, 10, 10], vmin=-.10, vmax=.10)
    show()

    print 'after', sp.linalg.norm(ref_mskd-temp)
    all_data[:,:,ind] = temp
    all_data_orig[:,:,ind] = sub_mskd
    

var_before = sp.average(sp.var(all_data_orig,axis=2),axis=1)
var_before=nifti_masker.inverse_transform(var_before)
var_before=var_before.get_data()
show_slices([var_before[:,:,30],var_before[:,30,:],var_before[30,:,:]],vmax=1.0,vmin=0.0)



var_after = sp.average(sp.var(all_data,axis=2),axis=1)
var_after=nifti_masker.inverse_transform(var_after)
var_after=var_after.get_data()
show_slices([var_after[:,:,30],var_after[:,30,:],var_after[30,:,:]],vmax=1.0,vmin=0.0)


#    print ind, 



plt.show()


    
