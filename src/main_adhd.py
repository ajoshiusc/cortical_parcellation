

####################################################################
# First we load the ADHD200 data
from nilearn import datasets, image
import nibabel as nib
from nilearn.datasets import load_mni152_brain_mask
from fmri_methods_sipi import show_slices
import matplotlib.pyplot as plt
from nilearn.masking import apply_mask
from nilearn.image import image

adhd_dataset = datasets.fetch_adhd(n_subjects=30)
func_filenames = adhd_dataset.func  # list of 4D nifti files for each subject
mask_img = load_mni152_brain_mask()
#msk_ind = (mni_template.get_data()>0)
# print basic information on the dataset
print('First functional nifti image (4D) is at: %s' %
      func_filenames[0])  # 4D data
for ind in range(2):#len(func_filenames)):
    NiftiMasker--< use this
    anat_img = image.smooth_img(func_filenames[ind],'fast')##check fwhm
    mask_img = image.resample_img(mask_img, target_affine=None)
    anat_img_data = anat_img.get_data()
    print anat_img_data.shape
    show_slices([mask_img.get_data()[28, :, :],
                 anat_img_data[:, 33, :,0],
                 anat_img_data[:, :, 28,0]])
    v=anat_img_data[:, 33, :,0]
#    masked_data = apply_mask(func_filenames[ind], mask_img)


plt.show()


    
