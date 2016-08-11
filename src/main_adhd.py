
import matplotlib.pyplot as plt
def show_slices(slices):
       """ Function to display row of image slices """
       fig, axes = plt.subplots(1, len(slices))
       for i, slice in enumerate(slices):
           axes[i].imshow(slice.T, cmap="gray", origin="lower")

####################################################################
# First we load the ADHD200 data
from nilearn import datasets, image
import nibabel as nib
from nilearn.datasets import load_mni152_brain_mask

adhd_dataset = datasets.fetch_adhd(n_subjects=30)
func_filenames = adhd_dataset.func  # list of 4D nifti files for each subject
mni_template = load_mni152_brain_mask()
# print basic information on the dataset
print('First functional nifti image (4D) is at: %s' %
      func_filenames[0])  # 4D data
for ind in range(2):#len(func_filenames)):
    anat_img = nib.load(func_filenames[ind])
    anat_img_data = anat_img.get_data()
    print anat_img_data.shape
    show_slices([mni_template.get_data()[28, :, :],
                 anat_img_data[:, 33, :,0],
                 anat_img_data[:, :, 28,0]])
plt.show()
    
