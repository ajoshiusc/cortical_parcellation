# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 20:08:10 2016

@author: ajoshi
"""

from nilearn import datasets

adhd_dataset = datasets.fetch_adhd(n_subjects=30)
func_filenames = adhd_dataset.func  # list of 4D nifti files for each subject

# print basic information on the dataset
print('First functional nifti image (4D) is at: %s' %
      func_filenames[0])  # 4D data
      
from nilearn.decomposition import CanICA      

CanICA(hi)