# %%
"""
Created on Tue Sep  6 00:08:31 2016

@author: ajoshi
"""
from fmri_methods_sipi import interpolate_labels
from dfsio import readdfs, writedfs
import time
import scipy as sp
import nibabel as nib


left_mid = readdfs('/home/ajoshi/data/BCI-DNI_brain_atlas/BCI-DNI_brain.left.\
mid.cortex_refined_labs.dfs')
left_mid1 = readdfs('/home/ajoshi/data/BCI-DNI_brain_atlas/BCI-DNI_brain.left.\
mid.cortex.dfs')
left_mid.vertices = left_mid1.vertices

left_inner = readdfs('/home/ajoshi/data/BCI-DNI_brain_atlas/BCI-DNI_brain\
.left.inner.cortex.dfs')
right_inner = readdfs('/home/ajoshi/data/BCI-DNI_brain_atlas/BCI-DNI_brain\
.right.inner.cortex.dfs')
left_pial = readdfs('/home/ajoshi/data/BCI-DNI_brain_atlas/BCI-DNI_brain\
.left.pial.cortex.dfs')
right_pial = readdfs('/home/ajoshi/data/BCI-DNI_brain_atlas/BCI-DNI_brain\
.right.pial.cortex.dfs')

right_mid = readdfs('/home/ajoshi/data/BCI-DNI_brain_atlas/BCI-DNI_brain.right.\
mid.cortex_refined_labs.dfs')
right_mid1 = readdfs('/home/ajoshi/data/BCI-DNI_brain_atlas/BCI-DNI_brain.right.\
mid.cortex.dfs')
right_mid.vertices = right_mid1.vertices

r1_vert = (right_pial.vertices + right_mid.vertices)/2.0
r2_vert = (right_inner.vertices + right_mid.vertices)/2.0
l1_vert = (left_pial.vertices + left_mid.vertices)/2.0
l2_vert = (left_inner.vertices + left_mid.vertices)/2.0

vol_lab = nib.load('/home/ajoshi/data/BCI-DNI_brain_atlas/\
BCI-DNI_brain.label.nii.gz')
vol_img = vol_lab.get_data()

xres = vol_lab.header['pixdim'][1]
yres = vol_lab.header['pixdim'][2]
zres = vol_lab.header['pixdim'][3]

X, Y, Z = sp.meshgrid(sp.arange(vol_lab.shape[0]), sp.arange(vol_lab.shape[1]),
                      sp.arange(vol_lab.shape[2]), indexing='ij')

X = X*xres
Y = Y*yres
Z = Z*zres
#vol_img = sp.mod(vol_img, 1000)
ind = (vol_img >= 120) & (vol_img < 600)
Xc = X[ind]
Yc = Y[ind]
Zc = Z[ind]
v_lab = vol_img[ind]


class t:
    pass


class f:
    pass


t.vertices = sp.concatenate((Xc[:, None], Yc[:, None], Zc[:, None]), axis=1)
t.labels = sp.mod(v_lab,1000)
f.vertices = sp.concatenate((left_mid.vertices, right_mid.vertices,
                             left_inner.vertices, right_inner.vertices,
                             left_pial.vertices, right_pial.vertices,
                             l1_vert, r1_vert, l2_vert, r2_vert))

f.labels = sp.concatenate((left_mid.labels, right_mid.labels,
                           left_mid.labels, right_mid.labels,
                           left_mid.labels, right_mid.labels,
                           left_mid.labels, right_mid.labels,
                           left_mid.labels, right_mid.labels))

tic = time.time()
# t = interpolate_labels(f, t)
v_labels = t.labels
for labid in sp.unique(v_labels):
    indf = (sp.floor(f.labels/10.0) == labid)
    if sp.sum(indf) == 0:
        continue
    indt = (v_labels == labid)
    tree = sp.spatial.cKDTree(f.vertices[indf, :])
    d, inds = tree.query(t.vertices[indt, :], k=1, p=2)
    t.labels[indt] = f.labels[indf][inds]
    toc = time.time()
    print 'Time Elapsed = %f sec.' % (toc - tic)
# here make sure that hanns labels are not modified TBD
vol_img[ind] = t.labels

new_img = nib.Nifti1Image(vol_img, vol_lab.affine)
nib.save(new_img, '/home/ajoshi/data/BCI-DNI_brain_atlas/\
BCI-DNI_brain.refined.label.nii.gz')

right_inner.labels = right_mid.labels
right_inner.vColor = right_mid.vColor
writedfs('/home/ajoshi/data/BCI-DNI_brain_atlas/BCI-DNI_brain\
.right.inner.cortex.refined.dfs', right_inner)

right_pial.labels = right_mid.labels
right_pial.vColor = right_mid.vColor
writedfs('/home/ajoshi/data/BCI-DNI_brain_atlas/BCI-DNI_brain\
.right.pial.cortex.refined.dfs', right_pial)

left_inner.labels = left_mid.labels
left_inner.vColor = left_mid.vColor
writedfs('/home/ajoshi/data/BCI-DNI_brain_atlas/BCI-DNI_brain\
.left.inner.cortex.refined.dfs', left_inner)

left_pial.labels = left_mid.labels
left_pial.vColor = left_mid.vColor
writedfs('/home/ajoshi/data/BCI-DNI_brain_atlas/BCI-DNI_brain\
.left.pial.cortex.refined.dfs', left_pial)

writedfs('/home/ajoshi/data/BCI-DNI_brain_atlas/BCI-DNI_brain\
.left.mid.cortex.refined.dfs', left_mid)

writedfs('/home/ajoshi/data/BCI-DNI_brain_atlas/BCI-DNI_brain\
.right.mid.cortex.refined.dfs', right_mid)
