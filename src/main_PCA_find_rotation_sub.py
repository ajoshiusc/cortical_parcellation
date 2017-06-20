# ||AUM||
import scipy.io
import scipy as sp
import numpy as np
from dfsio import readdfs, writedfs
from mayavi import mlab
from sklearn.decomposition import PCA
from fmri_methods_sipi import rot_sub_data
#import h5py
import os
from surfproc import view_patch, view_patch_vtk, get_cmap, smooth_patch, patch_color_attrib, smooth_surf_function
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import DictionaryLearning
from scipy.stats import trim_mean
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.metrics import silhouette_score
p_dir = '/big_disk/ajoshi/HCP_data/data'
p_dir_ref='/big_disk/ajoshi/HCP_data'
lst = os.listdir(p_dir)
r_factor = 3
ref_dir = os.path.join(p_dir_ref, 'reference')
nClusters=2

ref = '100307'
sub = '100307' #'101309' #lst[25]
print sub, ref
print(ref + '.reduce' + str(r_factor) + '.LR_mask.mat')
fn1 = ref + '.reduce' + str(r_factor) + '.LR_mask.mat'
fname1 = os.path.join(ref_dir, fn1)
msk = scipy.io.loadmat(fname1)  # h5py.File(fname1);
dfs_left = readdfs(os.path.join(p_dir_ref, 'reference', ref + '.aparc.a2009s.32k_fs.reduce3.left.dfs'))
dfs_left_sm = readdfs(os.path.join(p_dir_ref, 'reference', ref + '.aparc.a2009s.32k_fs.reduce3.very_smooth.left.dfs'))
count1 = 0
#roiregion=['motor','precuneus','temporal','cingulate','semato','visual']

#roilist = np.array([[29,69,70],[(30, 72, 9, 47)],[33,34,35,36,74],[6,7,8,9,10],[28],[(2,22,11,58,59,20,43,19,45)]])

#roilist= [2,6,22] # [30, 72, 9, 47,6,7,8,9,10] #pc
roilist= [29,9,7] ##This is in the paper
#roilist=[6,7,8,9,10] #cing

#roilist = np.array([[29],[7,8],[(22,45)]]) used this for
#ref=lst[11]
datasub = scipy.io.loadmat(os.path.join(p_dir, sub, sub + '.rfMRI_REST1_LR.reduce3.ftdata.NLM_11N_hvar_25.mat'))
dataref = scipy.io.loadmat(os.path.join(p_dir, ref, ref + '.rfMRI_REST1_RL.reduce3.ftdata.NLM_11N_hvar_25.mat'))

LR_flag = msk['LR_flag']
LR_flag = np.squeeze(LR_flag) > 0
data = dataref['ftdata_NLM']
sub1 = data[LR_flag, :]
m = np.mean(sub1, 1)
sub1 = sub1 - m[:, None]
s = np.std(sub1, 1)+1e-16
sub1 = sub1/(s[:, None]*sp.sqrt(1200))

data = datasub['ftdata_NLM']
sub2 = data[LR_flag, :]
m = np.mean(sub2, 1)
sub2 = sub2 - m[:, None]
s = np.std(sub2, 1)+1e-16
sub2 = sub2/(s[:, None]*sp.sqrt(1200))

msk_small_region = np.in1d(dfs_left.labels, roilist)
sub = sp.concatenate((sub1[msk_small_region,:], sub2[msk_small_region,:]), axis=0)
pca = PCA(n_components=3)
pca.fit(sub)

sub2_rot,_ = rot_sub_data(sub1, sub2)


sub1_3d = pca.transform(sub1)
sub2_3d = pca.transform(sub2)
sub2_rot_3d = pca.transform(sub2_rot)

print sub1.shape
sub1 = sub1_3d
sub2 = sub2_3d
sub2_rot = sub2_rot_3d
#sub1=sp.random.rand(sub1.shape[0],sub1.shape[1])-.5
#sub2=sp.random.rand(sub2.shape[0],sub2.shape[1])-.5
print sub1.shape
#
m = np.mean(sub1, 1)
#sub1 = sub1 - m[:, None]
sub1 = sub1/sp.sqrt(sp.sum(sub1**2,1))[:,None]
#
m = np.mean(sub2, 1)
#sub2 = sub2 - m[:, None]
#s = np.std(sub2, 1)+1e-16
sub2 = sub2/sp.sqrt(sp.sum(sub2**2,1))[:,None] #(s[:, None]*sp.sqrt(3))
#sub1 = sub1[msk_small_region, :]
#sub2 = sub2[msk_small_region, :]

sub2_rot = sub2_rot/sp.sqrt(sp.sum(sub2_rot**2,1))[:,None] #(s[:, None]*sp.sqrt(3))

# Create a sphere
r = 1
pi = np.pi
cos = np.cos
sin = np.sin
phi, theta = np.mgrid[0:pi:200j, 0:2 * pi:200j]

x = r * sin(phi) * cos(theta)
y = r * sin(phi) * sin(theta)
z = r * cos(phi)

mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 300))
#mlab.clf()
clr=[[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1]]
# Represent spherical harmonics on the surface of the sphere
mlab.mesh(x, y, z, color=(0.5, 0.5, 0.5), opacity=1)
for ind in range(len(roilist)):
    msk_roi=np.in1d(dfs_left.labels, roilist[ind])
    mlab.points3d(sub1[msk_roi,0], sub1[msk_roi,1], sub1[msk_roi,2], scale_factor=0.05, color=tuple(clr[ind]))

mlab.figure(2, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 300))
## Represent spherical harmonics on the surface of the sphere
mlab.mesh(x, y, z, color=(0.5, 0.5, 0.5), opacity=1)
for ind in range(len(roilist)):
    msk_roi=np.in1d(dfs_left.labels, roilist[ind])
    mlab.points3d(sub2[msk_roi,0], sub2[msk_roi,1], sub2[msk_roi,2], scale_factor=0.05, color=tuple(clr[ind]))
#
# mlab.points3d(sub2[:,0], sub2[:,1], sub2[:,2], scale_factor=0.05, color=(0, 1, 0))
#mlab.show()


mlab.figure(3, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 300))
## Represent spherical harmonics on the surface of the sphere
mlab.mesh(x, y, z, color=(0.5, 0.5, 0.5), opacity=1)
for ind in range(len(roilist)):
    msk_roi=np.in1d(dfs_left.labels, roilist[ind])
    mlab.points3d(sub2_rot[msk_roi,0], sub2_rot[msk_roi,1], sub2_rot[msk_roi,2], scale_factor=0.05, color=tuple(clr[ind]))
#
# mlab.points3d(sub2[:,0], sub2[:,1], sub2[:,2], scale_factor=0.05, color=(0, 1, 0))
mlab.draw()
mlab.savefig('sph3.png')

mlab.show()

dfs_left_sm.vColor = sp.zeros(dfs_left_sm.vertices.shape)+0.5
view_patch(dfs_left_sm, close=0)
sub_vert = dfs_left_sm.vertices
for ind in range(len(roilist)):
    msk_roi=np.in1d(dfs_left.labels, roilist[ind])
    mlab.points3d(sub_vert[msk_roi,0], sub_vert[msk_roi,1], sub_vert[msk_roi,2], scale_factor=5, color=tuple(clr[ind]))

#
#
##    msk_small_region = (dfs_left.labels == 30) | (dfs_left.labels == 72) | (dfs_left.labels == 9) |  (dfs_left.labels == 47)  # % motor
#d = sub1 # [msk_small_region, :]
#
#ref_mean_pc = sp.mean(d,axis=0)
#ref_mean_pc=ref_mean_pc-sp.mean(ref_mean_pc)
#ref_mean_pc=ref_mean_pc/(sp.std(ref_mean_pc))
#
#rho = np.dot(ref_mean_pc,sub1.T)/ref_mean_pc.shape[0]
#rho[~np.isfinite(rho)] = 0
#
#simil_mtx=sp.pi/2.0 + sp.arcsin(rho)
##    simil_mtx=0.3*sp.ones(rho.shape)
##SC = SpectralClustering(n_clusters=nClusters, affinity='precomputed')
##labs_all = SC.fit_predict(simil_mtx)+1
#
#dfs_left_sm.attributes=sp.zeros(len(dfs_left_sm.vertices))
## rho = smooth_surf_function(dfs_left_sm, rho)
#dfs_left_sm.attributes[msk_small_region] = rho
#dfs_left_sm=patch_color_attrib(dfs_left_sm, clim=[0,1/sp.sqrt(3)])
#view_patch_vtk(dfs_left_sm, azimuth=90,elevation=180, roll=90, outfile='sub1to1_view1_pc.png', show=1)
#view_patch_vtk(dfs_left_sm, azimuth=-90,elevation=-180, roll=-90, outfile='sub1to1_view2_pc.png', show=1)
#
#
#rho = np.dot(ref_mean_pc,sub2.T)/ref_mean_pc.shape[0]
#rho[~np.isfinite(rho)] = 0
## rho = smooth_surf_function(dfs_left_sm, rho)
#dfs_left.attributes=dfs_left_sm.attributes
#dfs_left.attributes[msk_small_region] = rho
#dfs_left_sm=patch_color_attrib(dfs_left_sm, clim=[0,1/sp.sqrt(3)])
#view_patch_vtk(dfs_left_sm, azimuth=90,elevation=180, roll=90, outfile='sub1to2_view1_pc.png', show=1)
#view_patch_vtk(dfs_left_sm, azimuth=-90,elevation=-180, roll=-90, outfile='sub1to2_view2_pc.png', show=1)
#
##sm=smooth_patch(dfs_left,iter=1000)
##view_patch(sm)
##view_patch(dfs_left,rho)
#
#sub_rot = rot_sub_data(sub1, sub2)
#
#rho = sp.dot(ref_mean_pc,sub_rot.T)/ref_mean_pc.shape[0]
##rho=rho[0,1:]
#rho[~np.isfinite(rho)] = 0
## rho = smooth_surf_function(dfs_left_sm, rho)
#dfs_left.attributes[msk_small_region] = rho
#dfs_left_sm=patch_color_attrib(dfs_left_sm, clim=[0,1/sp.sqrt(3)])
#view_patch_vtk(dfs_left_sm, azimuth=90,elevation=180, roll=90, outfile='sub1to2_view1_pc_rot.png', show=1)
#view_patch_vtk(dfs_left_sm, azimuth=-90,elevation=-180, roll=-90, outfile='sub1to2_view2_pc_rot.png', show=1)
