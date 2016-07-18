from mayavi import mlab
from centroid import  all_separate, initialize, plot_graph, affinity_mat
import numpy as np
import scipy as sp
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA

r_all,vertices,faces,mask,rho,rho_1=initialize()
data_file = 'all_data_file'
sp.savez(data_file  + 'motor_rho.npz', rho=rho,mask=mask,vertices=vertices,faces=faces)
'''rho = np.load('all_data_filecingulate_RHO.npz')['rho']
mask = np.load('all_data_filecingulate_RHO.npz')['mask']
vertices = np.load('all_data_filecingulate_RHO.npz')['vertices']
faces = np.load('all_data_filecingulate_RHO.npz')['faces']'''
#pca = PCA()
#pca.fit_transform(rho)
#store1 = pca.explained_variance_ratio_.cumsum()
nCluster=3
#nCluster=107
SC = SpectralClustering(n_clusters=nCluster, affinity='precomputed')
labels = SC.fit_predict(rho)

label = np.zeros([vertices.shape[0]])
label[mask] = labels + 1
#new_cent = all_separate(label,vertices,nCluster)
mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], faces, representation='surface',
                     opacity=1, scalars=np.float64(label))
mlab.gcf().scene.parallel_projection = True
mlab.view(azimuth=0, elevation=90)
mlab.colorbar(orientation='vertical')
mlab.show()
mlab.close()