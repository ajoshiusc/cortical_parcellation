import scipy as sp
from centroid import  all_separate, initialize, plot_graph, find_location_smallmask
import numpy as np
from mayavi import mlab
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.linear_model import LogisticRegression

nSubjects=40
training_data=np.array([])
training_labels=np.array([])
R_all,vertices,faces,mask,rho,rho_1=initialize()
nCluster=3
SC = SpectralClustering(n_clusters=nCluster, affinity='precomputed')
labels = SC.fit_predict(rho)
label=np.zeros(vertices.shape[0],dtype=float)
label[mask]=labels+1

temp_d = R_all[mask, :39*1200]
temp_rho = np.corrcoef(temp_d)
temp_rho[~np.isfinite(temp_rho)] = 0
temp_labels = SC.fit_predict(temp_rho)
temp_label=np.zeros(vertices.shape[0],dtype=float)
temp_label[mask]=temp_labels+1
mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], faces, representation='surface',
                     opacity=1, scalars=np.float64(temp_label))
mlab.gcf().scene.parallel_projection = True
mlab.view(azimuth=0, elevation=90)
mlab.colorbar(orientation='horizontal')
mlab.close()

all_centroid=all_separate(label,vertices,nCluster)
for i in range(0,nSubjects-1):
    d = R_all[mask, i*1200:i*1200+1200]
    rho = np.corrcoef(d)
    rho[~np.isfinite(rho)] = 0
    correlation_within_precuneus_vector=sp.array([])
    required_labels=sp.array([])
    for j in range(0,nCluster):
        correspondence_point = find_location_smallmask(vertices, all_centroid[j], mask)
        if j == 0:
            # change
            correlation_within_precuneus_vector = sp.array(rho[correspondence_point])
            required_labels=sp.array(labels[correspondence_point]+1)
        else:
            correlation_within_precuneus_vector = sp.vstack([correlation_within_precuneus_vector, rho[correspondence_point]])
            required_labels = sp.vstack([required_labels,(labels[correspondence_point] + 1)])
    if i == 0:
        training_data=sp.array(correlation_within_precuneus_vector)
        training_labels=sp.array(required_labels)
    else :
        training_data=sp.vstack([training_data,correlation_within_precuneus_vector])
        training_labels=sp.vstack([training_labels,required_labels])
clf = LogisticRegression(penalty='l2')
clf.fit(training_data,training_labels)
for i in range(0,4):
    data_file='data_file'+str(nCluster)+str(i)+'log.npz'
    temp = sp.array(np.load(data_file)['temp'])
    d = temp[mask, :]
    rho = np.corrcoef(d)
    rho[~np.isfinite(rho)] = 0
    sub_labels = clf.predict(rho)
    sub_label = np.zeros(vertices.shape[0], dtype=float)
    sub_label[mask] = sub_labels
    mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], faces, representation='surface',
                         opacity=1, scalars=np.float64(sub_label))
    mlab.gcf().scene.parallel_projection = True
    mlab.view(azimuth=0, elevation=90)
    mlab.colorbar(orientation='horizontal')
    mlab.close()
    data_file = 'data_file' + str(nCluster) + str(i) + 'logistic_labels.npz'
    sp.savez(data_file ,labels=sub_label)
    SC = SpectralClustering(n_clusters=nCluster, affinity='precomputed')
    labels = SC.fit_predict(rho)
    label = np.zeros(vertices.shape[0], dtype=float)
    label[mask] = labels + 1
    mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], faces, representation='surface',
                         opacity=1, scalars=np.float64(label))
    mlab.gcf().scene.parallel_projection = True
    mlab.view(azimuth=0, elevation=90)
    mlab.colorbar(orientation='horizontal')
    mlab.close()