from sklearn.metrics import silhouette_samples
from centroid import affinity_mat, all_separate, find_location_smallmask, change_order, change_labels
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import matplotlib as matplot
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import SpectralClustering

def decide_number_cluster():
    all_silhoutte_avg = []
    rho = np.load('all_data_filemotor_rho.npz')['rho']

    range_n_clusters = np.arange(6)
    range_n_clusters = range_n_clusters + 2

    for n_clusters in range_n_clusters:
        fig, (ax1) = plt.subplots(1, 1)
        fig.set_size_inches(18, 7)
        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(rho) + (n_clusters + 1) * 10])

        clusterer = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
        cluster_labels = clusterer.fit_predict(rho)

        silhouette_avg = silhouette_score(rho, cluster_labels)
        all_silhoutte_avg.append(silhouette_avg)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        sample_silhouette_values = silhouette_samples(rho, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhoutte score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.suptitle(("Silhouette analysis for Spectral clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

        plt.show()
        all_silhoutte_avg=np.array(all_silhoutte_avg)
        return all_silhoutte_avg.max()

def display(nClusters):
    from mayavi import mlab
    SC = SpectralClustering(n_clusters=nClusters, affinity='precomputed')
    labels = SC.fit_predict(rho)

    label = np.zeros([vertices.shape[0]])
    label[mask] = labels + 1
    # new_cent = all_separate(label,vertices,nCluster)
    cent = all_separate(label, vertices, nClusters)

    manual_order = np.array([0 for x in range(nClusters)])
    save = np.array([0 for x in range(nClusters)])

    for i in range(0, nClusters):
        if nClusters > 1:
            choose_vector = np.argmax(cent.transpose(), axis=1)
            save[i] = cent[choose_vector[1]][1]
            cent[choose_vector[1]][1] = -np.Inf
            manual_order[i] = choose_vector[1]

    manual_order = change_order(manual_order, nClusters)
    label = change_labels(label, manual_order, nClusters)

    temp=(label==4)
    label[temp]=0
    temp=label==3
    label[temp] = 0
    mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], faces, representation='surface',
                         opacity=1, scalars=np.float64(label))
    mlab.gcf().scene.parallel_projection = True
    mlab.view(azimuth=0, elevation=90)
    mlab.colorbar(orientation='vertical')
    mlab.show()
    return label>0

def reduce_temporal():
    temporal=display(4)
    R_all = np.load('all_data_filetemporal.npz')['R_all']
    d = R_all[temporal, :]
    rho = np.corrcoef(d)
    rho[~np.isfinite(rho)] = 0
    return affinity_mat(rho),temporal

all_silhoutte_avg = []
rho = np.load('all_data_filesemato.npz')['rho']
faces=np.load('all_data_filesemato.npz')['faces']

#check the corr_of_corr and fisher z-transformation
'''f_rho = np.arctanh(rho)
f_rho[~np.isfinite(f_rho)] = 0
rho = np.corrcoef(rho)
rho[~np.isfinite(rho)] = 0'''
#end

vertices=np.load('all_data_filesemato.npz')['vertices']
mask=np.load('all_data_filesemato.npz')['mask']
rho=np.arcsin(rho)

# only superior temporal gyrus to reduce the ROI region
#rho,temporal=reduce_temporal()
#end

# PCA plot
from sklearn.decomposition import PCA
pca = PCA()
pca.fit_transform(rho)
store1 = pca.explained_variance_ratio_.cumsum()
from centroid import plot_graph, affinity_mat, affinity_mat
print plot_graph(store1)
plt.rcParams.update({'font.size': 50})
#end

range_n_clusters = np.arange(10)
range_n_clusters = range_n_clusters + 2

for n_clusters in range_n_clusters:

    fig, (ax1) = plt.subplots(1, 1)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(rho) + (n_clusters + 1) * 10])

    clusterer = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
    cluster_labels = clusterer.fit_predict(rho)

    silhouette_avg = silhouette_score(rho, cluster_labels)
    all_silhoutte_avg.append(silhouette_avg)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    #CalinskiHarabaszEvaluation
    '''mean = np.mean(rho, axis=0)
    labels = np.zeros([vertices.shape[0]])
    labels[mask] = cluster_labels + 1
    centroid = all_separate(labels, vertices, n_clusters)
    centroids=np.array( [[0 for x in range(cluster_labels.shape[0])] for y in range(0,n_clusters)] ,dtype=float)
    for i in range(n_clusters):
        centroids[i] = rho[find_location_smallmask(vertices, centroid[i], mask)]
    B = np.sum([np.sum(labels == i) * (c - mean) ** 2 for i, c in enumerate(centroids)])
    W = np.sum([(x - centroids[labels[i]]) ** 2 for i, x in enumerate(rho)])
    c = len(centroids)
    n = len(rho)
    print ((n - c) * B) / 1.0 / ((c - 1) * W)'''
    #end

    sample_silhouette_values = silhouette_samples(rho, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhoutte score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                     ax1.get_xticklabels() + ax1.get_yticklabels()):
        item.set_fontsize(50)

    plt.suptitle(("Silhouette analysis for Spectral clustering "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=53, fontweight='bold')

    plt.show()
all_silhoutte_avg=np.array(all_silhoutte_avg)
from centroid import plot_graph_1
#plot graph for avg_silhouette scores for different number of cluster
plot_graph_1(all_silhoutte_avg)