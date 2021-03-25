import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import pickle
import os
import sys
import numpy as np
import seaborn as sns
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from visualize_clusters import *

PLOT_DIR = 'plots'


def plot_kmeans_on_reduce_dim(dataset_name, reduce_method, selected_n_cluster, selected_kmeans_index):
    path = 'results/'
    clustering_method = "kmeans"
    filename = '{dataset_name}_{step}_{clustering_method}_{reduce_method}.pkl'
    metrics = load_data(path=path, filename=filename.format(dataset_name=dataset_name,
                                                            step='clustering_on_reduced',
                                                            clustering_method=clustering_method,
                                                            reduce_method=reduce_method))

    n_cluster = []
    cluster_labels = []
    avg_distance = []
    silhouette_avg_list = []
    silhouette_value_list = []
    for k, v in metrics.items():
        n_cluster.append(k)
        avg_d = np.mean(np.min(v.get('distance'), axis=1))
        avg_distance.append(avg_d)
        silhouette_avg = silhouette_score(v.get('data'), v.get('label'))
        sample_silhouette_values = silhouette_samples(v.get('data'), v.get('label'))
        silhouette_avg_list.append(silhouette_avg)
        silhouette_value_list.append(sample_silhouette_values)
        cluster_labels.append(v.get('label'))

    fig, ax = plt.subplots(1, 3, figsize=(24, 6), dpi=80, facecolor='w', edgecolor='k')
    # visualize the cluster size
    p1 = ax[0].plot(n_cluster, avg_distance, color='tab:blue', label='averge distance')
    ax2 = ax[0].twinx()
    p2 = ax2.plot(n_cluster, silhouette_avg_list, color='tab:orange', label='silhouette score')
    p_all = p1 + p2
    labs = [l.get_label() for l in p_all]
    ax[0].legend(p_all, labs)
    ax[0].set_title("Decide a Proper # of Clusters")
    ax[0].set_xlabel('n_clusters')
    ax[0].set_ylabel('euclidean')
    ax2.set_ylabel('silhouette score')

    plot_silhouette_values(n_cluster=selected_n_cluster,
                           sample_silhouette_values=silhouette_value_list[selected_kmeans_index],
                           cluster_labels=cluster_labels[selected_kmeans_index],
                           silhouette_avg=silhouette_avg_list[selected_kmeans_index],
                           ax=ax[1])

    # project clusters on the PC dimenstions
    pca1, pca2 = calc_pca(list(metrics.values())[selected_kmeans_index].get('data'))
    plot_pca_dimension_proj(pca1=pca1, pca2=pca2, n_cluster=selected_n_cluster,
                            cluster_label=cluster_labels[selected_kmeans_index], ax=ax[2])

    fig.suptitle('Dataset: {dataset_name}'.format(dataset_name=dataset_name))
    plot_path = '{plot_dir}/{dataset_name}_{step}_{method}_{reduce_method}.png'
    plt.savefig(plot_path.format(plot_dir=PLOT_DIR,
                                 dataset_name=dataset_name,
                                 step='clustering_on_reduced',
                                 method='kmeans',
                                 reduce_method=reduce_method))
    plt.close()


def plot_em_on_reduce_dim(dataset_name, reduce_method, selected_n_components, selected_index):
    path = 'results/'
    clustering_method = "em"
    filename = '{dataset_name}_{step}_{clustering_method}_{reduce_method}.pkl'
    cov_shape = 'full'
    metrics = load_data(path=path, filename=filename.format(dataset_name=dataset_name,
                                                            step='clustering_on_reduced',
                                                            clustering_method=clustering_method,
                                                            reduce_method=reduce_method))

    n_components = np.array([])
    bics = np.array([])
    loglikelihoods = np.array([])
    cluster_labels = []

    for k, v in metrics.items():
        n_components = np.append(n_components, k)
        bics = np.append(bics, v.get('bic'))
        loglikelihoods = np.append(loglikelihoods, v.get('loglikelihood'))
        cluster_labels.append(v.get('label'))

    cluster_labels = np.array(cluster_labels)
    fig, ax = plt.subplots(1, 2, figsize=(16, 6), dpi=80, facecolor='w', edgecolor='k')
    # visualize the cluster size
    p1 = ax[0].plot(n_components, bics, color='tab:blue',
                    label='BIC')
    ax2 = ax[0].twinx()
    p2 = ax2.plot(n_components, loglikelihoods, color='tab:orange',
                  label='loglikelihood')
    p_all = p1 + p2
    labs = [l.get_label() for l in p_all]
    ax[0].legend(p_all, labs)
    title = 'Information Criteria - {cov_shape}'
    ax[0].set_title(title.format(cov_shape=cov_shape))
    ax[0].set_xlabel('n_components')
    ax[0].set_ylabel('BIC')
    ax2.set_ylabel('loglikelihood')
    pca1, pca2 = calc_pca(list(metrics.values())[selected_index].get('data'))
    plot_pca_dimension_proj(pca1=pca1, pca2=pca2, n_cluster=selected_n_components,
                            cluster_label=cluster_labels[selected_index],
                            ax=ax[1], label='- {cov_shape}'.format(cov_shape=cov_shape))

    fig.suptitle('Dataset: {dataset_name}'.format(dataset_name=dataset_name))
    plot_path = '{plot_dir}/{dataset_name}_{step}_{method}_{reduce_method}.png'
    plt.savefig(plot_path.format(plot_dir=PLOT_DIR,
                                 dataset_name=dataset_name,
                                 step='clustering_on_reduced',
                                 method='em',
                                 reduce_method=reduce_method))
    plt.close()


if __name__ == "__main__":
    dataset_name = sys.argv[1]
    if dataset_name == 'campaign_marketing':
        plot_kmeans_on_reduce_dim(dataset_name=dataset_name,
                                  reduce_method='pca',
                                  selected_n_cluster=4,
                                  selected_kmeans_index=1)

        plot_kmeans_on_reduce_dim(dataset_name=dataset_name,
                                  reduce_method='ica',
                                  selected_n_cluster=4,
                                  selected_kmeans_index=1)

        plot_kmeans_on_reduce_dim(dataset_name=dataset_name,
                                  reduce_method='rca',
                                  selected_n_cluster=4,
                                  selected_kmeans_index=1)

        plot_kmeans_on_reduce_dim(dataset_name=dataset_name,
                                  reduce_method='dt',
                                  selected_n_cluster=4,
                                  selected_kmeans_index=1)
        plot_em_on_reduce_dim(dataset_name=dataset_name,
                              reduce_method='pca',
                              selected_n_components=4,
                              selected_index=1)

        plot_em_on_reduce_dim(dataset_name=dataset_name,
                              reduce_method='ica',
                              selected_n_components=4,
                              selected_index=1)

        plot_em_on_reduce_dim(dataset_name=dataset_name,
                              reduce_method='rca',
                              selected_n_components=4,
                              selected_index=1)

        plot_em_on_reduce_dim(dataset_name=dataset_name,
                              reduce_method='dt',
                              selected_n_components=4,
                              selected_index=1)
    elif dataset_name == 'university_recommendation':
        plot_kmeans_on_reduce_dim(dataset_name=dataset_name,
                                  reduce_method='pca',
                                  selected_n_cluster=10,
                                  selected_kmeans_index=4)

        plot_kmeans_on_reduce_dim(dataset_name=dataset_name,
                                  reduce_method='ica',
                                  selected_n_cluster=18,
                                  selected_kmeans_index=8)

        plot_kmeans_on_reduce_dim(dataset_name=dataset_name,
                                  reduce_method='rca',
                                  selected_n_cluster=6,
                                  selected_kmeans_index=2)

        plot_kmeans_on_reduce_dim(dataset_name=dataset_name,
                                  reduce_method='dt',
                                  selected_n_cluster=8,
                                  selected_kmeans_index=3)

        plot_em_on_reduce_dim(dataset_name=dataset_name,
                              reduce_method='pca',
                              selected_n_components=8,
                              selected_index=3)

        plot_em_on_reduce_dim(dataset_name=dataset_name,
                              reduce_method='ica',
                              selected_n_components=4,
                              selected_index=1)

        plot_em_on_reduce_dim(dataset_name=dataset_name,
                              reduce_method='rca',
                              selected_n_components=14,
                              selected_index=6)

        plot_em_on_reduce_dim(dataset_name=dataset_name,
                              reduce_method='dt',
                              selected_n_components=14,
                              selected_index=6)
    else:
        raise ValueError('Invalid dataset name!')
