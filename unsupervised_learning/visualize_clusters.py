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

PLOT_DIR = 'plots'


def load_data(path, filename):
    with open(os.path.join(path, filename), 'rb') as f:
        saved_data = pickle.load(f)
    return saved_data


def plot_silhouette_values(n_cluster, cluster_labels, sample_silhouette_values, silhouette_avg,ax):
    y_lower = 10
    for i in range(n_cluster):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_cluster)

        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax.set_title("The silhouette plot for the various clusters.")
    ax.set_xlabel("The silhouette coefficient values")
    ax.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])


def calc_pca(data):
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(data)
    pca1 = pca_result[:, 0]
    pca2 = pca_result[:, 1]
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    return pca1, pca2


def plot_pca_dimension_proj(pca1, pca2, n_cluster, cluster_label, ax, label=''):
    sns.scatterplot(
        x=pca1,
        y=pca2,
        hue=cluster_label,
        palette=sns.color_palette("hls", n_cluster),
        legend="full",
        alpha=0.5,
        ax=ax
    )
    ax.set_title('Cluster Projection to First 2 Principle Components ' + label)
    ax.set_xlabel('the 1st pc')
    ax.set_ylabel('the 2st pc')


if __name__ == "__main__":
    dataset_name = sys.argv[1]
    if dataset_name == 'campaign_marketing':
        # kmeans params
        selected_n_cluster = 4
        selected_kmeans_index = 1
        # em params
        selected_n_components_full = 8
        selected_n_index_full = 3
        selected_n_components_spherical = 18
        selected_n_index_spherical = 8
    elif dataset_name == 'university_recommendation':
        # kmeans params
        selected_n_cluster = 16
        selected_kmeans_index = 7
        # em params
        selected_n_components_full = 6
        selected_n_index_full = 2
        selected_n_components_spherical = 18
        selected_n_index_spherical = 8

    result = 'results/'
    filename = '{dataset_name}_{step}_{method}.pkl'
    data = load_data(path='results/',
                     filename=filename.format(dataset_name=dataset_name, step='clustering', method='cluster_input'))

    kmeans_metrics = load_data(path='results/',
                               filename=filename.format(dataset_name=dataset_name, step='clustering', method='kmeans'))

    em_metrics = load_data(path='results/',
                           filename=filename.format(dataset_name=dataset_name, step='clustering', method='em'))

    # visualize kmeans
    n_cluster = []
    cluster_labels = []
    avg_distance = []
    silhouette_avg_list = []
    silhouette_value_list = []
    for k, v in kmeans_metrics.items():
        n_cluster.append(k)
        avg_d = np.mean(np.min(v.get('distance'), axis=1))
        avg_distance.append(avg_d)
        silhouette_avg = silhouette_score(data, v.get('label'))
        sample_silhouette_values = silhouette_samples(data, v.get('label'))
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

    # after observe the plot, n_cluster=4 is decided to be the elbow
    # silhouette_score
    plot_silhouette_values(n_cluster=selected_n_cluster,
                           sample_silhouette_values=silhouette_value_list[selected_kmeans_index],
                           cluster_labels=cluster_labels[selected_kmeans_index],
                           silhouette_avg=silhouette_avg_list[selected_kmeans_index],
                           ax=ax[1])

    # project clusters on the PC dimenstions
    pca1, pca2 = calc_pca(data)
    plot_pca_dimension_proj(pca1=pca1, pca2=pca2, n_cluster=selected_n_cluster,
                            cluster_label=cluster_labels[selected_kmeans_index], ax=ax[2])
    fig.suptitle('Dataset: {dataset_name}'.format(dataset_name=dataset_name))
    plot_path = '{plot_dir}/{dataset_name}_{step}_{method}.png'
    plt.savefig(plot_path.format(plot_dir=PLOT_DIR, dataset_name=dataset_name, step='clustering', method='kmeans'))
    plt.close()

    # visualize EM
    # based on BIC, choose n_component = 8 for covariance_shape = full
    # based on BIC, choose n_component = 18 for covariance_shape = spherical
    n_components = np.array([])
    bics = np.array([])
    aics = np.array([])
    loglikelihoods = np.array([])
    cluster_labels = []
    cov_shapes = np.array([])

    for k, v in em_metrics.items():
        cov_shape = k.split('_')[0]
        cov_shapes = np.append(cov_shapes, cov_shape)
        n_components = np.append(n_components, int(k.split('_')[1]))
        bics = np.append(bics, v.get('bic'))
        aics = np.append(aics, v.get('bic'))
        loglikelihoods = np.append(loglikelihoods, v.get('loglikelihood'))
        cluster_labels.append(v.get('label'))
    cluster_labels = np.array(cluster_labels)

    fig, ax = plt.subplots(2, 2, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')

    p1 = ax[0, 0].plot(n_components[cov_shapes == 'full'], bics[cov_shapes == 'full'], color='tab:blue', label='BIC')
    ax2 = ax[0, 0].twinx()
    p2 = ax2.plot(n_components[cov_shapes == 'full'], loglikelihoods[cov_shapes == 'full'], color='tab:orange',
                  label='loglikelihood')
    p_all = p1 + p2
    labs = [l.get_label() for l in p_all]
    ax[0, 0].legend(p_all, labs)
    title = 'Information Criteria - {cov_shape}'
    ax[0, 0].set_title(title.format(cov_shape='full'))
    ax[0, 0].set_xlabel('n_components')
    ax[0, 0].set_ylabel('BIC')
    ax2.set_ylabel('loglikelihood')

    plot_pca_dimension_proj(pca1=pca1, pca2=pca2, n_cluster=selected_n_components_full,
                            cluster_label=cluster_labels[cov_shapes == 'full'][selected_n_index_full],
                            ax=ax[0, 1], label='- full')

    cov_shape = 'spherical'

    # visualize the cluster size
    p1 = ax[1, 0].plot(n_components[cov_shapes == cov_shape], bics[cov_shapes == cov_shape], color='tab:blue',
                       label='BIC')
    ax2 = ax[1, 0].twinx()
    p2 = ax2.plot(n_components[cov_shapes == cov_shape], loglikelihoods[cov_shapes == cov_shape], color='tab:orange',
                  label='loglikelihood')
    p_all = p1 + p2
    labs = [l.get_label() for l in p_all]
    ax[1, 0].legend(p_all, labs)
    title = 'Information Criteria - {cov_shape}'
    ax[1, 0].set_title(title.format(cov_shape=cov_shape))
    ax[1, 0].set_xlabel('n_components')
    ax[1, 0].set_ylabel('BIC')
    ax2.set_ylabel('loglikelihood')

    plot_pca_dimension_proj(pca1=pca1, pca2=pca2, n_cluster=selected_n_components_spherical,
                            cluster_label=cluster_labels[cov_shapes == cov_shape][selected_n_index_spherical],
                            ax=ax[1, 1], label='- {cov_shape}'.format(cov_shape=cov_shape))

    fig.suptitle('Dataset: {dataset_name}'.format(dataset_name=dataset_name))
    plot_path = '{plot_dir}/{dataset_name}_{step}_{method}.png'
    plt.savefig(plot_path.format(plot_dir=PLOT_DIR, dataset_name=dataset_name, step='clustering', method='em'))
    plt.close()
