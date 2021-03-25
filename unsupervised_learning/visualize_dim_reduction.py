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
from visualize_clusters import load_data

PLOT_DIR = 'plots'
RESUTL_PATH = 'results/'

if __name__ == "__main__":
    dataset_name = sys.argv[1]

    filename = '{dataset_name}_{step}_{method}.pkl'
    pca = load_data(path=RESUTL_PATH, filename=filename.format(dataset_name=dataset_name,
                                                               step='dim_reduction',
                                                               method='pca'))
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    # visualize the pc explained variance
    N = len(pca.get('explained_variance_ratio'))
    p1 = ax.plot(list(range(N)), np.cumsum(pca.get('explained_variance_ratio')),
                 color='tab:blue', label='explained_variance_ratio')
    ax2 = ax.twinx()
    p2 = ax2.plot(list(range(N)), pca.get('singular_values'),
                  color='tab:orange', label='Eigenvalues')
    ax.hlines(0.9, xmin=0, xmax=(N - 1), linestyles='dashed', color='tab:grey')
    p_all = p1 + p2
    labs = [l.get_label() for l in p_all]
    ax.legend(p_all, labs)
    ax.set_title("{dataset_name}: Explained Variance vs # of Components".format(dataset_name=dataset_name))
    ax.set_xlabel('n_components')
    ax.set_ylabel('explained_variance_ratio')
    ax2.set_ylabel('Eigenvalues')
    plot_path = '{plot_dir}/{dataset_name}_{step}_{method}.png'
    plt.savefig(plot_path.format(plot_dir=PLOT_DIR, dataset_name=dataset_name, step='dim_reduction', method='pca'))
    plt.close()

    # ica
    ica = load_data(path=RESUTL_PATH,
                    filename=filename.format(dataset_name=dataset_name, step='dim_reduction', method='ica'))
    n_components_list = [k for k, v in ica.items()]
    dt_roc_list = [v.get('dt_roc_auc') for k,v in ica.items()]
    dt_logloss_list = [v.get('dt_logloss') for k,v in ica.items()]
    logistic_roc_list = [v.get('logistic_roc_auc') for k,v in ica.items()]
    logistic_logloss_list = [v.get('logistic_logloss') for k,v in ica.items()]
    fig, ax = plt.subplots(1, 2, figsize=(16, 6), dpi=80, facecolor='w', edgecolor='k')
    ax[0].plot(n_components_list, logistic_roc_list, color='tab:blue', label='logistic regression')
    ax[0].plot(n_components_list, dt_roc_list, color='tab:orange', label='random forest')
    ax[0].set_title("ROC AUC on Validation Set")
    ax[0].set_xlabel('n_components')
    ax[0].set_ylabel('auc')
    ax[0].legend()

    ax[1].plot(n_components_list, logistic_logloss_list, color='tab:blue', label='logistic regression')
    ax[1].plot(n_components_list, dt_logloss_list, color='tab:orange', label='random forest')
    ax[1].set_title("logloss on Validation Set")
    ax[1].set_xlabel('n_components')
    ax[1].set_ylabel('logloss')
    ax[1].legend()
    fig.suptitle('Dataset: {dataset_name}'.format(dataset_name=dataset_name))
    plot_path = '{plot_dir}/{dataset_name}_{step}_{method}.png'
    plt.savefig(plot_path.format(plot_dir=PLOT_DIR, dataset_name=dataset_name, step='dim_reduction', method='ica'))
    plt.close()

    # rca
    fig, ax = plt.subplots(1, 4, figsize=(32, 6), dpi=80, facecolor='w', edgecolor='k')
    rca_sparse = load_data(path=RESUTL_PATH, filename=filename.format(dataset_name=dataset_name, step='dim_reduction', method='rca_sparse'))
    rca_gaussian = load_data(path=RESUTL_PATH, filename=filename.format(dataset_name=dataset_name, step='dim_reduction', method='rca_gaussian'))

    n_components_list = [k for k, v in rca_sparse.items()]
    dt_roc_list = [v.get('dt_roc_auc') for k,v in rca_sparse.items()]
    dt_logloss_list = [v.get('dt_logloss') for k,v in rca_sparse.items()]
    logistic_roc_list = [v.get('logistic_roc_auc') for k,v in rca_sparse.items()]
    logistic_logloss_list = [v.get('logistic_logloss') for k,v in rca_sparse.items()]

    ax[0].plot(n_components_list, logistic_roc_list, color='tab:blue', label='logistic regression')
    ax[0].plot(n_components_list, dt_roc_list, color='tab:orange', label='random forest')
    ax[0].set_title("ROC AUC on Validation Set (Sparse)")
    ax[0].set_xlabel('n_components')
    ax[0].set_ylabel('auc')
    ax[0].legend()

    ax[1].plot(n_components_list, logistic_logloss_list, color='tab:blue', label='logistic regression')
    ax[1].plot(n_components_list, dt_logloss_list, color='tab:orange', label='random forest')
    ax[1].set_title("logloss on Validation Set (Sparse)")
    ax[1].set_xlabel('n_components')
    ax[1].set_ylabel('logloss')
    ax[1].legend()

    n_components_list = [k for k, v in rca_gaussian.items()]
    dt_roc_list = [v.get('dt_roc_auc') for k,v in rca_gaussian.items()]
    dt_logloss_list = [v.get('dt_logloss') for k,v in rca_gaussian.items()]
    logistic_roc_list = [v.get('logistic_roc_auc') for k,v in rca_gaussian.items()]
    logistic_logloss_list = [v.get('logistic_logloss') for k,v in rca_gaussian.items()]

    ax[2].plot(n_components_list, logistic_roc_list, color='tab:blue', label='logistic regression')
    ax[2].plot(n_components_list, dt_roc_list, color='tab:orange', label='random forest')
    ax[2].set_title("ROC AUC on Validation Set (Gaussian)")
    ax[2].set_xlabel('n_components')
    ax[2].set_ylabel('auc')
    ax[2].legend()

    ax[3].plot(n_components_list, logistic_logloss_list, color='tab:blue', label='logistic regression')
    ax[3].plot(n_components_list, dt_logloss_list, color='tab:orange', label='random forest')
    ax[3].set_title("logloss on Validation Set (Gaussian)")
    ax[3].set_xlabel('n_components')
    ax[3].set_ylabel('logloss')
    ax[3].legend()
    fig.suptitle('Dataset: {dataset_name}'.format(dataset_name=dataset_name))
    plot_path = '{plot_dir}/{dataset_name}_{step}_{method}.png'
    plt.savefig(plot_path.format(plot_dir=PLOT_DIR, dataset_name=dataset_name, step='dim_reduction', method='rca'))
    plt.close()

    # dt
    fig, ax = plt.subplots(1, 3, figsize=(24, 6), dpi=80, facecolor='w', edgecolor='k')
    rfc = load_data(path=RESUTL_PATH, filename=filename.format(dataset_name=dataset_name, step='dim_reduction', method='rfc'))
    feature_importance_cutoff = [k for k, v in rfc.items() if k != 'feature_importance']
    dt_roc_list = [v.get('dt_roc_auc') for k, v in rfc.items() if k != 'feature_importance']
    dt_logloss_list = [v.get('dt_logloss') for k, v in rfc.items() if k != 'feature_importance']
    logistic_roc_list = [v.get('logistic_roc_auc') for k, v in rfc.items() if k != 'feature_importance']
    logistic_logloss_list = [v.get('logistic_logloss') for k, v in rfc.items() if k != 'feature_importance']
    feature_imp = rfc.get('feature_importance')
    ax[0].bar(feature_imp.index.astype(str), feature_imp.feature_importance, color='tab:blue')
    ax[0].set_title("Sorted Feature Importance")
    ax[0].set_xlabel('Feature Index')
    ax[0].set_ylabel('impurity explanation')
    ax[0].set_xticks([], [])

    ax[1].plot(feature_importance_cutoff, logistic_roc_list, color='tab:blue', label='logistic regression')
    ax[1].plot(feature_importance_cutoff, dt_roc_list, color='tab:orange', label='random forest')
    ax[1].set_title("ROC AUC on Validation Set")
    ax[1].set_xlabel('n_components')
    ax[1].set_ylabel('auc')
    ax[1].legend()

    ax[2].plot(feature_importance_cutoff, logistic_logloss_list, color='tab:blue', label='logistic regression')
    ax[2].plot(feature_importance_cutoff, dt_logloss_list, color='tab:orange', label='random forest')
    ax[2].set_title("logloss on Validation Set")
    ax[2].set_xlabel('n_components')
    ax[2].set_ylabel('logloss')
    ax[2].legend()

    fig.suptitle('Dataset: {dataset_name}'.format(dataset_name=dataset_name))
    plot_path = '{plot_dir}/{dataset_name}_{step}_{method}.png'
    plt.savefig(plot_path.format(plot_dir=PLOT_DIR, dataset_name=dataset_name, step='dim_reduction', method='rfc'))
    plt.close()
