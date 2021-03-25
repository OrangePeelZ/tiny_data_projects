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
import pandas as pd

PLOT_DIR = 'plots'
RESUTL_PATH = 'results/'

if __name__ == "__main__":
    dataset_name = sys.argv[1]

    filename = '{dataset_name}_{step}_{clustering_method}_{reduce_method}.pkl'
    metrics = load_data(path='results/', filename=filename.format(dataset_name=dataset_name,
                                                                  step='dnn_comparison',
                                                                  clustering_method='all',
                                                                  reduce_method='all'))
    methods = []
    models = []
    y_predicted = []
    roc_auc = []
    pr_auc = []
    logloss = []

    for method, model in metrics.items():
        for model_size, v in model.items():
            methods.append(method)
            models.append(model_size)
            y_predicted.append(v.get('y_predicted'))
            roc_auc.append(v.get('roc_auc'))
            pr_auc.append(v.get('pr_auc'))
            logloss.append(v.get('logloss'))

    metrics_df = pd.DataFrame(
        {'method': methods, 'models': models, 'roc_auc': roc_auc, 'pr_auc': pr_auc, 'logloss': logloss})
    fig, ax = plt.subplots(1, 3, figsize=(24, 6), dpi=80, facecolor='w', edgecolor='k')
    sns.barplot(x="method", y="roc_auc", hue="models", data=metrics_df, ax=ax[0])
    sns.barplot(x="method", y="pr_auc", hue="models", data=metrics_df, ax=ax[1])
    sns.barplot(x="method", y="logloss", hue="models", data=metrics_df, ax=ax[2])
    if dataset_name == 'campaign_marketing':
        ax[1].axhline(y=0.424691, color="grey", linestyle="--")
        ax[0].axhline(y=0.774000, color="grey", linestyle="--")
        ax[2].axhline(y=0.287591, color="grey", linestyle="--")
        ax[0].set_ylim(0.4, 1)
        ax[1].set_ylim(0.2, 0.6)
        ax[2].set_ylim(0.2, 0.4)
    elif dataset_name == 'university_recommendation':
        ax[0].axhline(y=0.757911, color="grey", linestyle="--")
        ax[1].axhline(y=0.762253, color="grey", linestyle="--")
        ax[2].axhline(y=0.605752, color="grey", linestyle="--")
        ax[0].set_ylim(0.4, 0.8)
        ax[1].set_ylim(0.3, 0.9)
        ax[2].set_ylim(0.4, 0.8)
    else:
        raise ValueError('Invalid Dataset Name!')

    fig.suptitle('{dataset_name}: Comparison of Different Feature Transformation Strategy'.format(
        dataset_name=dataset_name))
    plot_path = '{plot_dir}/{dataset_name}_{step}_{clustering_method}_{reduce_method}.png'
    plt.savefig(
        plot_path.format(plot_dir=PLOT_DIR, dataset_name=dataset_name, step='dnn_comparison', clustering_method='all',
                         reduce_method='all'))
    plt.close()
