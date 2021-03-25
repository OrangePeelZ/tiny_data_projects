from sklearn.decomposition import PCA, FastICA
from sklearn.linear_model import LogisticRegression
from meta_data import get_features_by_type, get_label, clean_data_set
import os
import pandas as pd
import sys
from dnn_estimator_builder import *
from preprocessor import *
from clustering import save_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from visualize_clusters import load_data
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from meta_data import get_features_by_type, get_train_params, get_label, clean_data_set
import os
import pandas as pd
import sys
from utils import *
from dnn_estimator_builder import *
from preprocessor import *
from dimensionality_reduction import run_dummy_clf_with_rf_feature_importance
import pickle

SEED = 12345
DATA_FOLDER = 'data/'
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2


def define_params(dataset_name):
    if dataset_name == 'campaign_marketing':
        return {
            'pca': {'n_components': 20},
            'ica': {'n_components': 8},
            'sparse_rca': {'n_components': 45},
            'dt': {'cutoff': 0.8}
        }
    elif dataset_name == 'university_recommendation':
        return {
            'pca': {'n_components': 200},
            'ica': {'n_components': 90},
            'sparse_rca': {'n_components': 320},
            'dt': {'cutoff': 0.95}
        }
    else:
        raise ValueError('Invalid data set name!')


def construct_pipline(dim_reduce, clustering):
    return Pipeline([('dim_reduce', dim_reduce),
                     ('clustering', clustering)])


def construct_iterative_run(clustering_method, dim_reduce, data):
    if clustering_method == 'kmeans':
        clustering_algo = KMeans
        n_init = 10
        clustering_results = {}
        n_clusters = range(2, 20, 2)
        for n_cluster in n_clusters:
            print('n_cluster = ', n_cluster)
            cluster_metrics = {}
            pp = construct_pipline(dim_reduce,
                                   clustering_algo(n_clusters=n_cluster, random_state=SEED, n_init=n_init))

            distance = pp.fit_transform(data.get('X_train'))
            cluster_metrics.update({'label': pp[1].labels_,
                                    'distance': distance,
                                    'data':pp[0].transform(data.get('X_train'))})
            clustering_results.update({n_cluster: cluster_metrics})

    elif clustering_method == 'em':
        clustering_algo = GaussianMixture
        covariance_type = 'full'  # use "full convariance shape to the experiment"
        clustering_results = {}
        n_components = range(2, 20, 2)
        for n_component in n_components:
            print('n_component = ', n_component)
            cluster_metrics = {}
            pp = construct_pipline(dim_reduce,
                                   clustering_algo(n_components=n_component, covariance_type=covariance_type))

            labels = pp.fit_predict(data.get('X_train'))
            transformed_data = pp[0].transform(data.get('X_train'))
            cluster_metrics.update({'bic': pp[1].bic(transformed_data),
                                    'loglikelihood': pp.score(data.get('X_train')),
                                    'label': labels,
                                    'data':transformed_data})
            clustering_results.update({n_component: cluster_metrics})
    else:
        raise ValueError('Invalid Clustering Method!')
    return clustering_results


if __name__ == "__main__":
    dataset_name = sys.argv[1]
    clustering_method = sys.argv[2]
    filename = '{dataset_name}_{step}_{method}.pkl'
    data = load_data(path='results/', filename=filename.format(dataset_name=dataset_name,
                                                               step='full_splitted_data',
                                                               method='cluster_input'))
    params = define_params(dataset_name)

    # PCA
    print ('start PCA:')
    clustering_results = construct_iterative_run(clustering_method=clustering_method,
                                                 dim_reduce=PCA(**params.get('pca')),
                                                 data=data)
    filename = '{dataset_name}_{step}_{clustering_method}_{reduce_method}.pkl'
    save_data(clustering_results,
              path='results/',
              filename=filename.format(dataset_name=dataset_name,
                                       step='clustering_on_reduced',
                                       clustering_method=clustering_method,
                                       reduce_method='pca'))

    # ICA
    print ('start ICA:')
    clustering_results = construct_iterative_run(clustering_method=clustering_method,
                                                 dim_reduce=FastICA(**params.get('ica')),
                                                 data=data)
    filename = '{dataset_name}_{step}_{clustering_method}_{reduce_method}.pkl'
    save_data(clustering_results,
              path='results/',
              filename=filename.format(dataset_name=dataset_name,
                                       step='clustering_on_reduced',
                                       clustering_method=clustering_method,
                                       reduce_method='ica'))
    # RCA
    print ('start RCA:')
    clustering_results = construct_iterative_run(clustering_method=clustering_method,
                                                 dim_reduce=SparseRandomProjection(**params.get('sparse_rca')),
                                                 data=data)
    filename = '{dataset_name}_{step}_{clustering_method}_{reduce_method}.pkl'
    save_data(clustering_results,
              path='results/',
              filename=filename.format(dataset_name=dataset_name,
                                       step='clustering_on_reduced',
                                       clustering_method=clustering_method,
                                       reduce_method='rca'))
    # DT ft importance:
    print ('start DT:')
    path = 'results/'
    filename = '{dataset_name}_{step}_{method}.pkl'
    rfc = load_data(path=path, filename=filename.format(dataset_name=dataset_name, step='dim_reduction', method='rfc'))
    _, mask = run_dummy_clf_with_rf_feature_importance(feature_importance=rfc.get('feature_importance'),
                                                       cutoff=params.get('dt').get('cutoff'),
                                                       **data)
    X_train = data.get('X_train')[:, mask]
    if clustering_method == 'kmeans':
        clustering_algo = KMeans
        n_init = 10
        n_clusters = range(2, 20, 2)

        clustering_results = {}
        for n_cluster in n_clusters:
            print('n_cluster = ', n_cluster)
            cluster_metrics = {}
            clst = clustering_algo(n_clusters=n_cluster, random_state=SEED, n_init=n_init)
            distance = clst.fit_transform(X_train)
            cluster_metrics.update({'label': clst.labels_,
                                    'distance': distance,
                                    'data':X_train})
            clustering_results.update({n_cluster: cluster_metrics})
    elif clustering_method == 'em':
        clustering_algo = GaussianMixture
        covariance_type = 'full'  # use "full convariance shape to the experiment"
        clustering_results = {}
        n_components = range(2, 20, 2)

        clustering_results = {}
        for n_component in n_components:
            print('n_component = ', n_component)
            cluster_metrics = {}
            clst = clustering_algo(n_components=n_component, covariance_type=covariance_type)
            clst.fit(X_train)
            cluster_metrics.update({'bic': clst.bic(X_train),
                                    'loglikelihood': clst.score(X_train),
                                    'label': clst.predict(X_train),
                                    'data':X_train})
            clustering_results.update({n_component: cluster_metrics})
    else:
        raise ValueError('Invalid clustering method!')

    filename = '{dataset_name}_{step}_{clustering_method}_{reduce_method}.pkl'
    save_data(clustering_results,
              path='results/',
              filename=filename.format(dataset_name=dataset_name,
                                       step='clustering_on_reduced',
                                       clustering_method=clustering_method,
                                       reduce_method='dt'))

