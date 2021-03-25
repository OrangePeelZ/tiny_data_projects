from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from meta_data import get_features_by_type, get_train_params, get_label, clean_data_set
import os
import pandas as pd
import sys
from utils import *
from dnn_estimator_builder import *
from preprocessor import *
import pickle


SEED = 12345
DATA_FOLDER = 'data/'
TEST_SIZE=0.2
VALIDATION_SIZE=0.2


def save_data(data, path, filename):
    with open(os.path.join(path, filename), 'wb') as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    dataset_name = sys.argv[1]
    data_path = os.path.join(DATA_FOLDER, dataset_name, 'training.csv')
    data = pd.read_csv(data_path)
    numeric_features, categorical_features = get_features_by_type(dataset_name)
    pre_encode_features = clean_data_set(data, dataset_name=dataset_name)
    data['label'] = get_label(name=dataset_name, data=data)

    X_train_validation, X_test, y_train_validation, y_test = train_test_split(
        data[(numeric_features + categorical_features)],
        data['label'],
        test_size=TEST_SIZE,
        random_state=SEED)

    X_train, X_validation, y_train, y_validation = train_test_split(X_train_validation,
                                                                    y_train_validation,
                                                                    test_size=VALIDATION_SIZE,
                                                                    random_state=SEED)

    preprocessor = basic_preprocessor(numeric_features=numeric_features,
                                      categorical_features=categorical_features)
    preprocessor.fit(X_train)
    cluster_input = preprocessor.transform(X_train)

    # store transformed data for visualization purpose
    filename = '{dataset_name}_{step}_{method}.pkl'
    save_data(cluster_input, path='results/',
              filename=filename.format(dataset_name=dataset_name, step='clustering', method='cluster_input'))

    # Kmeans
    k_means_results = {}
    n_init = 10
    n_clusters = range(2,20,2)
    for n_cluster in n_clusters:
        cluster_metrics = {}
        kmeans = KMeans(n_clusters=n_cluster, random_state=SEED, n_init=n_init).fit(cluster_input)
        cluster_metrics.update({'label': kmeans.labels_,
                                'cluster_centers': kmeans.cluster_centers_,
                                'distance': kmeans.transform(cluster_input)})

        k_means_results.update({n_cluster: cluster_metrics})
    filename = '{dataset_name}_{step}_{method}.pkl'
    save_data(k_means_results, path='results/',
              filename=filename.format(dataset_name=dataset_name, step='clustering', method='kmeans'))

    # EM
    n_components = range(2,20,2)
    covariance_types = ['full', 'spherical']
    EM_results = {}
    for covariance_type in covariance_types:
        for n_component in n_components:
            cluster_metrics = {}
            em = GaussianMixture(n_components=n_component, covariance_type=covariance_type).fit(cluster_input)
            cluster_metrics.update({'means': em.means_,
                                    'covariances': em.covariances_,
                                    'bic': em.bic(cluster_input),
                                    'aic': em.aic(cluster_input),
                                    'loglikelihood': em.score(cluster_input),
                                    'label':em.predict(cluster_input)})
            params_em = '{covariance_type}_{n_component}'.format(covariance_type=covariance_type,
                                                                 n_component=str(n_component))
            EM_results.update({params_em: cluster_metrics})

    filename = '{dataset_name}_{step}_{method}.pkl'
    save_data(EM_results, path='results/',
              filename=filename.format(dataset_name=dataset_name, step='clustering', method='EM'))
