from sklearn.decomposition import PCA, FastICA
import sys
from dnn_estimator_builder import *
from clustering import save_data
from sklearn.random_projection import SparseRandomProjection
from visualize_clusters import load_data
from dnn_estimator_builder import train_dnn_estimators
from dimensionality_reduction import run_dummy_clf_with_rf_feature_importance
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

SEED = 12345
DATA_FOLDER = 'data/'
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

PARAMS = {'tiny': {'hidden_units': [8]},
          'medium': {'hidden_units': [16, 8, 4]}}

SELECT_PARAMS = {'pca': PCA(n_components=20),
                 'lca': FastICA(n_components=8),
                 'rca': SparseRandomProjection(n_components=45),
                 'dt': 0.8,
                 'kmeans': KMeans(n_clusters=4),
                 'em': GaussianMixture(n_components=14, covariance_type='full')}


def apply_transformer(data, transformer):
    if transformer in ['pca', 'lca', 'rca', 'kmeans']:
        pp = SELECT_PARAMS[transformer]
        X_train = pp.fit_transform(data.get('X_train'))
        return {'X_train': X_train,
                'X_validation': pp.transform(data.get('X_validation')),
                'X_test': pp.transform(data.get('X_test')),
                'y_train': data.get('y_train'),
                'y_validation': data.get('y_validation'),
                'y_test': data.get('y_test')}
    elif transformer == 'dt':
        cutoff = SELECT_PARAMS[transformer]
        path = 'results/'
        filename = '{dataset_name}_{step}_{method}.pkl'
        rfc = load_data(path=path,
                        filename=filename.format(dataset_name=dataset_name, step='dim_reduction', method='rfc'))
        _, mask = run_dummy_clf_with_rf_feature_importance(feature_importance=rfc.get('feature_importance'),
                                                           cutoff=cutoff,
                                                           **data)

        return {'X_train': data.get('X_train')[:, mask],
                'X_validation': data.get('X_validation')[:, mask],
                'X_test': data.get('X_test')[:, mask],
                'y_train': data.get('y_train'),
                'y_validation': data.get('y_validation'),
                'y_test': data.get('y_test')}
    elif transformer == 'em':
        pp = SELECT_PARAMS[transformer]
        X_train = pp.fit(data.get('X_train'))
        return {'X_train': pp.predict_proba(data.get('X_train')),
                'X_validation': pp.predict_proba(data.get('X_validation')),
                'X_test': pp.predict_proba(data.get('X_test')),
                'y_train': data.get('y_train'),
                'y_validation': data.get('y_validation'),
                'y_test': data.get('y_test')}


if __name__ == "__main__":
    dataset_name = sys.argv[1]
    metrics = {}
    filename = '{dataset_name}_{step}_{method}.pkl'
    data = load_data(path='results/', filename=filename.format(dataset_name=dataset_name,
                                                               step='full_splitted_data',
                                                               method='cluster_input'))
    # train a baseline
    # From supervised learning experiment, the small network size fit the data the best
    estimators, histories = train_dnn_estimators(**data,
                                                 dnn_params=PARAMS,
                                                 shuffle_buffer_size=100,
                                                 batch_size=64,
                                                 max_epochs=100,
                                                 base_lr=0.005,
                                                 per_epoch_decay=10,
                                                 patience=5,
                                                 preprocessor=None,
                                                 seed=1234567)

    metrics.update({'baseline': evaluate_dnn_estimator_on_test(estimators=estimators, **data)})

    # apply the dimension reduction/clustering algorithm
    transformers = ['pca', 'lca', 'rca', 'dt', 'kmeans', 'em']
    for transformer in transformers:
        data_transformed = apply_transformer(data=data, transformer=transformer)
        estimators, histories = train_dnn_estimators(**data_transformed,
                                                     dnn_params=PARAMS,
                                                     shuffle_buffer_size=100,
                                                     batch_size=64,
                                                     max_epochs=100,
                                                     base_lr=0.005,
                                                     per_epoch_decay=10,
                                                     patience=5,
                                                     preprocessor=None,
                                                     seed=1234567)
        metrics.update({transformer: evaluate_dnn_estimator_on_test(estimators=estimators, **data_transformed)})
    filename = '{dataset_name}_{step}_{clustering_method}_{reduce_method}.pkl'
    save_data(metrics,
              path='results/',
              filename=filename.format(dataset_name=dataset_name,
                                       step='dnn_comparison',
                                       clustering_method='all',
                                       reduce_method='all'))
