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

SEED = 12345
DATA_FOLDER = 'data/'
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2


def run_pca(data, n_components):
    pca_result = {}
    pca = PCA(n_components=n_components).fit(data)
    pca_result.update({'explained_variance_ratio': pca.explained_variance_ratio_,
                       'explained_variance': pca.explained_variance_,
                       'singular_values': pca.singular_values_})

    return pca_result, pca


def run_ica(X_train, y_train, X_validation, y_validation, n_components, X_test, y_test):
    ica_result = {}
    ica = FastICA(n_components=n_components)
    X_train = ica.fit_transform(X_train)
    X_validation = ica.transform(X_validation)
    # decision tree
    dt = RandomForestClassifier(min_samples_split=8, min_impurity_decrease=0.0001)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict_proba(X_validation)[:, (dt.classes_ == 1)].reshape((-1))
    dt_roc_auc, dt_pr_auc, dt_logloss = summarize_offline_metrics(y_true=y_validation, y_score=y_pred_dt)
    # logistic regression
    logistic = LogisticRegression()
    logistic.fit(X_train, y_train)
    y_pred_logistic = logistic.predict_proba(X_validation)[:, (logistic.classes_ == 1)].reshape((-1))
    logistic_roc_auc, logistic_pr_auc, logistic_logloss = summarize_offline_metrics(y_true=y_validation,
                                                                                    y_score=y_pred_logistic)
    # remained to implement

    ica_result.update({'dt_roc_auc': dt_roc_auc,
                       'dt_pr_auc': dt_pr_auc,
                       'dt_logloss': dt_logloss,
                       'logistic_roc_auc': logistic_roc_auc,
                       'logistic_pr_auc': logistic_pr_auc,
                       'logistic_logloss': logistic_logloss})

    return ica_result, ica


def run_rca(X_train, y_train, X_validation, y_validation, n_components, random_projection_type, X_test, y_test):
    rca_result = {}
    rca = random_projection_type(n_components=n_components)
    X_train = rca.fit_transform(X_train)
    X_validation = rca.transform(X_validation)
    # decision tree
    dt = RandomForestClassifier(min_samples_split=8, min_impurity_decrease=0.0001)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict_proba(X_validation)[:, (dt.classes_ == 1)].reshape((-1))
    dt_roc_auc, dt_pr_auc, dt_logloss = summarize_offline_metrics(y_true=y_validation, y_score=y_pred_dt)
    # logistic regression
    logistic = LogisticRegression()
    logistic.fit(X_train, y_train)
    y_pred_logistic = logistic.predict_proba(X_validation)[:, (logistic.classes_ == 1)].reshape((-1))
    logistic_roc_auc, logistic_pr_auc, logistic_logloss = summarize_offline_metrics(y_true=y_validation,
                                                                                    y_score=y_pred_logistic)
    # remained to implement

    rca_result.update({'dt_roc_auc': dt_roc_auc,
                       'dt_pr_auc': dt_pr_auc,
                       'dt_logloss': dt_logloss,
                       'logistic_roc_auc': logistic_roc_auc,
                       'logistic_pr_auc': logistic_pr_auc,
                       'logistic_logloss': logistic_logloss})

    return rca_result, rca


def run_rf_feature_importance(X_train, y_train, **kwargs):
    dt = RandomForestClassifier(min_samples_split=8, min_impurity_decrease=0.0001)
    dt.fit(X_train, y_train)
    feature_importance = pd.DataFrame({"feature_importance": dt.feature_importances_}).reset_index().sort_values(
        by='feature_importance', ascending=False)
    feature_importance['cumsum_importance'] = feature_importance['feature_importance'].cumsum()
    return feature_importance


def run_dummy_clf_with_rf_feature_importance(feature_importance, cutoff, X_train, y_train, X_validation, y_validation,
                                             **kwargs):
    cutoff = max(cutoff, feature_importance['cumsum_importance'].min())
    ind = feature_importance.loc[feature_importance['cumsum_importance'] <= cutoff, 'index'].values
    mask = [True if i in ind else False for i in list(range(feature_importance.shape[0]))]
    X_train = X_train[:, mask]
    X_validation = X_validation[:, mask]
    dt_result = {}
    # decision tree
    dt = RandomForestClassifier(min_samples_split=8, min_impurity_decrease=0.0001)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict_proba(X_validation)[:, (dt.classes_ == 1)].reshape((-1))
    dt_roc_auc, dt_pr_auc, dt_logloss = summarize_offline_metrics(y_true=y_validation, y_score=y_pred_dt)
    # logistic regression
    logistic = LogisticRegression()
    logistic.fit(X_train, y_train)
    y_pred_logistic = logistic.predict_proba(X_validation)[:, (logistic.classes_ == 1)].reshape((-1))
    logistic_roc_auc, logistic_pr_auc, logistic_logloss = summarize_offline_metrics(y_true=y_validation,
                                                                                    y_score=y_pred_logistic)
    # remained to implement

    dt_result.update({'dt_roc_auc': dt_roc_auc,
                      'dt_pr_auc': dt_pr_auc,
                      'dt_logloss': dt_logloss,
                      'logistic_roc_auc': logistic_roc_auc,
                      'logistic_pr_auc': logistic_pr_auc,
                      'logistic_logloss': logistic_logloss})

    return dt_result, mask


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
    splitted_data = {
        'X_train': cluster_input,
        'y_train': y_train,
        'X_validation': preprocessor.transform(X_validation),
        'y_validation': y_validation,
        'X_test': preprocessor.transform(X_test),
        'y_test': y_test
    }
    # store transformed data for visualization purpose
    print("Start to store data:")
    filename = '{dataset_name}_{step}_{method}.pkl'
    save_data(splitted_data, path='results/',
              filename=filename.format(dataset_name=dataset_name, step='full_splitted_data', method='cluster_input'))

    # PCA:
    print("Start PCA")
    filename = '{dataset_name}_{step}_{method}.pkl'
    pca_result, _ = run_pca(cluster_input, n_components=None)
    save_data(pca_result, path='results/',
              filename=filename.format(dataset_name=dataset_name, step='dim_reduction', method='pca'))

    # ICA: found that n=5; n=12 render the best results
    print("Start ICA")
    n_iter = 20
    n_features = cluster_input.shape[1]
    ica_results = {}
    for n_components in np.linspace(2, n_features, n_iter).astype(int):
        print('n_components = ', n_components)
        ica_result, _ = run_ica(**splitted_data, n_components=n_components)
        ica_results.update({n_components: ica_result})

    filename = '{dataset_name}_{step}_{method}.pkl'
    save_data(ica_results, path='results/',
              filename=filename.format(dataset_name=dataset_name, step='dim_reduction', method='ica'))

    # RCA
    print("Start RCA")
    n_iter = 20
    # gaussian projection
    rca_results = {}
    for n_components in np.linspace(2, n_features, n_iter).astype(int):
        print('n_components = ', n_components)
        rca_result, _ = run_rca(**splitted_data, n_components=n_components,
                                random_projection_type=GaussianRandomProjection)
        rca_results.update({n_components: rca_result})

    filename = '{dataset_name}_{step}_{method}.pkl'
    save_data(rca_results, path='results/',
              filename=filename.format(dataset_name=dataset_name, step='dim_reduction', method='rca_gaussian'))
    # sparse projection
    rca_results = {}
    for n_components in np.linspace(2, n_features, n_iter).astype(int):
        print('n_components = ', n_components)
        rca_result, _ = run_rca(**splitted_data, n_components=n_components,
                                random_projection_type=SparseRandomProjection)
        rca_results.update({n_components: rca_result})

    filename = '{dataset_name}_{step}_{method}.pkl'
    save_data(rca_results, path='results/',
              filename=filename.format(dataset_name=dataset_name, step='dim_reduction', method='rca_sparse'))

    # Decision Tree (Random Forest feature selection) .8 as the cutoff
    n_iter = 20
    dt_results = {}
    feature_importance = run_rf_feature_importance(**splitted_data)
    dt_results.update({'feature_importance': feature_importance})
    for cutoff in np.linspace(0.1, 1, n_iter):
        print('cutoff = ', cutoff)
        dt_result, _ = run_dummy_clf_with_rf_feature_importance(feature_importance=feature_importance, cutoff=cutoff,
                                                                **splitted_data)
        dt_results.update({cutoff: dt_result})
    filename = '{dataset_name}_{step}_{method}.pkl'
    save_data(dt_results, path='results/',
              filename=filename.format(dataset_name=dataset_name, step='dim_reduction', method='rfc'))
