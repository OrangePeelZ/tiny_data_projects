import pandas as pd
import sys
import os
from operator import itemgetter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

from preprocessor import *
from utils import *
from dnn_estimator_builder import *
from meta_data import get_features_by_type, get_train_params, get_label

SEED = 12345
DATA_FOLDER = 'data/'
MODEL_TYPE = dict({'dt': DecisionTreeClassifier,
                   'gbt': GradientBoostingClassifier,
                   'svm': SVC,
                   'knn': KNeighborsClassifier})


def find_best_param(eval_list, model_type):
    if model_type == 'svm':
        l = [(k, v.get('accuracy')) for k, v in eval_list.items()]
        return max(l, key=itemgetter(1))[0]

    if model_type in ['dt', 'gbt', 'knn', 'svm', 'dnn']:
        l = [(k, v.get('logloss')) for k, v in eval_list.items()]
        return min(l, key=itemgetter(1))[0]

    else:
        raise ValueError('Invalid Model Type!!')


def train_and_evaluation_models(dataset_name,
                                model_type,
                                # dnn parameters
                                shuffle_buffer_size,
                                batch_size,
                                max_epochs,
                                base_lr,
                                per_epoch_decay,
                                patience,
                                params,
                                preprocessor=None,
                                seed=SEED):
    if model_type and model_type in ['dt', 'gbt', 'knn', 'svm']:
        name = dataset_name + '_' + model_type
        clf = MODEL_TYPE[model_type]
        params = params[model_type]
        estimators = construct_estimators(params, clf)
        train_estimators(X_train, y_train, estimators, preprocessor=preprocessor)
        validation_eval_list = evaluate_estimator_on_test(estimators,
                                                          X_validation,
                                                          y_validation,
                                                          preprocessor=preprocessor,
                                                          clf_type=model_type)
        save_metrics(name=name + '_validation',
                     d=validation_eval_list)
        plot_aucs(y_predicted_list=validation_eval_list,
                  y_test=y_validation,
                  name=name + '_validation')

        eval_list = evaluate_estimator_on_test(estimators,
                                               X_test,
                                               y_test,
                                               preprocessor=preprocessor,
                                               clf_type=model_type)
        save_metrics(name=name,
                     d=eval_list)
        plot_aucs(y_predicted_list=eval_list,
                  y_test=y_test,
                  name=name)



    elif model_type == 'dnn':
        name = dataset_name + '_' + model_type
        estimators, _ = train_dnn_estimators(X_train,
                                             y_train,
                                             dnn_params=params[model_type],
                                             shuffle_buffer_size=shuffle_buffer_size,
                                              batch_size=batch_size,
                                             max_epochs=max_epochs,
                                             base_lr=base_lr,
                                             per_epoch_decay=per_epoch_decay,
                                             patience=patience,
                                             preprocessor=preprocessor,
                                             seed=seed)
        eval_list = evaluate_dnn_estimator_on_test(estimators, X_test, y_test, preprocessor=preprocessor)
        save_metrics(name=name, d=eval_list)
        plot_aucs(y_predicted_list=eval_list, y_test=y_test, name=name)

    else:
        raise ValueError('Invalid Model Type')

    # return best parameters based on the evaluation
    return find_best_param(eval_list=eval_list, model_type=model_type)


def train_size_iter_analysis(X_train,
                             y_train,
                             X_test,
                             y_test,
                             best_param_key,
                             numeric_features,
                             categorical_features,
                             pre_encode_features,
                             shuffle_buffer_size,
                             base_lr,
                             batch_size,
                             per_epoch_decay,
                             random_state=SEED,
                             ):
    if model_type and model_type in ['dt', 'knn', 'svm']:
        best_params = params[model_type][best_param_key]
        print('best params: ', best_params)
        # dt/knn will be measured by logloss and svm will be measured by accuracy
        test_error = []
        train_error = []
        n_exp = 20
        for ratio in np.linspace(0.1, 1, n_exp):

            X_train_sub, y_train_sub = X_train.sample(frac=ratio, random_state=random_state), \
                                       y_train.sample(frac=ratio, random_state=random_state)
            print('train size: ', ratio, X_train_sub.shape)
            print('x test size: ', X_test.shape)
            preprocessor = basic_preprocessor(numeric_features=numeric_features,
                                              categorical_features=categorical_features,
                                              pre_encode_features=pre_encode_features)
            preprocessor.fit(X_train_sub)
            clf = MODEL_TYPE[model_type]
            estimator = clf(**best_params)
            estimator.fit(preprocessor.transform(X_train_sub), y_train_sub)

            # evaluate on the test and train dataset
            X_test_transform = preprocessor.transform(X_test)

            if model_type == 'svm':
                y_pred_test = estimator.predict(X_test_transform)
                y_pred_train = estimator.predict(preprocessor.transform(X_train_sub))
                test_error.append((1 - accuracy_score(y_true=y_test, y_pred=y_pred_test)))
                train_error.append((1 - accuracy_score(y_true=y_train_sub, y_pred=y_pred_train)))
            else:
                y_pred_test = estimator.predict_proba(X_test_transform)[:, (estimator.classes_ == 1)].reshape((-1))
                y_pred_train = estimator.predict_proba(preprocessor.transform(X_train_sub))[:,
                               (estimator.classes_ == 1)].reshape((-1))
                test_error.append(log_loss(y_true=y_test, y_pred=y_pred_test))
                train_error.append(log_loss(y_true=y_train_sub, y_pred=y_pred_train))
        return np.linspace(0.1, 1, n_exp), test_error, train_error

    elif model_type == 'gbt':
        best_params = params[model_type][best_param_key]
        print(best_params)
        test_error = []
        train_error = []
        base_n_estimators = best_params.get('n_estimators')
        preprocessor = basic_preprocessor(numeric_features=numeric_features,
                                          categorical_features=categorical_features,
                                          pre_encode_features=pre_encode_features)
        preprocessor.fit(X_train)
        X_test = preprocessor.transform(X_test)
        n_exp = 16
        for n_est in np.linspace(base_n_estimators / 8, base_n_estimators * 2, n_exp).astype('int'):
            print('n_iter = ', n_est)
            best_params.update({'n_estimators': n_est})
            clf = MODEL_TYPE[model_type]
            estimator = clf(**best_params)
            estimator.fit(preprocessor.transform(X_train), y_train)

            y_pred_test = estimator.predict_proba(X_test)[:, (estimator.classes_ == 1)].reshape((-1))
            y_pred_train = estimator.predict_proba(preprocessor.transform(X_train))[:,
                           (estimator.classes_ == 1)].reshape((-1))
            test_error.append(log_loss(y_true=y_test, y_pred=y_pred_test))
            train_error.append(log_loss(y_true=y_train, y_pred=y_pred_train))
        return np.linspace(base_n_estimators / 8, base_n_estimators * 4, n_exp), test_error, train_error

    elif model_type == 'dnn':
        best_params = params[model_type][best_param_key]
        print(best_param_key, best_params)
        preprocessor = basic_preprocessor(numeric_features=numeric_features,
                                          categorical_features=categorical_features,
                                          pre_encode_features=pre_encode_features)
        preprocessor.fit(X_train)
        X_test = preprocessor.transform(X_test)
        X_train = preprocessor.transform(X_train)
        max_patience = 10

        if scipy.sparse.issparse(X_train):
            X_train = convert_sparse_matrix_to_sparse_tensor(X_train)
            X_test = convert_sparse_matrix_to_sparse_tensor(X_test)

        n_train, n_features = X_train.shape
        steps_per_epoch = n_train / batch_size
        estimators = construct_dnn_estimators(params={'best_param': best_params},
                                              n_features=n_features)

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        train_dataset = train_dataset.shuffle(shuffle_buffer_size).repeat().batch(batch_size)
        test_dataset = test_dataset.batch(batch_size)
        histories = {}
        for k, estimator in estimators.items():
            histories[k] = compile_and_fit(model=estimator,
                                           name=k,
                                           train_dataset=train_dataset,
                                           validation_dataset=test_dataset,
                                           steps_per_epoch=steps_per_epoch,
                                           base_lr=base_lr,
                                           per_epoch_decay=per_epoch_decay,
                                           patience=max_patience)
        return histories


def clean_data_set(data, dataset_name):
    if dataset_name == 'campaign_marketing':
        print('I am already clean :)')
    elif dataset_name == 'university_recommendation':
        data['toeflEssay'] = data['toeflEssay'].apply(clean_str_in_numeric_feature)
        data['journalPubs'] = data['journalPubs'].apply(clean_str_in_numeric_feature)
        data['confPubs'] = data['confPubs'].apply(clean_str_in_numeric_feature)

        clean_rare_category_group(data=data, feature='major')
        clean_rare_category_group(data=data, feature='department')
        clean_rare_category_group(data=data, feature='ugCollege')
        precode_specialization = tokenize_features(feature='specialization', data=data, n=200)
        precode_term_year = generate_term_year(data=data, feature='termAndYear')
        return precode_specialization + precode_term_year


def plot_size_iter_analysis(history, model_type, dataset_name, path='plots/'):
    if isinstance(history, tuple):
        iter, test_error, train_error = history
    elif isinstance(history, dict):
        history = history['best_param'].history
        train_error = history['loss']
        test_error = history['val_loss']
        iter = list(range(len(test_error)))
    else:
        raise ValueError('Invalid history type!')

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    lw = 2
    # colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green']
    x_label = 'n_iter' if model_type in ('gbt', dnn) else 'train_size'
    y_label = 'logloss' if model_type in ('gbt', 'dnn', 'dt', 'knn') else 'accuracy'
    title = model_type + ': ' + x_label + ' vs ' + y_label
    # print(history)
    p1 = ax.plot(iter,
                 test_error,
                 lw=lw,
                 label='test error',
                 color='tab:blue')
    ax2 = ax.twinx()
    p2 = ax2.plot(iter,
                  train_error,
                  lw=lw,
                  label='train error',
                  color='tab:orange')
    p_all = p1 + p2
    labs = [l.get_label() for l in p_all]
    ax.legend(p_all, labs)

    ax.set_xlabel(x_label)
    ax.set_ylabel('test error/' + y_label)
    ax2.set_ylabel('train error/' + y_label)
    ax.set_title(title)

    name = dataset_name + '_further_analysis_' + model_type
    result_path = path + name + '.png'
    plt.savefig(result_path)
    plt.close()


if __name__ == "__main__":
    dataset_name = sys.argv[1]
    operation = sys.argv[2]
    try:
        model_type = sys.argv[3]

    except:
        model_type = None

    try:
        futher_analysis = sys.argv[4]
    except:
        futher_analysis = None

    data_path = os.path.join(DATA_FOLDER, dataset_name, 'training.csv')
    data = pd.read_csv(data_path)
    params = get_train_params(dataset_name)
    numeric_features, categorical_features = get_features_by_type(dataset_name)
    pre_encode_features = clean_data_set(data, dataset_name=dataset_name)
    data['label'] = get_label(name=dataset_name, data=data)

    if dataset_name == 'campaign_marketing':

        if operation == 'train_and_evaluation':
            X_train_validation, X_test, y_train_validation, y_test = train_test_split(
                data[(numeric_features + categorical_features)],
                data['label'],
                test_size=0.2,
                random_state=SEED)

            X_train, X_validation, y_train, y_validation = train_test_split(X_train_validation,
                                                                            y_train_validation,
                                                                            test_size=0.2,
                                                                            random_state=SEED)

            preprocessor = basic_preprocessor(numeric_features=numeric_features,
                                              categorical_features=categorical_features)
            preprocessor.fit(X_train)

            if model_type:
                if model_type in ['dt', 'gbt', 'knn', 'svm', 'dnn']:
                    best_param = train_and_evaluation_models(dataset_name=dataset_name,
                                                             model_type=model_type,
                                                             params=params,
                                                             preprocessor=preprocessor,
                                                             shuffle_buffer_size=100,
                                                             batch_size=64,
                                                             max_epochs=100,
                                                             base_lr=0.005,
                                                             per_epoch_decay=10,
                                                             patience=5)
                    if futher_analysis == 'True':
                        history = train_size_iter_analysis(X_train=X_train,
                                                           y_train=y_train,
                                                           X_test=X_test,
                                                           y_test=y_test,
                                                           best_param_key=best_param,
                                                           numeric_features=numeric_features,
                                                           categorical_features=categorical_features,
                                                           pre_encode_features=pre_encode_features,
                                                           shuffle_buffer_size=100,
                                                           base_lr=0.005,
                                                           batch_size=64,
                                                           per_epoch_decay=10,
                                                           random_state=SEED,
                                                           )

                        plot_size_iter_analysis(history, model_type=model_type, dataset_name=dataset_name)

                else:
                    raise ValueError('Unknown Model Type!')

            else:
                for i in ['dt', 'gbt', 'knn', 'svm', 'dnn']:
                    train_and_evaluation_models(dataset_name=dataset_name,
                                                model_type=i,
                                                params=params,
                                                preprocessor=preprocessor,
                                                shuffle_buffer_size=100,
                                                batch_size=64,
                                                max_epochs=100,
                                                base_lr=0.005,
                                                per_epoch_decay=10,
                                                patience=5)

        elif operation == 'EDA':
            eda_path = 'EDA/'

            title = dataset_name + ": label distribution"
            data.label.hist()
            plt.title(title)
            plt.savefig(eda_path + dataset_name + '_label.png')
            plt.close()

            n_row = 3
            fig, ax = plt.subplots(7, n_row, figsize=(24, 48), dpi=80, facecolor='w', edgecolor='k')
            fig.suptitle('Feature Distributions')
            for i, ft in enumerate(numeric_features + categorical_features):
                data[ft].hist(bins=20, ax=ax[int(i / n_row), i % n_row])
                ax[int(i / n_row), i % n_row].set_title("{ft} distribution".format(ft=ft))
            plt.savefig(eda_path + dataset_name + '_features.png')
            plt.close()

        else:
            raise ValueError('Illegal Operation')


    elif dataset_name == 'university_recommendation':
        if operation == 'train_and_evaluation':

            X_train_validation, X_test, y_train_validation, y_test = train_test_split(
                data[(numeric_features + categorical_features + pre_encode_features)],
                data['label'],
                test_size=0.2,
                random_state=SEED)

            X_train, X_validation, y_train, y_validation = train_test_split(X_train_validation,
                                                                            y_train_validation,
                                                                            test_size=0.2,
                                                                            random_state=SEED)

            preprocessor = basic_preprocessor(numeric_features=numeric_features,
                                              categorical_features=categorical_features,
                                              pre_encode_features=pre_encode_features)
            preprocessor.fit(X_train)

            if model_type:
                if model_type in ['dt', 'gbt', 'knn', 'svm', 'dnn']:
                    best_param = train_and_evaluation_models(dataset_name=dataset_name,
                                                             model_type=model_type,
                                                             params=params,
                                                             preprocessor=preprocessor,
                                                             shuffle_buffer_size=1024 * 8,
                                                             batch_size=1024,
                                                             max_epochs=500,
                                                             base_lr=0.0001,
                                                             per_epoch_decay=30,
                                                             patience=3)
                else:
                    raise ValueError('Unknown Model Type!')

                if futher_analysis == 'True':
                    history = train_size_iter_analysis(X_train=X_train,
                                                       y_train=y_train,
                                                       X_test=X_test,
                                                       y_test=y_test,
                                                       best_param_key=best_param,
                                                       numeric_features=numeric_features,
                                                       categorical_features=categorical_features,
                                                       pre_encode_features=pre_encode_features,
                                                       shuffle_buffer_size=1024 * 8,
                                                       base_lr=0.0001,
                                                       batch_size=1024,
                                                       per_epoch_decay=30,
                                                       random_state=SEED,
                                                       )

                    plot_size_iter_analysis(history, model_type=model_type, dataset_name=dataset_name)

            else:
                for i in ['dt', 'gbt', 'knn', 'svm', 'dnn']:
                    train_and_evaluation_models(dataset_name=dataset_name,
                                                model_type=i,
                                                params=params,
                                                preprocessor=preprocessor,
                                                shuffle_buffer_size=1024 * 8,
                                                batch_size=1024,
                                                max_epochs=100,
                                                base_lr=0.0001,
                                                per_epoch_decay=30,
                                                patience=3)

        elif operation == 'EDA':
            eda_path = 'EDA/'

            title = dataset_name + ": label distribution"
            data.label.hist()
            plt.title(title)
            plt.savefig(eda_path + dataset_name + '_label.png')
            plt.close()

            n_row = 4
            fig, ax = plt.subplots(6, n_row, figsize=(24, 36), dpi=80, facecolor='w', edgecolor='k')
            fig.suptitle('Feature Distributions')
            for i, ft in enumerate(numeric_features + categorical_features):
                data[ft].hist(bins=20, ax=ax[int(i / n_row), i % n_row])
                ax[int(i / n_row), i % n_row].set_title("{ft} distribution".format(ft=ft))
            plt.savefig(eda_path + dataset_name + '_features.png')
            plt.close()

        else:
            raise ValueError('Illegal Operation')

    else:
        raise ValueError('Illegal Dataset Name!')
