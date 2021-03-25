import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, average_precision_score, log_loss, auc, \
    precision_recall_curve, PrecisionRecallDisplay, confusion_matrix
import json


def plot_auc(y_true, y_score, ax, color, label=''):
    lw = 2
    roc_auc = roc_auc_score(y_true=y_true,
                            y_score=y_score)

    pr_auc = average_precision_score(y_true=y_true,
                                     y_score=y_score)

    fpr, tpr, thresholds = roc_curve(y_true=y_true,
                                     y_score=y_score, pos_label=1)

    prec, recall, _ = precision_recall_curve(y_true=y_true,
                                             probas_pred=y_score, pos_label=1)

    ax[0].plot(fpr,
               tpr,
               lw=lw,
               label='{label} ROC curve (area = {roc_auc:0.4f})'.format(label=label, roc_auc=roc_auc),
               color=color)
    ax[0].legend()
    pr_label = '{label} PR AUC curve (area = {pr_auc:0.4f})'
    PrecisionRecallDisplay(precision=prec, recall=recall).plot(ax=ax[1],
                                                               color=color,
                                                               label=pr_label.format(
                                                                   label=label,
                                                                   pr_auc=pr_auc))


def summarize_offline_metrics(y_true, y_score):
    roc_auc = roc_auc_score(y_true=y_true,
                            y_score=y_score)

    print('roc_auc: ', roc_auc)

    pr_auc = average_precision_score(y_true=y_true,
                                     y_score=y_score)

    print('pr_auc: ', pr_auc)

    logloss = log_loss(y_true=y_true,
                       y_pred=y_score)

    print('logloss: ', logloss)

    return roc_auc, pr_auc, logloss


def construct_estimators(params, estimator):
    estimators = dict()
    for k, param in params.items():
        estimators.update({k: estimator(**param)})
    return estimators


def train_estimators(X_train, y_train, estimators, preprocessor=None):
    if preprocessor:
        for k, estimator in estimators.items():
            estimator.fit(preprocessor.transform(X_train), y_train)
    else:
        for k, estimator in estimators.items():
            estimator.fit(X_train, y_train)


def evaluate_estimator_on_test(estimators, X_test, y_test, clf_type, preprocessor=None):
    y_predicted_list = dict()
    if clf_type in ['dt', 'gbt', 'knn']:
        if preprocessor:
            X_test = preprocessor.transform(X_test)

        for k, estimator in estimators.items():
            y_predicted = estimator.predict_proba(X_test)[:, (estimator.classes_ == 1)].reshape((-1))
            roc_auc, pr_auc, logloss = summarize_offline_metrics(y_true=y_test, y_score=y_predicted)
            accuracy = accuracy_score(y_true=y_test, y_pred=estimator.predict(X_test))
            y_predicted_list.update({k: {'y_predicted': y_predicted,
                                         'roc_auc': roc_auc,
                                         'pr_auc': pr_auc,
                                         'logloss': logloss,
                                         'accuracy': accuracy}})

    elif clf_type == 'svm':
        if preprocessor:
            X_test = preprocessor.transform(X_test)

        for k, estimator in estimators.items():
            y_predicted = estimator.predict(X_test)
            confusion = confusion_matrix(y_true=y_test, y_pred=y_predicted)
            accuracy = accuracy_score(y_true=y_test, y_pred=y_predicted)
            print ('confusion matrix: ', confusion)
            print ('Accuracy: ', accuracy)
            y_predicted_list.update({k: {'y_predicted': y_predicted,
                                         'confusion': confusion.tolist(),
                                         'accuracy': accuracy}})
    else:
        raise ValueError('Unsupported Model Type!')

    return y_predicted_list


def plot_aucs(y_predicted_list, y_test, name, path='plots/'):
    fig, ax = plt.subplots(1, 2, figsize=(16, 6), dpi=80, facecolor='w', edgecolor='k')
    lw = 2
    colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green']
    for i, (k, metric) in enumerate(y_predicted_list.items()):
        plot_auc(y_true=y_test, y_score=metric.get('y_predicted'), ax=ax, color=colors[i], label=k)

    ax[0].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax[0].set_xlim([0.0, 1.0])
    ax[0].set_ylim([0.0, 1.05])
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate')
    ax[0].set_title('ROC AUC')
    ax[0].legend()

    ax[1].set_xlabel('precision')
    ax[1].set_ylabel('recall')
    ax[1].set_title('PR AUC')
    ax[1].legend()

    result_path = path + name + '.png'
    plt.savefig(result_path)
    plt.close()

def save_results(d, name, path='metrics/'):
    file_path = path + name + '.json'
    with open(file_path, 'w') as fp:
        json.dump(d, fp)
    fp.close()

def save_metrics(d, name, path='metrics/'):
    metrics = dict({})
    for k, ret in d.items():
        metric = dict({})
        for a,b in ret.items():
            if a != 'y_predicted':
                metric.update({a: b})
        metrics.update({k: metric})

    file_path = path + name + '.json'
    with open(file_path, 'w') as fp:
        json.dump(metrics, fp)
    fp.close()
