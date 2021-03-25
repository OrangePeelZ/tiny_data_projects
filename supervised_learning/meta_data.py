def get_train_params(name):
    if name == 'campaign_marketing':
        params = dict({'dt': dict({'case1': {'min_samples_split': 2, 'max_features': 1},
                                   'case2': {'min_samples_split': 64, 'max_features': 1},
                                   'case3': {'min_samples_split': 16, 'max_features': 0.5},
                                   'case4': {'min_samples_split': 64, 'max_features': 0.5}}),
                       "gbt": dict({
                           'case1': {'min_samples_split': 2, 'n_estimators': 128, 'learning_rate': 0.1},
                           'case2': {'min_samples_split': 64, 'n_estimators': 128, 'learning_rate': 0.1},
                           'case3': {'min_samples_split': 64, 'n_estimators': 512, 'learning_rate': 0.1},
                           'case4': {'min_samples_split': 64, 'n_estimators': 1024, 'learning_rate': 0.01}
                       }),
                       "svm": dict({
                           'case1': {'kernel': 'rbf', 'C': 1},
                           'case2': {'kernel': 'rbf', 'C': 2},
                           'case3': {'kernel': 'poly', 'C': 0.5, 'degree': 3},
                           'case4': {'kernel': 'poly', 'C': 0.5, 'degree': 5}
                       }),

                       "knn": dict({
                           'case1': {'n_neighbors': 5, 'p': 1},
                           'case2': {'n_neighbors': 32, 'p': 1},
                           'case3': {'n_neighbors': 5, 'p': 2},
                           'case4': {'n_neighbors': 32, 'p': 2}
                       }),

                       "dnn":
                           dict({
                               'tiny': {'hidden_units': [8]},
                               'small': {'hidden_units': [16, 8]},
                               'medium': {'hidden_units': [32, 16, 8]},
                               'larger': {'hidden_units': [64, 64, 32]}
                           })})
    if name == 'university_recommendation':
        params = dict({'dt': dict({'case1': {'min_samples_split': 2, 'max_features': 1},
                                   'case2': {'min_samples_split': 64, 'max_features': 1},
                                   'case3': {'min_samples_split': 16, 'max_features': 0.5},
                                   'case4': {'min_samples_split': 64, 'max_features': 0.5}}),
                       "gbt": dict({
                           'case1': {'min_samples_split': 2, 'n_estimators': 100, 'learning_rate': 0.1},
                           'case2': {'min_samples_split': 16, 'n_estimators': 100, 'learning_rate': 0.1},
                           'case3': {'min_samples_split': 64, 'n_estimators': 500, 'learning_rate': 0.1},
                           'case4': {'min_samples_split': 64, 'n_estimators': 10000, 'learning_rate': 0.05}
                       }),
                       "svm": dict({
                           'case1': {'kernel': 'rbf', 'C': 1},
                           'case2': {'kernel': 'rbf', 'C': 2},
                           'case3': {'kernel': 'poly', 'C': 0.5, 'degree': 3},
                           'case4': {'kernel': 'poly', 'C': 0.5, 'degree': 5}
                       }),

                       "knn": dict({
                           'case1': {'n_neighbors': 5, 'p': 1},
                           'case2': {'n_neighbors': 32, 'p': 1},
                           'case3': {'n_neighbors': 5, 'p': 2},
                           'case4': {'n_neighbors': 32, 'p': 2}
                       }),

                       "dnn":
                           dict({
                               'tiny': {'hidden_units': [8]},
                               'small': {'hidden_units': [16, 8]},
                               'medium': {'hidden_units': [32, 16, 8]},
                               'larger': {'hidden_units': [64, 64, 32]}
                           })})

    return params


def get_features_by_type(name):
    if name == 'campaign_marketing':
        numeric_features = ['custAge', 'campaign', 'pdays', 'previous', 'emp.var.rate',
                            'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed',
                            'pmonths', 'pastEmail']
        categorical_features = ['profession', 'marital', 'schooling',
                                'default', 'housing', 'loan', 'contact',
                                'month', 'day_of_week', 'poutcome']
    if name == 'university_recommendation':
        numeric_features = ['researchExp', 'industryExp','toeflScore','internExp',
                            'greV', 'greQ','greA', 'topperCgpa', 'gmatA', 'cgpa', 'gmatQ',
                            'cgpaScale', 'gmatV','toeflEssay','journalPubs','confPubs']
        categorical_features = [ 'major', 'program', 'department', 'ugCollege','univName']

    return numeric_features, categorical_features


def get_label(name, data):
    if name == 'campaign_marketing':
        label_dict = dict({'no':0, 'yes':1})
        return data.responded.map(label_dict)
    if name == 'university_recommendation':
        return data['admit']


