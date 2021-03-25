from preprocessor import *

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


