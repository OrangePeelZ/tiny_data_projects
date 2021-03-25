from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from collections import Counter


def basic_preprocessor(numeric_features, categorical_features, pre_encode_features=None):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    transformers = [
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)]
    if pre_encode_features:
        transformers.append(('pre_enc', 'passthrough', pre_encode_features))

    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor


# The `university_recommendation` dataset needs data cleaning

def clean_str_in_numeric_feature(x):
    """ If the value is a string, then remove currency symbol and delimiters
    otherwise, the value is numeric and can be converted
    """
    try:
        x = int(x)
    except:
        x = None
    return (x)


def clean_rare_category_group(data, feature, key='userName', other_group_name='other', cutoff=5):
    data[feature] = data[feature].str.lower().replace("-!@#$%^&*\(\)\[\]\{\};:,./<>?\|`~=_+", "").replace(' ', '')
    per_category = data[[key, feature]].groupby([feature]).count().sort_values(by=key)
    data.loc[data[feature].isin(per_category[per_category[key] < cutoff].index.tolist()), feature] = other_group_name


def word_count(l, n=5):
    words = [i for i in l.split(' ') if i not in ['', ' ']]
    counter_obj = Counter(words)
    return counter_obj.most_common(n=n)


# feature = 'specialization'

def tokenize_features(feature, data, n=5):
    l = ' '.join([str(i) for i in data[feature].values if str(i) != 'nan']).lower()
    translation_table = dict.fromkeys(map(ord, '-!@#$%^&*\(\)\[\]\{\};:,./<>?\|`~=_+'), ' ')
    common_words = word_count(l.translate(translation_table), n=n)
    ret = []
    for word, _ in common_words:
        coded_feature = feature + '_' + word
        ret.append(coded_feature)
        data[coded_feature] = data[feature].str.contains(word, regex=False, na=False, case=False).astype(int)
    return ret


def generate_term_year(data, feature):
    TERM = ['spring', 'fall', 'summer']
    YEAR = [str(i) for i in list(range(2005, 2016))]
    ret = []
    for i in TERM + YEAR:
        coded_feature = feature + '_' + i
        ret.append(coded_feature)
        data[coded_feature] = data[feature].str.contains(i, regex=False, na=False, case=False).astype(int)
    return ret
