import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow as tf

from utils import summarize_offline_metrics
from sklearn.model_selection import train_test_split
import scipy
import numpy as np

LOGDIR = 'logdir/'


def get_callbacks(name,patience):
    return [
        tfdocs.modeling.EpochDots(report_every=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy',
                                         patience=patience),
        tf.keras.callbacks.TensorBoard(LOGDIR + '/' + name),
    ]

def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


def dnn(input,
        hidden_units,
        output_units,
        activation,
        dropout=None,
        batch_norm=None,
        initializer=None,
        regularizer=None):
    net = input
    # build the input layer; input as tensors
    if (dropout is not None) and dropout[0] > 0:
        net.add(tf.keras.layers.Dropout(rate=dropout[0]))

    for layer_id, num_hidden_units in enumerate(hidden_units):
        net.add(tf.keras.layers.Dense(units=num_hidden_units,
                                      activation=activation,
                                      kernel_initializer=initializer,
                                      kernel_regularizer=regularizer))
        if (batch_norm is not None) and batch_norm[layer_id]:
            net.add(tf.keras.layers.BatchNormalization())
        if (dropout is not None) and dropout[layer_id + 1] > 0:
            net.add(tf.keras.layers.Dropout(rate=dropout[layer_id + 1]))
    # define the output layer
    output = tf.keras.layers.Dense(units=output_units,
                                   activation='sigmoid',
                                   kernel_initializer=initializer)
    net.add(output)
    return net


def compile_and_fit(model,
                    name,
                    train_dataset,
                    validation_dataset,
                    steps_per_epoch,
                    optimizer=tf.keras.optimizers.Adam(),
                    max_epochs=100,
                    base_lr=0.005,
                    per_epoch_decay=10,
                    patience=5):
    if optimizer == 'decay':
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            base_lr,
            decay_steps=steps_per_epoch // per_epoch_decay,
            decay_rate=1,
            staircase=False)
        optimizer = tf.keras.optimizers.Adam(lr_schedule)

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[
                      tf.keras.losses.BinaryCrossentropy(name='binary_crossentropy'),
                      'accuracy', 'AUC'])

    model.summary()

    history = model.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=max_epochs,
        validation_data=validation_dataset,
        callbacks=get_callbacks(name, patience=patience),
        verbose=0)
    return history


# input layer
def construct_dnn_estimators(params, n_features, output_unit=1, activation='relu'):
    estimators = dict()
    for k, param in params.items():
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(n_features,)))
        estimators.update({k: dnn(input=model, output_units=output_unit, activation=activation, **param)})
    return estimators


def train_dnn_estimators(X_train,
                         y_train,
                         dnn_params,
                         validation_set_size=0.2,
                         shuffle_buffer_size=100,
                         batch_size=64,
                         max_epochs=100,
                         base_lr=0.005,
                         per_epoch_decay=10,
                         patience=5,
                         preprocessor=None,
                         seed=1234567):
    histories = {}
    dnn_X_train, dnn_X_validation, dnn_y_train, dnn_y_validation = train_test_split(X_train,
                                                                                    y_train,
                                                                                    test_size=validation_set_size,
                                                                                    random_state=seed)

    print("----------")
    print(dnn_X_train.shape, dnn_X_validation.shape, dnn_y_train.shape, dnn_y_validation.shape)

    if preprocessor:
        dnn_X_train = preprocessor.transform(dnn_X_train)
        dnn_X_validation = preprocessor.transform(dnn_X_validation)

    if scipy.sparse.issparse(dnn_X_train):
        dnn_X_train = convert_sparse_matrix_to_sparse_tensor(dnn_X_train)
        dnn_X_validation = convert_sparse_matrix_to_sparse_tensor(dnn_X_validation)

    n_train, n_features = dnn_X_train.shape
    steps_per_epoch = n_train / batch_size
    estimators = construct_dnn_estimators(params=dnn_params, n_features=n_features)

    train_dataset = tf.data.Dataset.from_tensor_slices((dnn_X_train, dnn_y_train))
    validation_dataset = tf.data.Dataset.from_tensor_slices((dnn_X_validation, dnn_y_validation))

    train_dataset = train_dataset.shuffle(shuffle_buffer_size).repeat().batch(batch_size)
    validation_dataset = validation_dataset.batch(batch_size)
    for k, estimator in estimators.items():
        histories[k] = compile_and_fit(model=estimator,
                                       name=k,
                                       train_dataset=train_dataset,
                                       validation_dataset=validation_dataset,
                                       steps_per_epoch=steps_per_epoch,
                                       max_epochs=max_epochs,
                                       base_lr=base_lr,
                                       per_epoch_decay=per_epoch_decay,
                                       patience=patience)
    return estimators,histories


def evaluate_dnn_estimator_on_test(estimators, X_test, y_test, preprocessor=None):
    y_predicted_list = dict()
    if preprocessor:
        X_test = preprocessor.transform(X_test)

    if scipy.sparse.issparse(X_test):
        X_test = convert_sparse_matrix_to_sparse_tensor(X_test)

    for k, estimator in estimators.items():
        y_predicted = estimator.predict(X_test)
        roc_auc, pr_auc, logloss = summarize_offline_metrics(y_true=y_test, y_score=y_predicted)
        y_predicted_list.update({k: {'y_predicted': y_predicted,
                                     'roc_auc': roc_auc,
                                     'pr_auc': pr_auc,
                                     'logloss': logloss}})

    return y_predicted_list
