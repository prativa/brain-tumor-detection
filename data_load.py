import numpy as np
from hyperparams import Hyperparams as hp
from math import floor
from keras.utils import np_utils
from keras import backend as K


def split_valid_data(all_feature, all_label, percentage):
    all_data_size = len(all_feature)
    valid_data_size = int(floor(all_data_size * percentage))

    X_all, Y_all = shuffle(all_feature, all_label)

    X_train, Y_train = X_all[valid_data_size:], Y_all[valid_data_size:]
    X_valid, Y_valid = X_all[0:valid_data_size], Y_all[0:valid_data_size]

    return X_train, Y_train, X_valid, Y_valid

def reshape_image(X_train, X_valid):
    #input pixels matrix resizing
    if K.image_data_format() == 'channels_first':
        x_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        x_valid = X_valid.reshape(X_valid.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        x_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    return x_train, x_valid, input_shape


def load_data(features_filename, labels_filename):
    # loading features
    with open(features_filename) as features_file:
        features_str = features_file.readlines()
        features = convert_to_tensor(features_str, False) # [10897, 4096]

    # loading labels
    with open(labels_filename) as labels_file:
        labels_str = labels_file.readlines()
        labels = convert_to_tensor(labels_str, True)   # [10897, 1]

    # shuffle
    (shuffled_features, shuffled_labels) = shuffle(features,labels)

    # normalize to (0,1)
    norm_shuffled_features = shuffled_features.astype('float32') / 255.

    #split training set & validation set
    x_train, y_train, x_valid, y_valid = split_valid_data(norm_shuffled_features, shuffled_labels, split_percentage)

    # one-hot encoding
    y_train = keras.utils.to_categorical(y_train, num_classes) 
    y_valid = keras.utils.to_categorical(y_valid, num_classes)

    return (x_train, y_train), (x_valid, y_valid)

