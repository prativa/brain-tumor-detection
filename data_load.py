import numpy as np
from math import floor
from hyperparam import Hyperparams as hp

def convert_to_tensor(str_file, isLabel):
    raw_data = [data.split(',') for data in str_file]
    dataset = []
    for per_data in raw_data:
        if isLabel:
            dataset = [int(data) for data in per_data]
        else:
            num_data = [float(data) for data in per_data]
            dataset.append(num_data)
    
    tensor = np.array(dataset)
    return tensor

def shuffle(X, Y):
    randomizer = np.arange(len(X))
    np.random.shuffle(randomizer)
    return (X[randomizer], Y[randomizer])

def split_valid_data(all_feature, all_label, percentage):
    all_data_size = len(all_feature)
    valid_data_size = int(floor(all_data_size * percentage))
    X_all, Y_all = shuffle(all_feature, all_label) 

    #split train & valid set
    X_train, Y_train = X_all[valid_data_size:], Y_all[valid_data_size:]
    X_valid, Y_valid = X_all[0:valid_data_size], Y_all[0:valid_data_size]

    # save validation set as testing data
    np.save('data/eval_X.npy',X_valid)
    np.save('data/eval_Y.npy', Y_valid)

    return X_train, Y_train, X_valid, Y_valid

def image_reshape(img_data, img_rows, img_cols):
    #input pixels matrix resizing
    if K.image_data_format() == 'channels_first': # (channels, cols, rows)
        img_data = img_data.reshape(img_data.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else: # (cols, rows, channels)
        img_data = img_data.reshape(img_data.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    return img_data, input_shape

def load_train_data(features_fname, labels_fname):
    # loading features
    with open(features_fname) as features_file:
        features_str = features_file.readlines()
        features = convert_to_tensor(features_str, False) # [10897, 4096]

    # loading labels
    with open(labels_fname) as labels_file:
        labels_str = labels_file.readlines()
        labels = convert_to_tensor(labels_str, True)   # [10897, 1]

    (shuffled_features, shuffled_labels) = shuffle(features,labels)
    norm_shuffled_features = shuffled_features.astype('float32') / 255.
    x_train, y_train, x_valid, y_valid = split_valid_data(norm_shuffled_features, shuffled_labels, hp.split_rate)

    # features: images reshape
    x_train = image_reshape(x_train)
    x_valid = image_reshape(x_valid)

    # labels: one-hot encoding
    y_train = keras.utils.to_categorical(y_train, num_classes) 
    y_valid = keras.utils.to_categorical(y_valid, num_classes)

    return x_train, y_train, x_valid, y_valid

def load_test_data(testX_fname, testY_fname):
    test_features = np.load(hp.testX_fname)
    test_labels = np.load(hp.testY_fname)
    x_test = image_reshape(test_features)

    return x_test, test_labels