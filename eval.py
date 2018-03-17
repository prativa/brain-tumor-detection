import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.utils import np_utils, load_model
from keras.callbacks import ModelCheckpoint
from hyperparams import Hyperparams as hp

# load testing data
x_test, y_test = load_test_data(hp.test_fname_X, hp.test_fname_Y)

# load model
tumor_recognizer = load_model(hp.model_fname)

# inference
infer_result = tumor_recognizer.predict_classes(x_test)

# test accuracy
#true_positive 
#false_positive


