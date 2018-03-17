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
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from data_load import load_data
from hyperparams import Hyperparams as hp

(x_train, y_train), (x_valid, y_valid) = load_data(features_filename, labels_filename)
x_train, x_valid = reshape_image(x_train, x_valid)


#data augmentation
datagen = ImageDataGenerator(featurewise_center=False, 
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            zoom_range=0.3,
                            shear_range=0.2,
                            rotation_range=10.,)
datagen.fit(x_train)

#build model
model = Sequential()
model.add(Convolution2D(64, (3, 3), padding='same', input_shape=input_shape))
model.add(keras.layers.advanced_activations.PReLU())
model.add(BatchNormalization())
model.add(Convolution2D(64, (3, 3), padding='same'))  # 刪掉？
model.add(keras.layers.advanced_activations.PReLU())
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Dropout(0.3))
model.add(keras.layers.advanced_activations.PReLU())
model.add(Dense(num_classes, activation = 'sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=x_train.shape[0]/batch_size,
                    epochs=epochs,
                    validation_data=(x_valid, y_valid),
                    callbacks=[earlyStopping,
                    ModelCheckpoint('tumor_recognizer1',save_best_only=True)])

print("Model training is completed.")