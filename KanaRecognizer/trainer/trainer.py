"""Example job for running a neural network."""

import h5py
import numpy as np
import os
import pickle
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import *
from keras.layers.normalization import *
from keras.optimizers import *
from util.util import *


def M7_1(weights_path=None, input_shape=(1, 64, 64), n_output=None):
    model = Sequential()

    model.add(Conv2D(64, (3, 3), input_shape=input_shape, data_format='channels_first'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(192, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(n_output))
    model.add(Activation('softmax'))
    if weights_path:
        model.load_weights(weights_path)
    return model


def load_model_weights(name, model):
    model.load_weights(name)


def save_model_weights(name, model):
    model.save_weights(name)


def load_data():
    x_train = pickle.load(open(os.path.join(PROCESSED_DATA_PATH, 'x_train.p'), 'rb'))
    x_test = pickle.load(open(os.path.join(PROCESSED_DATA_PATH, 'x_test.p'), 'rb'))
    y_train = pickle.load(open(os.path.join(PROCESSED_DATA_PATH, 'y_train.p'), 'rb'))
    y_test = pickle.load(open(os.path.join(PROCESSED_DATA_PATH, 'y_test.p'), 'rb'))

    print("Data Shape:")
    print("x_train:", x_train.shape)
    print("y_train:", y_train.shape)
    print("x_test:", x_test.shape)
    print("y_test:", y_test.shape)

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data()

    n_output = y_train.shape[1]

    model = M7_1(n_output=n_output, input_shape=(1, 64, 64), weights_path=None)

    # load_model_weights('weights/M7_1-hiragana_weights.h5', model)

    adam = Adam(lr=1e-4)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    model.fit(X_train, y_train,
              epochs=20,
              batch_size=16,
              verbose=2)

    save_model_weights('KanaRecognizer/trainer/weights/M16-hiragana_weights.h5', model)

    score, acc = model.evaluate(X_test, y_test,
                                batch_size=16,
                                verbose=1)
    print("Training size: ", X_train.shape[0])
    print("Test size: ", X_test.shape[0])
    print("Test Score: ", score)
    print("Test Accuracy: ", acc)

