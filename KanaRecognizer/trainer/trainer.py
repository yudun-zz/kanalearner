"""Example job for running a neural network."""

import h5py
import numpy as np
import argparse
import os
import pickle
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import *
from keras.layers.normalization import *
from keras.optimizers import *
from util.util import *


def M6_3(weights_path=None, input_shape=(1, 64, 64), n_output=None):
    model = Sequential()

    model.add(Conv2D(64, (3, 3), input_shape=input_shape, data_format='channels_first'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(4096))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(n_output))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    if weights_path:
        model.load_weights(weights_path)
    return model


def M7_2(weights_path=None, input_shape=(1, 64, 64), n_output=None):
    model = Sequential()

    model.add(Conv2D(64, (3, 3), input_shape=input_shape, data_format='channels_first'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(192, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(n_output))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    if weights_path:
        model.load_weights(weights_path)
    return model


def load_model_weights(name, model):
    model.load_weights(name)


def save_model_weights(name, model):
    model.save_weights(name)


def load_data(mode):
    x_train = pickle.load(open(os.path.join(PROCESSED_DATA_PATH, mode + '_x_train.p'), 'rb'))
    x_test = pickle.load(open(os.path.join(PROCESSED_DATA_PATH, mode + '_x_test.p'), 'rb'))
    y_train = pickle.load(open(os.path.join(PROCESSED_DATA_PATH, mode + '_y_train.p'), 'rb'))
    y_test = pickle.load(open(os.path.join(PROCESSED_DATA_PATH, mode + '_y_test.p'), 'rb'))

    print("Data Shape:")
    print("x_train:", x_train.shape)
    print("y_train:", y_train.shape)
    print("x_test:", x_test.shape)
    print("y_test:", y_test.shape)

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', action='store', dest='mode', help='foo help')
    parser.add_argument('-c', action="store_true", default=False)
    args = parser.parse_args()
    mode = args.mode
    continue_training = args.c

    if mode == 'hiragana':
        weight_path = 'KanaRecognizer/trainer/weights/M7_2-hiragana_weights.h5'
        model_func = M7_2
    elif mode == 'katakana':
        weight_path = 'KanaRecognizer/trainer/weights/M6_3-katakana_weights.h5'
        model_func = M6_3
    else:
        exit(-1)

    X_train, y_train, X_test, y_test = load_data(mode=mode)

    n_output = y_train.shape[1]

    model = model_func(n_output=n_output, input_shape=(1, 64, 64),
                       weights_path=weight_path if continue_training else None)

    adam = Adam(lr=1e-3)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    model.fit(X_train, y_train,
              epochs=20,
              batch_size=16,
              verbose=2)

    save_model_weights(weight_path, model)

    score, acc = model.evaluate(X_test, y_test,
                                batch_size=16,
                                verbose=1)

    print("Training size: ", X_train.shape[0])
    print("Test size: ", X_test.shape[0])
    print("Test Score: ", score)
    print("Test Accuracy: ", acc)

