"""Example job for running a neural network."""

import argparse
import os
import pickle
from util.util import PROCESSED_DATA_PATH, HIRAGANA, KATAKANA, get_model_path_from_model_name
from keras.optimizers import Adam
from KanaRecognizer.models import M7_2, M6_3

HIRAGANA_MODEL = M7_2
KATAKANA_MODEL = M6_3


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

    if mode == HIRAGANA:
        model_func = HIRAGANA_MODEL
    elif mode == KATAKANA:
        model_func = KATAKANA_MODEL
    else:
        raise Exception('Unrecognized mode', mode)

    weight_path = get_model_path_from_model_name(mode=mode, model_name=model_func.__name__)

    X_train, y_train, X_test, y_test = load_data(mode=mode)
    n_output = y_train.shape[1]

    model = model_func(n_output=n_output, input_shape=(1, 64, 64),
                       weights_path=weight_path if continue_training else None)

    adam = Adam(lr=1e-3)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    model.fit(X_train, y_train,
              epochs=20,
              validation_data=(X_test, y_test),
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

