"""Convert raw data into suitable inputs for Keras."""

import numpy as np
import pickle

from keras.utils import np_utils
from KanaRecognizer.trainer.preprocessing.data_utils import get_ETL_data
from sklearn.model_selection import train_test_split
from util.util import PROCESSED_DATA_PATH


def get_data(writers_per_char=160, mode='all', get_scripts=False, test_size=0.2):
    """
    Load the characters into a format suitable for Keras
    Args:
        writers_per_char (int): number of samples per Japanese character
        mode (string): specify the type of Japanese characters: 'all', 'hiragana', 'kanji', or 'katakana'
        get_scripts (bool): 'True' returns a label for the type of script corresponding to each Japanese character
        test_size (float): fraction of data to use for obtaining a test error
    Returns:
        X_train (np.array): training data
        Y_train (np.array): training labels
        X_test (np.array): test data
        Y_test (np.array): test labels
        labels_dict: label str list in the order of label index
    """
    characters = np.empty((0, 64, 64))
    labels = []
    scripts = []
    if mode in ('kanji', 'all'):
        for i in range(1, 4):
            if i == 3:
                max_records = 315
            else:
                max_records = 319

            if i != 1:
                start_record = 0
            else:
                start_record = 75

            chars, labs, spts = get_ETL_data(
                i, range(start_record, max_records), writers_per_char)

            characters = np.concatenate((characters, chars), axis=0)
            labels = np.concatenate((labels, labs), axis=0)
            scripts = np.concatenate((scripts, spts), axis=0)

    if mode in ('hiragana', 'all', 'kana'):
        max_records = 75
        chars, labs, spts = get_ETL_data(
            1, range(0, max_records), writers_per_char)
        print("hiragana data num:", chars.shape[0])
        characters = np.concatenate((characters, chars), axis=0)
        labels = np.concatenate((labels, labs), axis=0)
        scripts = np.concatenate((scripts, spts), axis=0)

    if mode in ('katakana', 'all', 'kana'):
        katakana_num = 0
        for i in range(7, 14):
            if i < 10:
                filename = '0' + str(i)
            else:
                filename = str(i)

            chars, labs, spts = get_ETL_data(filename, range(0, 8 if i < 13 else 3),
                                             writers_per_char, database='ETL1C')
            katakana_num += chars.shape[0]
            characters = np.concatenate((characters, chars), axis=0)
            labels = np.concatenate((labels, labs), axis=0)
            scripts = np.concatenate((scripts, spts), axis=0)

        print("katakana data num:", katakana_num)

    # rename labels from 0 to n_labels-1
    unique_labels = sorted(list(set(labels)))
    print('unique_labels num =', len(unique_labels))
    labels_dict = {unique_labels[i]: i for i in range(len(unique_labels))}
    new_labels = np.array([labels_dict[l] for l in labels], dtype=np.int32)

    if get_scripts:
        x_train, x_test, y_train, y_test = train_test_split(characters,
                                                            scripts,
                                                            test_size=test_size,
                                                            random_state=42)
    elif mode in ('all', 'kanji', 'hiragana', 'katakana', 'kana'):
        x_train, x_test, y_train, y_test = train_test_split(characters,
                                                            new_labels,
                                                            test_size=test_size,
                                                            random_state=42)
    print("X_train:", x_train.shape)
    print("y_train:", y_train.shape)
    print("X_test:", x_test.shape)
    print("y_test:", y_test.shape)

    # reshape to (1, 64, 64)
    X_train = x_train.reshape(
        (x_train.shape[0], 1, x_train.shape[1], x_train.shape[2]))
    X_test = x_test.reshape(
        (x_test.shape[0], 1, x_test.shape[1], x_test.shape[2]))

    # convert class vectors to binary class matrices
    nb_classes = len(unique_labels)
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, Y_train, X_test, Y_test, unique_labels


if __name__ == '__main__':
    # X_train, Y_train, X_test, Y_test, unique_labels = \
    #     get_data(writers_per_char=160, mode='hiragana', get_scripts=False, test_size=0.2)
    # pickle.dump(X_train, open(DATA_PATH + '/hiragana_x_train.p', 'wb'))
    # pickle.dump(Y_train, open(DATA_PATH + '/hiragana_y_train.p', 'wb'))
    # pickle.dump(X_test, open(DATA_PATH + '/hiragana_x_test.p', 'wb'))
    # pickle.dump(Y_test, open(DATA_PATH + '/hiragana_y_test.p', 'wb'))
    # print(unique_labels)

    # X_train, Y_train, X_test, Y_test, unique_labels = \
    #     get_data(writers_per_char=160, mode='katakana', get_scripts=False, test_size=0.2)
    # pickle.dump(X_train, open(PROCESSED_DATA_PATH + '/katakana_x_train.p', 'wb'))
    # pickle.dump(Y_train, open(PROCESSED_DATA_PATH + '/katakana_y_train.p', 'wb'))
    # pickle.dump(X_test, open(PROCESSED_DATA_PATH + '/katakana_x_test.p', 'wb'))
    # pickle.dump(Y_test, open(PROCESSED_DATA_PATH + '/katakana_y_test.p', 'wb'))
    # print(unique_labels)

    X_train, Y_train, X_test, Y_test, unique_labels = \
        get_data(writers_per_char=160, mode='kana', get_scripts=False, test_size=0.2)
    pickle.dump(X_train, open(PROCESSED_DATA_PATH + '/x_train.p', 'wb'))
    pickle.dump(Y_train, open(PROCESSED_DATA_PATH + '/y_train.p', 'wb'))
    pickle.dump(X_test, open(PROCESSED_DATA_PATH + '/x_test.p', 'wb'))
    pickle.dump(Y_test, open(PROCESSED_DATA_PATH + '/y_test.p', 'wb'))
    pickle.dump(unique_labels, open(PROCESSED_DATA_PATH + '/unique_labels.p', 'wb'))
    for i in range(0, len(unique_labels)//10 + 1):
        print(unique_labels[i*10: i*10 + 10])
