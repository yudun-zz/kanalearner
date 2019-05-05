import argparse
from os import listdir
from os.path import isfile, join, splitext
from util.util import HANDWRITING_HIRAGANA_LABEL_LIST, HANDWRITING_KATAKANA_LABEL_LIST, MANUAL_HANDWRITING_DATA_PATH
from KanaRecognizer.models import katakana_model, hiragana_model
from KanaRecognizer.KanaRecognizer import KanaRecognizer


def evaluate_model(mode):
    print('\nEvaluating ' + mode + ':')
    file_names = [f for f in listdir(MANUAL_HANDWRITING_DATA_PATH) if isfile(join(MANUAL_HANDWRITING_DATA_PATH, f))]

    if mode == 'hiragana':
        weight_path = 'KanaRecognizer/trainer/weights/M7_2-hiragana_weights.h5'
        model_func = hiragana_model
        kanas = [f for f in file_names if splitext(f)[0] in HANDWRITING_HIRAGANA_LABEL_LIST]
    elif mode == 'katakana':
        weight_path = 'KanaRecognizer/trainer/weights/M6_3-katakana_weights.h5'
        model_func = katakana_model
        kanas = [f for f in file_names if splitext(f)[0] in HANDWRITING_KATAKANA_LABEL_LIST]
    else:
        exit(-1)

    kanaRecognizer = KanaRecognizer(mode=mode, weight_path=weight_path, model_func=model_func)

    err_num = 0
    for kana in kanas:
        recognized_kana = kanaRecognizer.recognize(file_path=join(MANUAL_HANDWRITING_DATA_PATH, kana + '.png'))
        if not kanaRecognizer.is_equal(kana, recognized_kana):
            err_num += 1
            print('Recognize ' + kana + ' as ' + recognized_kana)

    print('acc = ', 1 - err_num / len(kanas))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', action='store', dest='mode', help='foo help')
    args = parser.parse_args()
    mode = args.mode

    evaluate_model('hiragana')
    evaluate_model('katakana')

