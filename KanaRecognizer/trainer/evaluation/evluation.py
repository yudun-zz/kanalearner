import argparse
from os import listdir
from os.path import isfile, join, splitext
from util.util import HANDWRITING_HIRAGANA_LABEL_LIST, HANDWRITING_KATAKANA_LABEL_LIST, MANUAL_HANDWRITING_DATA_PATH
from KanaRecognizer.KanaRecognizer import KanaRecognizer


def evaluate_model(mode, kanaRecognizer):
    print('\nEvaluating ' + mode + ':')
    file_names = [f for f in listdir(MANUAL_HANDWRITING_DATA_PATH) if isfile(join(MANUAL_HANDWRITING_DATA_PATH, f))]

    if mode == 'hiragana':
        kana_file_names = [f for f in file_names if splitext(f)[0] in HANDWRITING_HIRAGANA_LABEL_LIST]
    elif mode == 'katakana':
        kana_file_names = [f for f in file_names if splitext(f)[0] in HANDWRITING_KATAKANA_LABEL_LIST]
    else:
        exit(1)

    err_num = 0
    for kana_file_name in kana_file_names:
        recognized_kana = kanaRecognizer.recognize(
            file_path=join(MANUAL_HANDWRITING_DATA_PATH, kana_file_name),
            mode=mode
        )
        kana = splitext(kana_file_name)[0]
        if kana != recognized_kana:
            err_num += 1
            print('Recognize ' + kana + ' as ' + recognized_kana)

    print('acc = ', 1 - err_num / len(kana_file_names))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', action='store', dest='mode', help='hiragana or katakana')
    args = parser.parse_args()
    mode = args.mode

    kanaRecognizer = KanaRecognizer()

    evaluate_model('hiragana', kanaRecognizer)
    evaluate_model('katakana', kanaRecognizer)
