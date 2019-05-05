import numpy as np
from PIL import Image
from util.util import HANDWRITING_HIRAGANA_LABEL_LIST, HANDWRITING_KATAKANA_LABEL_LIST, HIRAGANA, KATAKANA, get_model_path_from_model_name
from KanaRecognizer.models import M6_3, M7_2


class KanaRecognizer(object):
    def __init__(self, hiragana_model=M7_2, katakana_model=M6_3):
        self.hiragana_model = hiragana_model(
            weights_path=get_model_path_from_model_name(mode=HIRAGANA, model_name=hiragana_model.__name__),
            input_shape=(1, 64, 64),
            n_output=len(HANDWRITING_HIRAGANA_LABEL_LIST))
        self.katakana_model = katakana_model(
            weights_path=get_model_path_from_model_name(mode=KATAKANA, model_name=katakana_model.__name__),
            input_shape=(1, 64, 64),
            n_output=len(HANDWRITING_KATAKANA_LABEL_LIST))

    def _recognize_hiragana(self, img):
        pass

    def _recognize_katakana(self, img):
        pass

    def recognize(self, file_path, mode):
        img_arr = np.uint8(np.array(Image.open(file_path).convert('1')))
        if mode == HIRAGANA:
            return self._recognize_hiragana(img_arr)
        elif mode == KATAKANA:
            return self._recognize_hiragana(img_arr)
        else:
            raise Exception('Unrecognized mode', mode)

