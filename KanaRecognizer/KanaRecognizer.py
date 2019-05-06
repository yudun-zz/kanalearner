import numpy as np
from PIL import Image, ImageOps
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

    def _recognize_hiragana(self, img_arr):
        return HANDWRITING_HIRAGANA_LABEL_LIST[
            np.argmax(self.hiragana_model.predict(img_arr))
        ]

    def _recognize_katakana(self, img_arr):
        return HANDWRITING_KATAKANA_LABEL_LIST[
            np.argmax(self.katakana_model.predict(img_arr))
        ]

    def center_(self, img):
        # getbbox will return box of the non-zero regions, so we need to invert
        # the img to make the surrounding border to be black (zero)
        inverted_img = ImageOps.invert(img)
        bbox = inverted_img.getbbox()

        new_w = bbox[2] - bbox[0]
        new_h = bbox[3] - bbox[1]
        ori_w = img.size[0]
        ori_h = img.size[1]

        pw = ori_w - new_w
        ph = ori_h - new_h

        inverted_img = inverted_img.crop(bbox)
        padding = (pw//2, ph//2, pw-(pw//2), ph-(ph//2))

        # Inverted it back to make the background to be white (non-zero)
        return ImageOps.invert(ImageOps.expand(inverted_img, padding))

    def recognize(self, file_path, mode):
        img = Image.open(file_path)
        img = self.center_(img)
        img_arr = np.uint8(np.array(img.convert('1')))
        assert len(img_arr.shape) == 2
        img_arr = img_arr.reshape(1, 1, img_arr.shape[0], img_arr.shape[1])
        if mode == HIRAGANA:
            return self._recognize_hiragana(img_arr)
        elif mode == KATAKANA:
            return self._recognize_katakana(img_arr)
        else:
            raise Exception('Unrecognized mode', mode)

