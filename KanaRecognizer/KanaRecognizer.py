from util.util import HANDWRITING_HIRAGANA_LABEL_LIST, HANDWRITING_KATAKANA_LABEL_LIST


class KanaRecognizer(object):
    def __init__(self, mode, weight_path, model_func):
        self.mode = mode
        self.weight_path = weight_path
        self.model_func = model_func

    def recognize(self, file_path):
        pass
