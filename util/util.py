HIRAGANA = 'hiragana'
KATAKANA = 'katakana'

# Specify the path to the processed files
PROCESSED_DATA_PATH = 'KanaRecognizer/trainer/processed_data'
MANUAL_HANDWRITING_DATA_PATH = 'hand_writing_data'
PROBLEMS_CACHE_PATH = 'problems_cache/problems_cache.p'

# Path templates of trained model
HIRAGANA_MODEL_PATH_TEMP = 'KanaRecognizer/trainer/weights/{}-hiragana_weights.h5'
KATAKANA_MODEL_PATH_TEMP = 'KanaRecognizer/trainer/weights/{}-katakana_weights.h5'


def get_model_path_from_model_name(mode, model_name):
    if mode == HIRAGANA:
        return HIRAGANA_MODEL_PATH_TEMP.format(model_name)
    elif mode == KATAKANA:
        return KATAKANA_MODEL_PATH_TEMP.format(model_name)
    else:
        raise Exception('Unrecognized mode', mode)


PRONUNCIATION_PROBLEM_SET_LIST = [
    # voiceless sound 0-10
    ['a', 'i', 'u', 'e', 'o'],
    ['ka', 'ki', 'ku', 'ke', 'ko'],
    ['sa', 'shi', 'su', 'se', 'so'],
    ['ta', 'chi', 'tsu', 'te', 'to'],
    ['na', 'ni', 'nu', 'ne', 'no'],
    ['ha', 'hi', 'fu', 'he', 'ho'],
    ['ma', 'mi', 'mu', 'me', 'mo'],
    ['ya', 'yu', 'yo'],
    ['ra', 'ri', 'ru', 're', 'ro'],
    ['wa', 'wo'],
    ['n'],

    # voiced sound 11 - 15
    ['ga', 'gi', 'gu', 'ge', 'go'],
    ['za', 'zi', 'zu', 'ze', 'zo'],
    ['da', 'di', 'du', 'de', 'do'],
    ['ba', 'bi', 'bu', 'be', 'bo'],
    ['pa', 'pi', 'pu', 'pe', 'po'],
]

HANDWRITING_HIRAGANA_LABEL_LIST = \
    ['あ', 'い', 'う', 'え', 'お', 'か', 'が', 'き', 'ぎ', 'く', 'ぐ', 'け', 'げ', 'こ', 'ご', 'さ',
     'ざ', 'し', 'じ', 'す', 'ず', 'せ', 'ぜ', 'そ', 'ぞ', 'た', 'だ', 'ち', 'ぢ', 'つ', 'づ', 'て',
     'で', 'と', 'ど', 'な', 'に', 'ぬ', 'ね', 'の', 'は', 'ば', 'ぱ', 'ひ', 'び', 'ぴ', 'ふ', 'ぶ',
     'ぷ', 'へ', 'べ', 'ぺ', 'ほ', 'ぼ', 'ぽ', 'ま', 'み', 'む', 'め', 'も', 'や', 'ゆ', 'よ', 'ら',
     'り', 'る', 'れ', 'ろ', 'わ', 'を', 'ん']

PRONUNCIATION_LIST = \
    ['a', 'i', 'u', 'e', 'o', 'ka', 'ga', 'ki', 'gi', 'ku', 'gu', 'ke', 'ge', 'ko',
     'go', 'sa', 'za', 'shi', 'ji', 'su', 'zu', 'se', 'ze', 'so', 'zo', 'ta', 'da',
     'chi', 'di', 'tsu', 'du', 'te', 'de', 'to', 'do', 'na', 'ni', 'nu', 'ne', 'no',
     'ha', 'ba', 'pa', 'hi', 'bi', 'pi', 'fu', 'bu', 'pu', 'he', 'be', 'pe', 'ho',
     'bo', 'po', 'ma', 'mi', 'mu', 'me', 'mo', 'ya', 'yu', 'yo', 'ra', 'ri', 'ru',
     're', 'ro', 'wa', 'wo', 'n']

KATAKANA_LIST = \
    ['ア', 'イ', 'ウ', 'エ', 'オ', 'カ', 'ガ', 'キ', 'ギ', 'ク', 'グ', 'ケ', 'ゲ', 'コ', 'ゴ', 'サ',
     'ザ', 'シ', 'ジ', 'ス', 'ズ', 'セ', 'ゼ', 'ソ', 'ゾ', 'タ', 'ダ', 'チ', 'ヂ', 'ツ', 'ヅ', 'テ',
     'デ', 'ト', 'ド', 'ナ', 'ニ', 'ヌ', 'ネ', 'ノ', 'ハ', 'バ', 'パ', 'ヒ', 'ビ', 'ピ', 'フ', 'ブ',
     'プ', 'ヘ', 'ベ', 'ペ', 'ホ', 'ボ', 'ポ', 'マ', 'ミ', 'ム', 'メ', 'モ', 'ヤ', 'ユ', 'ヨ', 'ラ',
     'リ', 'ル', 'レ', 'ロ', 'ワ', 'ヲ', 'ン']

HANDWRITING_KATAKANA_LABEL_LIST = \
    ['ア', 'イ', 'ウ', 'エ', 'オ', 'カ', 'キ', 'ク', 'ケ', 'コ', 'サ', 'シ', 'ス', 'セ', 'ソ', 'タ',
     'チ', 'ツ', 'テ', 'ト', 'ナ', 'ニ', 'ヌ', 'ネ', 'ノ', 'ハ', 'ヒ', 'フ', 'ヘ', 'ホ', 'マ', 'ミ',
     'ム', 'メ', 'モ', 'ヤ', 'ユ', 'ヨ', 'ラ', 'リ', 'ル', 'レ', 'ロ', 'ワ', 'ヲ', 'ン']

# Some HIRAGANA need to be mapped to equivalent character to improve accuracy
RAW_HIRAGANA_TO_LABEL_KATAKANA_MAP = {
    'ゃ': 'や',
    'ゅ': 'ゆ',
    'ょ': 'よ',
    'っ': 'つ',
}

# All KATAKANA need to be mapped to different font supported by romkan
RAW_KATAKANA_TO_LABEL_KATAKANA_MAP = {
    'ｦ': 'ヲ',
    'ｨ': 'イ',
    'ｪ': 'エ',
    'ｱ': 'ア',
    'ｲ': 'イ',
    'ｳ': 'ウ',
    'ｴ': 'エ',
    'ｵ': 'オ',
    'ｶ': 'カ',
    'ｷ': 'キ',
    'ｸ': 'ク',
    'ｹ': 'ケ',
    'ｺ': 'コ',
    'ｻ': 'サ',
    'ｼ': 'シ',
    'ｽ': 'ス',
    'ｾ': 'セ',
    'ｿ': 'ソ',
    'ﾀ': 'タ',
    'ﾁ': 'チ',
    'ﾂ': 'ツ',
    'ﾃ': 'テ',
    'ﾄ': 'ト',
    'ﾅ': 'ナ',
    'ﾆ': 'ニ',
    'ﾇ': 'ヌ',
    'ﾈ': 'ネ',
    'ﾉ': 'ノ',
    'ﾊ': 'ハ',
    'ﾋ': 'ヒ',
    'ﾌ': 'フ',
    'ﾍ': 'ヘ',
    'ﾎ': 'ホ',
    'ﾏ': 'マ',
    'ﾐ': 'ミ',
    'ﾑ': 'ム',
    'ﾒ': 'メ',
    'ﾓ': 'モ',
    'ﾔ': 'ヤ',
    'ﾕ': 'ユ',
    'ﾖ': 'ヨ',
    'ﾗ': 'ラ',
    'ﾘ': 'リ',
    'ﾙ': 'ル',
    'ﾚ': 'レ',
    'ﾛ': 'ロ',
    'ﾜ': 'ワ',
    'ﾝ': 'ン'
}
