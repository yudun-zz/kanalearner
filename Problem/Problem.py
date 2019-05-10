from enum import Enum


class ProblemType(Enum):
    HIRAGANA_TO_KATAKANA = 1
    KATAKANA_TO_HIRAGANA = 2
    PROUNCIATION_TO_HIRAGANA = 3
    PROUNCIATION_TO_KATAKANA = 4
    TO_PROUNCIATION = 5


class Problem(object):
    PROMPT_TEMPLATE_MAP = {
        ProblemType.TO_PROUNCIATION: 'Say {}:',
        ProblemType.KATAKANA_TO_HIRAGANA: 'Give hiragana for {}:',
        ProblemType.HIRAGANA_TO_KATAKANA: 'Give katakana for {}:',
        ProblemType.PROUNCIATION_TO_HIRAGANA: 'Give hiragana for {}:',
        ProblemType.PROUNCIATION_TO_KATAKANA: 'Give katakana for {}:',
    }

    def __init__(self, question, correct_answer, problem_type):
        self.question = question
        self.correct_answer = correct_answer
        self.problem_type = problem_type

    def __str__(self):
        return self.PROMPT_TEMPLATE_MAP[self.problem_type].format(self.question)

    def run(self):
        pass
