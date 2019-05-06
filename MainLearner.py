import argparse
import pickle
import random

import romkan

from LearningReport.LearningReport import LearningReport
from Problem.HandWritingProblem import HandWritingProblem
from Problem.Problem import ProblemType
from Problem.ProunciationProblem import ProunciationProblem
from util.util import PRONUNCIATION_PROBLEM_SET_LIST, PROBLEMS_CACHE_PATH


def init_problem_list(prounciation_problem_set_list):
    problems_list = []
    for prounciation_problem_set in prounciation_problem_set_list:
        for prounciation in prounciation_problem_set:
            hiragana = romkan.to_hiragana(prounciation)
            katakana = romkan.to_katakana(prounciation)

            problems_list.append(ProunciationProblem(
                question=katakana,
                correct_answer=prounciation,
            ))
            problems_list.append(ProunciationProblem(
                question=hiragana,
                correct_answer=prounciation,
            ))

            problems_list.append(HandWritingProblem(
                question=prounciation,
                correct_answer=hiragana,
                problem_type=ProblemType.PROUNCIATION_TO_HIRAGANA
            ))
            problems_list.append(HandWritingProblem(
                question=prounciation,
                correct_answer=katakana,
                problem_type=ProblemType.PROUNCIATION_TO_KATAKANA
            ))
            problems_list.append(HandWritingProblem(
                question=katakana,
                correct_answer=hiragana,
                problem_type=ProblemType.KATAKANA_TO_HIRAGANA
            ))
            problems_list.append(HandWritingProblem(
                question=hiragana,
                correct_answer=katakana,
                problem_type=ProblemType.HIRAGANA_TO_KATAKANA
            ))

    return problems_list


def run_problems(problems_list):
    learning_report = LearningReport()
    problem_idx = 0
    new_problems_list = []

    while len(problems_list) > 0:
        random.shuffle(problems_list)
        try:
            for problem_idx in range(0, len(problems_list)):
                problem = problems_list[problem_idx]
                print()
                report = problem.run()

                learning_report.add(report)

                if not report.is_successful:
                    new_problems_list.append(problem)
        except KeyboardInterrupt:
            # User Exit the learning, we should still save the learning progress
            break

        # update the problems_list to be unsolved problems and proceed to next round
        problems_list = new_problems_list
        new_problems_list = []

    remaining_problems = new_problems_list + problems_list[problem_idx + 1:]
    if len(remaining_problems) > 0:
        pickle.dump(remaining_problems, open(PROBLEMS_CACHE_PATH, 'wb'))

    return learning_report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--si', action='store', dest='start_index', default=0,
                        help='start index of the problem set list')
    parser.add_argument('--ei', action='store', dest='end_index', default=len(PRONUNCIATION_PROBLEM_SET_LIST),
                        help='end index of the problem set list')
    parser.add_argument('-c', action="store_true", default=False)

    args = parser.parse_args()
    start_index = int(args.start_index)
    end_index = int(args.end_index)
    continue_last_learning = args.c

    # Get the list of problems we want to be trained on
    if continue_last_learning:
        problems_list = pickle.load(open(PROBLEMS_CACHE_PATH, 'rb'))
    else:
        problems_list = init_problem_list(PRONUNCIATION_PROBLEM_SET_LIST[start_index:end_index])

    # Run these problem and collect learning reports
    learning_report = run_problems(problems_list)

    learning_report.show()
