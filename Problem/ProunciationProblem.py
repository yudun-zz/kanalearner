from timeit import default_timer as timer

from termcolor import colored

from Problem.Problem import Problem
from Problem.Problem import ProblemType
from Problem.ProblemReport import ProblemReport


class ProunciationProblem(Problem):
    def __init__(self, question, correct_answer):
        super(ProunciationProblem, self).__init__(question, correct_answer, ProblemType.TO_PROUNCIATION)

    def run(self):
        start = timer()
        prounciation = input(self.PROMPT_TEMPLATE_MAP[ProblemType.TO_PROUNCIATION].format(self.question))
        is_successful = prounciation == self.correct_answer

        if not is_successful:
            print(colored(self.correct_answer, 'red', attrs=['reverse', 'blink']))

        return ProblemReport(
            problem_type=self.problem_type,
            is_successful=is_successful,
            question=self.question,
            correct_answer=self.correct_answer,
            actual_answer=prounciation,
            time_elapsed_in_sec=timer() - start
        )
