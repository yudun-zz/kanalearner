from timeit import default_timer as timer

from Problem.Problem import Problem
from Problem.ProblemReport import ProblemReport


class HandWritingProblem(Problem):

    def __init__(self, question, correct_answer, problem_type):
        super(HandWritingProblem, self).__init__(question, correct_answer, problem_type)

    def run(self):
        start = timer()
        # Give user some time to think or write the desired character
        # TODO: Replace this with HandWritingFetcher and KanaRecognizer
        input(self.PROMPT_TEMPLATE_MAP[self.problem_type].format(self.question))

        # Show correct answer and let user decide if he got it correct
        print(self.correct_answer)
        is_successful = (input().strip() == '')

        return ProblemReport(
            problem_type=self.problem_type,
            is_successful=is_successful,
            question=self.question,
            correct_answer=self.correct_answer,
            # TODO: We are not able to get the user handwriting answer for now, will update as our
            # OCR mpdel ready
            actual_answer='',
            time_elapsed_in_sec=timer() - start
        )
