from Problem.Problem import Problem


class ProblemReport(object):
    def __init__(self, problem_type, is_successful, question, correct_answer, actual_answer, time_elapsed_in_sec):
        self.problem_type = problem_type
        self.is_successful = is_successful
        self.question = question
        self.correct_answer = correct_answer
        self.actual_answer = actual_answer
        self.time_elapsed_in_sec = time_elapsed_in_sec

    def to_probelm_str(self):
        return Problem.PROMPT_TEMPLATE_MAP[self.problem_type].format(self.question)
