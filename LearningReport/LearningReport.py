import datetime

from termcolor import colored


class LearningReport(object):
    def __init__(self):
        self.problem_report_list = []

    def add(self, problem_report):
        self.problem_report_list.append(problem_report)

    def show(self):
        total_time_elapsed = datetime.timedelta(
            seconds=sum([report.time_elapsed_in_sec for report in self.problem_report_list]))
        total_question_num = len(self.problem_report_list)
        wrong_num = total_question_num - len(
            list(filter(lambda report: report.is_successful, self.problem_report_list)))
        acc = (1 - wrong_num / total_question_num) if total_question_num != 0 else 0.0

        sorted_problem_reports = sorted([problem_report for problem_report in self.problem_report_list],
                                        key=lambda problem_report: problem_report.time_elapsed_in_sec, reverse=True)

        summary = 'You have learned {} hours {} minutes {} seconds today\n'.format(
            total_time_elapsed.seconds // 3600, (total_time_elapsed.seconds // 60) % 60,
            total_time_elapsed.seconds % 60) + \
                  'Solved {} Problems in total.\n'.format(total_question_num) + \
                  'Spent {} seconds per question.\n'.format(total_time_elapsed.seconds / total_question_num if total_question_num != 0 else 0.0) + \
                  '{} problems get wrong answers.\n'.format(wrong_num) + \
                  'Acc for all problems: {0:.2f}\n'.format(acc) + \
                  '\nAll wrong questions:\n' + ''.join(['%s\n' % (problem_report.to_probelm_str()
                    ) for problem_report in self.problem_report_list if not problem_report.is_successful]) + \
                  '\nTop 10 slowest questions:\n' + ''.join(['%.2fs\t%s\n' % (
                        problem_report.time_elapsed_in_sec, problem_report.to_probelm_str()
                    ) for problem_report in sorted_problem_reports[:10]])

        print(colored(summary, 'red', attrs=['reverse', 'blink']))
