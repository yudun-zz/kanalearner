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
        acc = len(list(filter(lambda report: report.is_successful, self.problem_report_list))) / len(
            self.problem_report_list)

        summary = 'You have learned {} hours {} minutes {} seconds today\n'.format(
            total_time_elapsed.seconds // 3600, (total_time_elapsed.seconds // 60) % 60,
            total_time_elapsed.seconds % 60) + \
                  'Solved {} Problems in total.\n'.format(len(self.problem_report_list)) + \
                  'Acc for all problems: {}'.format(acc)

        print(colored(summary, 'red', attrs=['reverse', 'blink']))
