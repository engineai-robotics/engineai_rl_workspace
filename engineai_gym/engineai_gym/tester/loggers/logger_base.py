import os
from abc import ABC
from collections import defaultdict
import csv


class LoggerBase(ABC):
    def __init__(self, name, env, time, test_dir, extra_args):
        self.name = name
        self.env = env
        self.time = time
        self.log = defaultdict(list)
        self.extra_args = extra_args
        class_name = self.__class__.__name__
        if class_name != "LoggerBase":
            self.test_dir = os.path.join(test_dir, name)

    def retrieve_data(self) -> None:
        raise NotImplementedError

    def log_data(self, extra_data) -> None:
        raise NotImplementedError

    def add_data_to_log(self, data_dict):
        for key, value in data_dict.items():
            self.log[key].append(value)

    def save_to_csv(
        self, file_name, lists, axis_label, header, label=[], type="separate"
    ):
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
        with open(
            os.path.join(self.test_dir, file_name + ".csv"), mode="w", newline=""
        ) as file:
            writer = csv.writer(file)
            writer.writerow([type])
            writer.writerow(header + ["time"])
            writer.writerow([axis_label] * len(lists) + ["time"])
            writer.writerow(label)
            for row in zip(*lists, self.time):
                writer.writerow(row)
