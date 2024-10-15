from abc import ABC
from engineai_rl_lib.class_operations import instance_name_to_class_name
from engineai_gym.tester.loggers import *


class TesterTypeBase(ABC):
    def __init__(self, name, loggers, env, time, test_dir, extra_args):
        self.name = name
        self.env = env
        if self.__class__.__name__ != "TesterBase":
            self.test_dir = os.path.join(test_dir, name)
        self.loggers = []
        for key, value in loggers.items():
            logger_class = eval(instance_name_to_class_name(value))
            self.loggers.append(
                logger_class(key, env, time, os.path.join(test_dir, name), extra_args)
            )

    def set_commands(self):
        return self.env.goal_dict

    def start_record_video(self):
        self.env.start_recording_video()

    def end_and_save_recording_video(self):
        self.env.end_and_save_recording_video(self.name + ".mp4")
