from .tester_base import TesterTypeBase
from engineai_rl_lib.command_filter import convert_to_visible_commands


class TesterYCommands(TesterTypeBase):
    def set_commands(self) -> None:
        convert_to_visible_commands(self.env.commands)
        self.env.commands[:, 0] = 0
        self.env.commands[:, 2] = 0
        return self.env.goal_dict
