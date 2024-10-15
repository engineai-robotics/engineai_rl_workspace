from .tester_base import TesterTypeBase
from engineai_rl_lib.command_filter import convert_to_visible_commands


class TesterNormalCommands(TesterTypeBase):
    def set_commands(self) -> None:
        convert_to_visible_commands(self.env.commands)
        return self.env.goal_dict
