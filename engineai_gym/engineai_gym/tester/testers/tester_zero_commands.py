from .tester_base import TesterTypeBase


class TesterZeroCommands(TesterTypeBase):
    def set_commands(self) -> None:
        self.env.commands[:] = 0
        return self.env.goal_dict
