class Goals:
    def __init__(self, env):
        self.env = env

    def commands(self):
        return self.env.commands * self.env.commands_scales
