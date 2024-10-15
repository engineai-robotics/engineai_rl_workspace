from engineai_gym.tester.loggers.logger_base import LoggerBase


class LoggerTypeBaseVel(LoggerBase):
    def log_data(self, extra_data):
        self.add_data_to_log(
            {
                "command_x": self.env.commands[
                    self.extra_args["robot_index"], 0
                ].item(),
                "command_y": self.env.commands[
                    self.extra_args["robot_index"], 1
                ].item(),
                "command_yaw": self.env.commands[
                    self.extra_args["robot_index"], 2
                ].item(),
                "base_vel_x": self.env.base_lin_vel[
                    self.extra_args["robot_index"], 0
                ].item(),
                "base_vel_y": self.env.base_lin_vel[
                    self.extra_args["robot_index"], 1
                ].item(),
                "base_vel_yaw": self.env.base_ang_vel[
                    self.extra_args["robot_index"], 2
                ].item(),
            }
        )

    def retrieve_data(self):
        self.save_to_csv(
            "Base velocity x",
            [self.log["base_vel_x"], self.log["command_x"]],
            "base lin vel [m/s]",
            ["base_vel_x", "command_x"],
            ["measured", "target"],
        )
        self.save_to_csv(
            "Base velocity y",
            [self.log["base_vel_y"], self.log["command_y"]],
            "base lin vel [m/s]",
            ["base_vel_y", "command_y"],
            ["measured", "target"],
        )
        self.save_to_csv(
            "Base velocity yaw",
            [self.log["base_vel_yaw"], self.log["command_yaw"]],
            "base ang vel [rad/s]",
            ["base_vel_yaw", "command_yaw"],
            ["measured", "target"],
        )
