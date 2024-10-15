from engineai_gym.tester.loggers.logger_base import LoggerBase


class LoggerTypePosition(LoggerBase):
    def log_data(self, extra_data):
        for idx in range(self.env.num_actions):
            self.add_data_to_log(
                {
                    f"dof_pos[{idx}]": self.env.dof_pos[
                        self.extra_args["robot_index"], idx
                    ].item(),
                    f"dof_pos_target[{idx}]": extra_data["actions"][
                        self.extra_args["robot_index"], idx
                    ].item()
                    * self.env.action_scales[idx].item(),
                }
            )

    def retrieve_data(self):
        for idx in range(self.env.num_actions):
            self.save_to_csv(
                f"DOF Position[{idx}]",
                [self.log[f"dof_pos[{idx}]"], self.log[f"dof_pos_target[{idx}]"]],
                "Position [rad]",
                [f"dof_pos[{idx}]", f"dof_pos_target[{idx}]"],
                ["measured", "target"],
                "combined",
            )
