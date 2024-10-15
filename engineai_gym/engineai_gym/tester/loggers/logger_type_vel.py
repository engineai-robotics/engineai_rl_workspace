from engineai_gym.tester.loggers.logger_base import LoggerBase
import numpy as np


class LoggerTypeVel(LoggerBase):
    def log_data(self, extra_data):
        self.add_data_to_log(
            {
                "dof_vel": self.env.dof_vel[self.extra_args["robot_index"]]
                .cpu()
                .numpy(),
            }
        )

    def retrieve_data(self):
        dof_vel = np.stack(self.log["dof_vel"])
        self.save_to_csv(
            "DOF Vel",
            [dof_vel[:, idx] for idx in range(dof_vel.shape[1])],
            "Velocity [rad/s]",
            [f"dof_vel_{idx}" for idx in range(dof_vel.shape[1])],
            [f"dof_{idx}" for idx in range(dof_vel.shape[1])],
            "separate",
        )
