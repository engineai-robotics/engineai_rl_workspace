from engineai_gym.tester.loggers.logger_base import LoggerBase
import numpy as np


class LoggerTypeTorque(LoggerBase):
    def log_data(self, extra_data):
        self.add_data_to_log(
            {
                "dof_torque": self.env.torques[self.extra_args["robot_index"]]
                .cpu()
                .numpy(),
            }
        )

    def retrieve_data(self):
        dof_torque = np.stack(self.log["dof_torque"])
        self.save_to_csv(
            "DOF Torque",
            [dof_torque[:, idx] for idx in range(dof_torque.shape[1])],
            "Joint Torque [Nm]",
            [f"dof_torque_{idx}" for idx in range(dof_torque.shape[1])],
            [f"dof_{idx}" for idx in range(dof_torque.shape[1])],
            "separate",
        )
