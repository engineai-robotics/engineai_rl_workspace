from engineai_gym.tester.loggers.logger_base import LoggerBase
import numpy as np


class LoggerTypeForce(LoggerBase):
    def log_data(self, extra_data):
        self.add_data_to_log(
            {
                "contact_forces_z": self.env.contact_forces[
                    self.extra_args["robot_index"], self.env.foot_indices, 2
                ]
                .cpu()
                .numpy()
            }
        )

    def retrieve_data(self):
        contact_forces_z = np.stack(self.log["contact_forces_z"])
        self.save_to_csv(
            "Foot Contact Forces z",
            [contact_forces_z[:, idx] for idx in range(contact_forces_z.shape[1])],
            "Forces z [N]",
            [f"contact_forces_z_{idx}" for idx in range(contact_forces_z.shape[1])],
            [f"foot_{idx}" for idx in range(contact_forces_z.shape[1])],
            "separate",
        )
