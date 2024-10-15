from engineai_gym.envs.base.obs.obs import Obs


class ObsPm01(Obs):
    def dof_vel_other0(self, lag=False):
        if lag:
            try:
                return self.env.lagged_obs_motor[:, self.env.num_dofs :]
            except:
                return None
        else:
            return self.env.dof_vel

    def base_ang_vel(self, lag=False):
        if lag:
            try:
                return self.env.lagged_obs_imu[:, :3]
            except:
                return None
        else:
            return self.env.base_ang_vel

    def projected_gravity(self, lag=False):
        if lag:
            try:
                return self.env.lagged_obs_imu[:, 3:]
            except:
                return None
        else:
            return self.env.projected_gravity

    def com_displacements(self):
        return self.env.domain_rands.domain_rands_type_rigid_body.com_displacements

    def rand_push_force(self):
        return self.env.domain_rands.domain_rands_type_disturbance.rand_push_force[
            :, :2
        ]

    def rand_push_torque(self):
        return self.env.domain_rands.domain_rands_type_disturbance.rand_push_torque

    def future_buf(self):
        return self.env.future_buf
