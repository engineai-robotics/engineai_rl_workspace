from engineai_gym.envs.base.domain_rands.domain_rands_base import DomainRandsBase
from isaacgym.torch_utils import *
from engineai_rl_lib.math import get_euler_xyz_tensor


class DomainRandsTypeObsLag(DomainRandsBase):
    def init_domain_rands(self):
        self.motor_lag = (
            getattr(self.env.cfg.domain_rands, "motor_lag_timesteps", 0) > 0
            or self.env.cfg.domain_rands.obs_lag.randomize_motor_lag_timesteps
        )
        self.imu_lag = (
            getattr(self.env.cfg.domain_rands, "imu_lag_timesteps", 0) > 0
            or self.env.cfg.domain_rands.obs_lag.randomize_imu_lag_timesteps
        )

    def init_buffer_after_create_env(self):
        if self.motor_lag:
            self.motor_lag_timestep = torch.zeros(
                self.env.num_envs, device=self.env.device, dtype=int
            )
            if self.env.cfg.domain_rands.obs_lag.randomize_motor_lag_timesteps:
                self.dof_pos_lag_buffer = torch.zeros(
                    self.env.num_envs,
                    self.env.num_dofs,
                    self.env.cfg.domain_rands.obs_lag.motor_lag_timesteps_range[1] + 1,
                    device=self.env.device,
                )
                self.dof_vel_lag_buffer = torch.zeros(
                    self.env.num_envs,
                    self.env.num_dofs,
                    self.env.cfg.domain_rands.obs_lag.motor_lag_timesteps_range[1] + 1,
                    device=self.env.device,
                )
                if (
                    self.env.cfg.domain_rands.obs_lag.randomize_motor_lag_timesteps_perstep
                ):
                    self.last_motor_lag_timestep = torch.zeros(
                        self.env.num_envs, device=self.env.device, dtype=int
                    )
            else:
                self.dof_pos_lag_buffer = torch.zeros(
                    self.env.num_envs,
                    self.env.num_dofs,
                    self.env.cfg.domain_rands.obs_lag.motor_lag_timesteps + 1,
                    device=self.env.device,
                )
                self.dof_vel_lag_buffer = torch.zeros(
                    self.env.num_envs,
                    self.env.num_dofs,
                    self.env.cfg.domain_rands.obs_lag.motor_lag_timesteps + 1,
                    device=self.env.device,
                )
                self.motor_lag_timestep[
                    :
                ] = self.env.cfg.domain_rands.obs_lag.motor_lag_timesteps
        if self.imu_lag:
            self.imu_lag_timestep = torch.zeros(
                self.env.num_envs, device=self.env.device, dtype=int
            )
            if self.env.cfg.domain_rands.obs_lag.randomize_imu_lag_timesteps:
                self.base_ang_vel_lag_buffer = torch.zeros(
                    self.env.num_envs,
                    3,
                    self.env.cfg.domain_rands.obs_lag.imu_lag_timesteps_range[1] + 1,
                    device=self.env.device,
                )
                self.base_euler_xyz_lag_buffer = torch.zeros(
                    self.env.num_envs,
                    3,
                    self.env.cfg.domain_rands.obs_lag.imu_lag_timesteps_range[1] + 1,
                    device=self.env.device,
                )
                if (
                    self.env.cfg.domain_rands.obs_lag.randomize_imu_lag_timesteps_perstep
                ):
                    self.last_imu_lag_timestep = torch.zeros(
                        self.env.num_envs, device=self.env.device, dtype=int
                    )
            else:
                self.base_ang_vel_lag_buffer = torch.zeros(
                    self.env.num_envs,
                    3,
                    self.env.cfg.domain_rands.obs_lag.imu_lag_timesteps + 1,
                    device=self.env.device,
                )
                self.base_euler_xyz_lag_buffer = torch.zeros(
                    self.env.num_envs,
                    3,
                    self.env.cfg.domain_rands.obs_lag.imu_lag_timesteps + 1,
                    device=self.env.device,
                )
                self.imu_lag_timestep[
                    :
                ] = self.env.cfg.domain_rands.obs_lag.imu_lag_timesteps

    def init_rand_vec_on_reset_idx(self, env_ids):
        if self.motor_lag:
            if self.env.cfg.domain_rands.obs_lag.randomize_motor_lag_timesteps:
                self.motor_lag_timestep[env_ids] = torch.randint(
                    self.env.cfg.domain_rands.obs_lag.motor_lag_timesteps_range[0],
                    self.env.cfg.domain_rands.obs_lag.motor_lag_timesteps_range[1] + 1,
                    (len(env_ids),),
                    device=self.env.device,
                )

        if self.imu_lag:
            if self.env.cfg.domain_rands.obs_lag.randomize_imu_lag_timesteps:
                self.imu_lag_timestep[env_ids] = torch.randint(
                    self.env.cfg.domain_rands.obs_lag.imu_lag_timesteps_range[0],
                    self.env.cfg.domain_rands.obs_lag.imu_lag_timesteps_range[1] + 1,
                    (len(env_ids),),
                    device=self.env.device,
                )

    def init_rand_vec_on_step(self):
        if self.motor_lag:
            if self.env.cfg.domain_rands.obs_lag.randomize_motor_lag_timesteps:
                if (
                    self.env.cfg.domain_rands.obs_lag.randomize_motor_lag_timesteps_perstep
                ):
                    self.motor_lag_timestep = torch.randint(
                        self.env.cfg.domain_rands.obs_lag.motor_lag_timesteps_range[0],
                        self.env.cfg.domain_rands.obs_lag.motor_lag_timesteps_range[1]
                        + 1,
                        (self.env.num_envs,),
                        device=self.env.device,
                    )
        if self.imu_lag:
            if self.env.cfg.domain_rands.obs_lag.randomize_imu_lag_timesteps:
                if (
                    self.env.cfg.domain_rands.obs_lag.randomize_imu_lag_timesteps_perstep
                ):
                    self.imu_lag_timestep = torch.randint(
                        self.env.cfg.domain_rands.obs_lag.imu_lag_timesteps_range[0],
                        self.env.cfg.domain_rands.obs_lag.imu_lag_timesteps_range[1]
                        + 1,
                        (self.env.num_envs,),
                        device=self.env.device,
                    )

    def process_on_reset_idx(self, env_ids):
        if self.motor_lag:
            self.dof_pos_lag_buffer[env_ids, :, :] = 0.0
            self.dof_vel_lag_buffer[env_ids, :, :] = 0.0
            if self.env.cfg.domain_rands.obs_lag.randomize_imu_lag_timesteps:
                if (
                    self.env.cfg.domain_rands.obs_lag.randomize_motor_lag_timesteps_perstep
                ):
                    self.last_motor_lag_timestep[
                        env_ids
                    ] = self.env.cfg.domain_rands.obs_lag.motor_lag_timesteps_range[1]
        if self.imu_lag:
            self.base_ang_vel_lag_buffer[env_ids, :, :] = 0.0
            self.base_euler_xyz_lag_buffer[env_ids, :, :] = 0.0
            if self.env.cfg.domain_rands.obs_lag.randomize_imu_lag_timesteps:
                if (
                    self.env.cfg.domain_rands.obs_lag.randomize_imu_lag_timesteps_perstep
                ):
                    self.last_imu_lag_timestep[
                        env_ids
                    ] = self.env.cfg.domain_rands.obs_lag.imu_lag_timesteps_range[1]

    def process_after_decimation(self):
        if self.motor_lag:
            if self.env.cfg.domain_rands.obs_lag.randomize_motor_lag_timesteps:
                self.dof_pos_lag_buffer[:, :, 1:] = self.dof_pos_lag_buffer[
                    :,
                    :,
                    : self.env.cfg.domain_rands.obs_lag.motor_lag_timesteps_range[1],
                ]
                self.dof_pos_lag_buffer[:, :, 0] = self.env.dof_pos
                self.dof_vel_lag_buffer[:, :, 1:] = self.dof_vel_lag_buffer[
                    :,
                    :,
                    : self.env.cfg.domain_rands.obs_lag.motor_lag_timesteps_range[1],
                ]
                self.dof_vel_lag_buffer[:, :, 0] = self.env.dof_vel
                if (
                    self.env.cfg.domain_rands.obs_lag.randomize_motor_lag_timesteps_perstep
                ):
                    cond = self.motor_lag_timestep > self.last_motor_lag_timestep + 1
                    self.motor_lag_timestep[cond] = (
                        self.last_motor_lag_timestep[cond] + 1
                    )
                    self.last_motor_lag_timestep = self.motor_lag_timestep.clone()
            else:
                self.dof_pos_lag_buffer[:, :, 1:] = self.dof_pos_lag_buffer[
                    :, :, : self.env.cfg.domain_rands.obs_lag.motor_lag_timesteps
                ]
                self.dof_pos_lag_buffer[:, :, 0] = self.env.dof_pos
                self.dof_vel_lag_buffer[:, :, 1:] = self.dof_vel_lag_buffer[
                    :, :, : self.env.cfg.domain_rands.obs_lag.motor_lag_timesteps
                ]
                self.dof_vel_lag_buffer[:, :, 0] = self.env.dof_vel

            self.lagged_dof_pos = self.dof_pos_lag_buffer[
                torch.arange(self.env.num_envs), :, self.motor_lag_timestep.long()
            ]
            self.lagged_dof_vel = self.dof_vel_lag_buffer[
                torch.arange(self.env.num_envs), :, self.motor_lag_timestep.long()
            ]
        if self.imu_lag:
            self.env.gym.refresh_actor_root_state_tensor(self.env.sim)
            base_quat = self.env.root_states[:, 3:7]
            base_ang_vel = quat_rotate_inverse(
                base_quat, self.env.root_states[:, 10:13]
            )
            base_euler_xyz = get_euler_xyz_tensor(base_quat)
            if self.env.cfg.domain_rands.obs_lag.randomize_imu_lag_timesteps:
                self.base_ang_vel_lag_buffer[:, :, 1:] = self.base_ang_vel_lag_buffer[
                    :, :, : self.env.cfg.domain_rands.obs_lag.imu_lag_timesteps_range[1]
                ]
                self.base_ang_vel_lag_buffer[:, :, 0] = base_ang_vel
                self.base_euler_xyz_lag_buffer[
                    :, :, 1:
                ] = self.base_euler_xyz_lag_buffer[
                    :, :, : self.env.cfg.domain_rands.obs_lag.imu_lag_timesteps_range[1]
                ]
                self.base_euler_xyz_lag_buffer[:, :, 0] = base_euler_xyz
                if (
                    self.env.cfg.domain_rands.obs_lag.randomize_imu_lag_timesteps_perstep
                ):
                    cond = self.imu_lag_timestep > self.last_imu_lag_timestep + 1
                    self.imu_lag_timestep[cond] = self.last_imu_lag_timestep[cond] + 1
                    self.last_imu_lag_timestep = self.imu_lag_timestep.clone()
            else:
                self.base_ang_vel_lag_buffer[:, :, 1:] = self.base_ang_vel_lag_buffer[
                    :, :, : self.env.cfg.domain_rands.obs_lag.imu_lag_timesteps
                ]
                self.base_ang_vel_lag_buffer[:, :, 0] = base_ang_vel
                self.base_euler_xyz_lag_buffer[
                    :, :, 1:
                ] = self.base_euler_xyz_lag_buffer[
                    :, :, : self.env.cfg.domain_rands.obs_lag.imu_lag_timesteps
                ]
                self.base_euler_xyz_lag_buffer[:, :, 0] = base_euler_xyz
            self.lagged_base_ang_vel = self.base_ang_vel_lag_buffer[
                torch.arange(self.env.num_envs), :, self.imu_lag_timestep.long()
            ]
            self.lagged_base_euler_xyz = self.base_euler_xyz_lag_buffer[
                torch.arange(self.env.num_envs), :, self.imu_lag_timestep.long()
            ]
