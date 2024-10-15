from engineai_gym.envs.base.domain_rands.domain_rands_base import DomainRandsBase
from isaacgym.torch_utils import *
from isaacgym import gymtorch


class DomainRandsTypeDisturbance(DomainRandsBase):
    def init_domain_rands(self):
        if self.env.cfg.domain_rands.disturbance.push_robots:
            self.push_interval = np.ceil(
                self.env.cfg.domain_rands.disturbance.push_interval_s / self.env.dt
            )

    def init_buffer_after_create_env(self):
        if self.env.cfg.domain_rands.disturbance.push_robots:
            self.rand_push_force = torch.zeros(
                (self.env.num_envs, 3), dtype=torch.float32, device=self.env.device
            )
            self.rand_push_torque = torch.zeros(
                (self.env.num_envs, 3), dtype=torch.float32, device=self.env.device
            )

    def init_rand_vec_on_step(self):
        if self.env.cfg.domain_rands.disturbance.push_robots:
            max_vel = self.env.cfg.domain_rands.disturbance.max_push_vel_xy
            self.rand_push_force[:, :2] = torch_rand_float(
                -max_vel, max_vel, (self.env.num_envs, 2), device=self.env.device
            )  # lin vel x/y
            max_push_angular = self.env.cfg.domain_rands.disturbance.max_push_ang_vel
            self.rand_push_torque = torch_rand_float(
                -max_push_angular,
                max_push_angular,
                (self.env.num_envs, 3),
                device=self.env.device,
            )

    def push_robots(self):
        """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
        self.env.root_states[:, 7:9] = self.rand_push_force[:, :2]

        self.env.root_states[:, 10:13] = self.rand_push_torque
        self.env.gym.set_actor_root_state_tensor(
            self.env.sim, gymtorch.unwrap_tensor(self.env.root_states)
        )

    def process_after_step(self):
        if self.env.cfg.domain_rands.disturbance.push_robots and (
            self.env.common_step_counter % self.push_interval == 0
        ):
            self.push_robots()
