from typing import Tuple
import torch


class VecGymWrapper:
    def __init__(self, env):
        self._env = env
        self._extras = self._env.extras
        self._extras["observations"] = {}

    def get_env_info(self) -> Tuple[dict, torch.Tensor, dict]:
        return self._env.obs_dict, self._env.goal_dict, self._extras

    def reset(self) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        self._env.reset()
        return self._env.obs_dict, self._env.goal_dict, self._extras

    def step(
        self, actions
    ) -> Tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        self._env.step(actions)
        return (
            self._env.obs_dict,
            self._env.goal_dict,
            self._env.rew_buf,
            self._env.reset_buf,
            self._env.extras,
        )

    def set_camera(self, position, lookat):
        self._env.set_camera(position=position, lookat=lookat)

    @property
    def cfg(self):
        return self._env.cfg

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self._env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self._env.episode_length_buf = value

    @property
    def reset_buf(self) -> torch.Tensor:
        """The episode length buffer."""
        return self._env.reset_buf

    @property
    def extras(self):
        return self._extras

    @property
    def envs(self):
        return self._env.envs

    @property
    def sim(self):
        return self._env.sim

    @property
    def gym(self):
        return self._env.gym

    @property
    def viewer(self):
        return self._env.viewer

    @property
    def num_envs(self):
        return self._env.num_envs

    @property
    def device(self):
        return self._env.device

    @property
    def max_episode_length(self):
        return self._env.max_episode_length

    @property
    def num_actions(self):
        return self._env.num_actions

    @property
    def foot_indices(self):
        return self._env.foot_indices

    @property
    def obs_scales(self):
        return self._env.obs_scales

    @property
    def dt(self):
        return self._env.dt

    @property
    def dof_pos(self):
        return self._env.dof_pos

    @property
    def dof_vel(self):
        return self._env.dof_vel

    @property
    def torques(self):
        return self._env.torques

    @property
    def commands(self):
        return self._env.commands

    @commands.setter
    def commands(self, value):
        self._env.commands = value
        self._env.goal_dict = self._env.compute_goals()

    @property
    def num_commands(self):
        return self.commands.shape[-1]

    @property
    def goal_dict(self):
        return self._env.goal_dict

    @property
    def domain_rands(self):
        return self._env.domain_rands

    @property
    def base_lin_vel(self):
        return self._env.base_lin_vel

    @property
    def base_ang_vel(self):
        return self._env.base_lin_vel

    @property
    def contact_forces(self):
        return self._env.contact_forces

    @property
    def dof_pos_limits(self):
        return self._env.dof_pos_limits

    @property
    def action_scales(self):
        return self._env.action_scales

    @property
    def dof_names(self):
        return self._env.dof_names

    @property
    def ref_state_loader(self):
        if hasattr(self._env, "ref_state_loader"):
            return self._env.ref_state_loader
        else:
            return None
