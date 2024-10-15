from engineai_gym.envs.base.config_legged_robot import ConfigLeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch

import torch
from engineai_gym.envs import LeggedRobot
from engineai_gym.utils.terrain import HumanoidTerrain


class BipedRobot(LeggedRobot):
    def __init__(
        self,
        obs_class,
        domain_rand_class,
        goal_class,
        reward_class,
        cfg: ConfigLeggedRobot,
        sim_params,
        physics_engine,
        sim_device,
        headless,
    ):
        super().__init__(
            obs_class,
            goal_class,
            domain_rand_class,
            reward_class,
            cfg,
            sim_params,
            physics_engine,
            sim_device,
            headless,
        )
        self.compute_ref_state()

    def create_sim(self):
        """Creates simulation, terrain and environments"""
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ["heightfield", "trimesh"]:
            self.terrain = HumanoidTerrain(self.cfg.terrain, self.num_envs)
        if mesh_type == "plane":
            self._create_ground_plane()
        elif mesh_type == "heightfield":
            self._create_heightfield()
        elif mesh_type == "trimesh":
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]"
            )
        self._create_envs()

    def get_gait_phase(self):
        # return float mask 1 is stance, 0 is swing
        phase = self.get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        # Add double support phase
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        stance_curve = torch.zeros((self.num_envs, 2), device=self.device)
        swing_curve = torch.zeros((self.num_envs, 2), device=self.device)
        # left foot
        stance_mask[:, 0] = sin_pos >= 0
        stance_curve[:, 0] = sin_pos
        stance_curve[:, 0][sin_pos < 0] = 0
        swing_curve[:, 0] = -sin_pos
        swing_curve[:, 0][sin_pos < 0] = 0
        # right foot
        stance_mask[:, 1] = sin_pos < 0
        stance_curve[:, 1] = -sin_pos
        stance_curve[:, 1][sin_pos < 0] = 0
        swing_curve[:, 1] = sin_pos
        swing_curve[:, 1][sin_pos < 0] = 0

        swing_mask = 1 - stance_mask
        return stance_mask, swing_mask, stance_curve, swing_curve

    def compute_ref_state(self):
        phase = self.get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        scale_1 = self.cfg.rewards.params.target_joint_pos_scale
        scale_2 = 2 * scale_1
        # left swing
        sin_pos_l[sin_pos_l > 0] = 0
        self.ref_dof_pos[:, 2] = sin_pos_l * scale_1
        self.ref_dof_pos[:, 3] = -sin_pos_l * scale_2
        self.ref_dof_pos[:, 4] = sin_pos_l * scale_1
        # print(phase[0], sin_pos_l[0])
        # right
        sin_pos_r[sin_pos_r < 0] = 0
        self.ref_dof_pos[:, 8] = -sin_pos_r * scale_1
        self.ref_dof_pos[:, 9] = sin_pos_r * scale_2
        self.ref_dof_pos[:, 10] = -sin_pos_r * scale_1

        self.ref_dof_pos[torch.abs(sin_pos) < 0.05] = 0.0

        self.ref_action = 2 * self.ref_dof_pos

        self.ref_dof_pos += self.default_dof_pos

    def step(self, actions):
        if self.cfg.env.use_ref_actions:
            actions += self.ref_action
        return super().step(actions)

    def _get_contact_info(self):
        self.contact = (
            self.contact_forces[:, self.foot_indices, 2]
            > self.cfg.contact.foot_contact_threshold
        )
        stance_mask, _, _, _ = self.get_gait_phase()
        self.contact_filt = torch.logical_or(
            torch.logical_or(self.contact, stance_mask), self.last_contacts
        )
        self.first_contact = (self.foot_air_time > 0.0) * self.contact_filt

    def _pre_compute_observations(self):
        self.compute_ref_state()
