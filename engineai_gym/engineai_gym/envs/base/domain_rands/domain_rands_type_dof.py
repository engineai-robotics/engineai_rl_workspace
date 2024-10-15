from engineai_gym.envs.base.domain_rands.domain_rands_base import DomainRandsBase
from isaacgym.torch_utils import *
from copy import deepcopy


class DomainRandsTypeDof(DomainRandsBase):
    def init_buffer_after_create_env(self):
        if self.env.cfg.domain_rands.dof.randomize_motor_offset:
            self.motor_offsets = torch.zeros(
                self.env.num_envs,
                self.env.num_dofs,
                dtype=torch.float,
                device=self.env.device,
                requires_grad=False,
            )
        self.original_dof_props = [
            deepcopy(self.env.gym.get_actor_dof_properties(env, 0))
            for env in self.env.envs
        ]
        if self.env.cfg.domain_rands.dof.randomize_joint_friction:
            if self.env.cfg.domain_rands.dof.randomize_joint_friction_each_joint:
                self.joint_friction_coeffs = torch.ones(
                    self.env.num_envs,
                    self.env.num_dofs,
                    dtype=torch.float,
                    device=self.env.device,
                    requires_grad=False,
                )
            else:
                self.joint_friction_coeffs = torch.ones(
                    self.env.num_envs,
                    1,
                    dtype=torch.float,
                    device=self.env.device,
                    requires_grad=False,
                )
        if self.env.cfg.domain_rands.dof.randomize_joint_armature:
            if self.env.cfg.domain_rands.dof.randomize_joint_armature_each_joint:
                self.joint_armature_multi = torch.zeros(
                    self.env.num_envs,
                    self.env.num_dofs,
                    dtype=torch.float,
                    device=self.env.device,
                    requires_grad=False,
                )
            else:
                self.joint_armature_multi = torch.zeros(
                    self.env.num_envs,
                    1,
                    dtype=torch.float,
                    device=self.env.device,
                    requires_grad=False,
                )
        if self.env.cfg.domain_rands.dof.randomize_gains:
            self.original_p_gains = self.env.p_gains.clone()
            self.original_d_gains = self.env.d_gains.clone()
            self.p_gains_multi = torch.zeros_like(self.original_p_gains)
            self.d_gains_multi = torch.zeros_like(self.original_d_gains)

    def init_rand_vec_on_decimation(self):
        if self.env.cfg.domain_rands.dof.randomize_torque:
            motor_strength_ranges = self.env.cfg.domain_rands.dof.torque_multip_range
            self.torque_multi = torch_rand_float(
                motor_strength_ranges[0],
                motor_strength_ranges[1],
                (self.env.num_envs, self.env.num_dofs),
                device=self.env.device,
            )
        if self.env.cfg.domain_rands.dof.randomize_coulomb_friction:
            self.joint_viscous = torch_rand_float(
                self.env.cfg.domain_rands.dof.joint_viscous_range[0],
                self.env.cfg.domain_rands.dof.joint_viscous_range[1],
                (self.env.num_envs, self.env.num_dofs),
                device=self.env.device,
            )
            self.joint_coulomb = torch_rand_float(
                self.env.cfg.domain_rands.dof.joint_coulomb_range[0],
                self.env.cfg.domain_rands.dof.joint_coulomb_range[1],
                (self.env.num_envs, self.env.num_dofs),
                device=self.env.device,
            )
        else:
            self.joint_viscous = self.joint_coulomb = 0

    def process_on_decimation(self):
        if self.env.cfg.domain_rands.dof.randomize_coulomb_friction:
            self.coulomb_friction = (
                self.joint_viscous * self.env.dof_vel + self.joint_coulomb
            )

    def init_rand_vec_on_reset_idx(self, env_ids):
        if self.env.cfg.domain_rands.dof.randomize_motor_offset:
            min_offset, max_offset = self.env.cfg.domain_rands.dof.motor_offset_range
            self.motor_offsets[env_ids, :] = torch_rand_float(
                min_offset,
                max_offset,
                (len(env_ids), self.env.num_dofs),
                device=self.env.device,
            )

        if self.env.cfg.domain_rands.dof.randomize_joint_friction:
            if self.env.cfg.domain_rands.dof.randomize_joint_friction_each_joint:
                for i, name in enumerate(self.env.dof_names):
                    for (
                        dof_type,
                        joint_friction_multiplier_range,
                    ) in (
                        self.env.cfg.domain_rands.dof.joint_friction_multi_range_each_joint.items()
                    ):
                        if dof_type in name:
                            self.joint_friction_coeffs[env_ids, i] = torch_rand_float(
                                joint_friction_multiplier_range[0],
                                joint_friction_multiplier_range[1],
                                (len(env_ids), 1),
                                device=self.env.device,
                            ).reshape(-1)
                            break
            else:
                joint_friction_multiplier_range = (
                    self.env.cfg.domain_rands.dof.joint_friction_multi_range
                )
                self.joint_friction_coeffs[env_ids] = torch_rand_float(
                    joint_friction_multiplier_range[0],
                    joint_friction_multiplier_range[1],
                    (len(env_ids), 1),
                    device=self.env.device,
                )
        if self.env.cfg.domain_rands.dof.randomize_joint_armature:
            if self.env.cfg.domain_rands.dof.randomize_joint_armature_each_joint:
                for i, name in enumerate(self.env.dof_names):
                    for (
                        dof_type,
                        joint_armature_multiplier_range,
                    ) in (
                        self.env.cfg.domain_rands.dof.joint_armature_multi_range_each_joint.items()
                    ):
                        if dof_type in name:
                            self.joint_armature_multi[env_ids, i] = torch_rand_float(
                                max(0, joint_armature_multiplier_range[0]),
                                joint_armature_multiplier_range[1],
                                (len(env_ids), 1),
                                device=self.env.device,
                            ).reshape(-1)
                            break
            else:
                joint_armature_range = (
                    self.env.cfg.domain_rands.dof.joint_armature_multi_range
                )
                self.joint_armature_multi[env_ids] = torch_rand_float(
                    max(0, joint_armature_range[0]),
                    joint_armature_range[1],
                    (len(env_ids), 1),
                    device=self.env.device,
                )
        if self.env.cfg.domain_rands.dof.randomize_gains:
            p_gains_range = self.env.cfg.domain_rands.dof.stiffness_multi_range
            d_gains_range = self.env.cfg.domain_rands.dof.damping_multi_range

            self.p_gains_multi[env_ids] = torch_rand_float(
                p_gains_range[0],
                p_gains_range[1],
                (len(env_ids), self.env.num_dofs),
                device=self.env.device,
            )
            self.d_gains_multi[env_ids] = torch_rand_float(
                d_gains_range[0],
                d_gains_range[1],
                (len(env_ids), self.env.num_dofs),
                device=self.env.device,
            )

    def process_on_reset_idx(self, env_ids):
        for env_id in env_ids:
            dof_props = self.env.gym.get_actor_dof_properties(self.env.envs[env_id], 0)

            for i in range(self.env.num_dofs):
                if self.env.cfg.domain_rands.dof.randomize_joint_friction:
                    if (
                        self.env.cfg.domain_rands.dof.randomize_joint_friction_each_joint
                    ):
                        dof_props["friction"][i] = (
                            self.original_dof_props[env_id]["friction"][i]
                            * self.joint_friction_coeffs[env_id, i]
                        )
                    else:
                        dof_props["friction"][i] = (
                            self.original_dof_props[env_id]["friction"][i]
                            * self.joint_friction_coeffs[env_id, 0]
                        )
                if isinstance(self.env.cfg.asset.min_joint_armature, float):
                    self.original_dof_props[env_id]["armature"][i] = max(
                        self.env.cfg.asset.min_joint_armature,
                        self.original_dof_props[env_id]["armature"][i],
                    )
                elif isinstance(self.env.cfg.asset.min_joint_armature, dict):
                    if self.env.dof_names[i] in self.env.cfg.asset.min_joint_armature:
                        self.original_dof_props[env_id]["armature"][i] = max(
                            self.env.cfg.asset.min_joint_armature[
                                self.env.dof_names[i]
                            ],
                            self.original_dof_props[env_id]["armature"][i],
                        )
                if self.env.cfg.domain_rands.dof.randomize_joint_armature:
                    if (
                        self.env.cfg.domain_rands.dof.randomize_joint_armature_each_joint
                    ):
                        dof_props["armature"][i] = (
                            self.original_dof_props[env_id]["armature"][i]
                            * self.joint_armature_multi[env_id, i]
                        )
                    else:
                        dof_props["armature"][i] = (
                            self.original_dof_props[env_id]["armature"][i]
                            * self.joint_armature_multi[env_id, 0]
                        )
            self.env.gym.set_actor_dof_properties(self.env.envs[env_id], 0, dof_props)

        if self.env.cfg.domain_rands.dof.randomize_gains:
            self.env.p_gains[env_ids] = (
                self.p_gains_multi[env_ids] * self.original_p_gains[env_ids]
            )
            self.env.d_gains[env_ids] = (
                self.d_gains_multi[env_ids] * self.original_d_gains[env_ids]
            )
