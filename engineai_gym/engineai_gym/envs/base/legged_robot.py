import os
from copy import deepcopy
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Dict

from engineai_gym import ENGINEAI_GYM_PACKAGE_DIR
from engineai_gym.envs.base.env_base import EnvBase
from engineai_gym.utils.terrain import Terrain
from engineai_rl_lib.math import quat_apply_yaw, wrap_to_pi, get_euler_xyz_tensor
from engineai_rl_lib.class_operations import class_to_dict
from .config_legged_robot import ConfigLeggedRobot


class LeggedRobot(EnvBase):
    def __init__(
        self,
        obs_class,
        goal_class,
        domain_rand_class,
        reward_class,
        cfg: ConfigLeggedRobot,
        sim_params,
        physics_engine,
        sim_device,
        headless,
    ):
        """Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initializes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg()
        super().__init__(
            obs_class,
            goal_class,
            domain_rand_class,
            reward_class,
            self.cfg,
            sim_params,
            physics_engine,
            sim_device,
            headless,
        )
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()

    def step(self, actions):
        """Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        self.domain_rands.init_rand_vec_on_step()
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.domain_rands.init_rand_vec_on_decimation()
            self.domain_rands.process_on_decimation()
            self.torques = self._compute_torques().view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.domain_rands.process_after_decimation()
        self.post_physics_step()

        return self.obs_dict, self.goal_dict, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """check terminations, compute observations and rewards
        calls self._post_physics_step_callback() for common computations
        calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 7:10]
        )
        self.base_ang_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )
        self.projected_gravity[:] = quat_rotate_inverse(
            self.base_quat, self.gravity_vec
        )
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        self._post_compute_reward()

        self._pre_compute_observations()
        obs_dict_before_reset = self.compute_observations()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        self._pre_compute_observations()
        obs_dict_after_reset = (
            self.compute_observations()
        )  # In some cases a simulation step might be required to refresh some obs (for example body positions)
        self.obs_dict = {}
        for obs_name in obs_dict_after_reset:
            self.obs_dict[obs_name] = {
                "before_reset": obs_dict_before_reset[obs_name],
                "after_reset": obs_dict_after_reset[obs_name],
            }
        self.goal_dict = self.compute_goals()
        self.last_last_actions[:] = torch.clone(self.last_actions[:])
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def check_termination(self):
        """Check if environments need to be reset"""
        self.reset_buf = torch.any(
            torch.norm(
                self.contact_forces[:, self.termination_contact_indices, :], dim=-1
            )
            > 1.0,
            dim=1,
        )
        self.time_out_buf = (
            self.episode_length_buf > self.max_episode_length
        )  # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (
            self.common_step_counter % self.max_episode_length == 0
        ):
            self.update_command_curriculum(env_ids)

        self.reset_dofs_and_root_states(env_ids)

        self._resample_commands(env_ids)

        self.domain_rands.init_rand_vec_on_reset_idx(env_ids)
        self.domain_rands.process_on_reset_idx(env_ids)

        # reset buffers
        self.last_last_actions[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.foot_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        self.extras["episode"]["rewards"] = {}
        self.extras["episode"]["rewards"]["raw"] = {}
        self.extras["episode"]["rewards"]["scaled"] = {}
        for key in self.episode_sums_raw.keys():
            self.extras["episode"]["rewards"]["raw"][key] = (
                torch.mean(self.episode_sums_raw[key][env_ids])
                / self.max_episode_length
            )
            self.episode_sums_raw[key][env_ids] = 0.0
            self.extras["episode"]["rewards"]["scaled"][key] = (
                torch.mean(self.episode_sums_scaled[key][env_ids])
                / self.max_episode_length
            )
            self.episode_sums_scaled[key][env_ids] = 0.0

        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(
                self.terrain_levels.float()
            )
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][
                1
            ]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.projected_gravity[env_ids] = quat_rotate_inverse(
            self.base_quat[env_ids], self.gravity_vec[env_ids]
        )
        self.base_lin_vel[env_ids] = quat_rotate_inverse(
            self.base_quat[env_ids], self.root_states[env_ids, 7:10]
        )
        self.base_ang_vel[env_ids] = quat_rotate_inverse(
            self.base_quat[env_ids], self.root_states[env_ids, 10:13]
        )

    def reset_dofs_and_root_states(self, env_ids):
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

    def compute_reward(self):
        """Compute rewards
        Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
        adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.0
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            if name == "termination":
                continue
            raw_rew = self.reward_functions[i]()
            scaled_rew = raw_rew * self.reward_scales[name]
            self.rew_buf += scaled_rew
            self.episode_sums_raw[name] += raw_rew
            self.episode_sums_scaled[name] += scaled_rew
        if self.cfg.rewards.params.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            raw_rew = self.rewards.reward_termination()
            scaled_rew = raw_rew * self.reward_scales["termination"]
            self.rew_buf += scaled_rew
            self.episode_sums_raw["termination"] += raw_rew
            self.episode_sums_scaled["termination"] += scaled_rew

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
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
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

    def set_camera(self, position, lookat):
        """Set camera position and direction"""
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    # ------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        self.domain_rands.domain_rands_type_rigid_shape.process_on_create_env(
            props, env_id
        )
        return props

    def _process_dof_props(self, props, env_id):
        """Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(
                self.num_dofs,
                2,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            self.dof_vel_limits = torch.zeros(
                self.num_dofs,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            self.torque_limits = torch.zeros(
                self.num_dofs,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                if self.cfg.safety.torque_hard_limit_multi is not None:
                    for dof_type in self.cfg.safety.torque_hard_limit_multi:
                        if dof_type in self.dof_names[i]:
                            self.torque_limits[
                                i
                            ] *= self.cfg.safety.torque_hard_limit_multi[dof_type]
        for i in range(len(props)):
            for dof_type in self.cfg.asset.joint_armature:
                if dof_type in self.dof_names[i]:
                    props["armature"][i] = self.cfg.asset.joint_armature[dof_type]
            for dof_type in self.cfg.asset.joint_friction:
                if dof_type in self.dof_names[i]:
                    props["friction"][i] = self.cfg.asset.joint_friction[dof_type]
        self.domain_rands.domain_rands_type_dof.process_on_create_env(props, env_id)
        return props

    def _process_rigid_body_props(self, props, env_id):
        self.domain_rands.domain_rands_type_rigid_body.process_on_create_env(
            props, env_id
        )
        return props

    def _post_physics_step_callback(self):
        """Callback called before computing terminations, rewards, and observations
        Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        self.domain_rands.process_after_step()
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()

        self._get_contact_info()
        self.foot_air_time += self.dt
        foot_z = (
            self.rigid_body_state[:, self.foot_indices, 2]
            - self.cfg.asset.foot_half_z_size
        )
        delta_z = foot_z - self.last_foot_z
        self.feet_heights += delta_z
        self.last_foot_z = foot_z

    def _get_contact_info(self):
        self.contact = (
            self.contact_forces[:, self.foot_indices, 2]
            > self.cfg.contact.foot_contact_threshold
        )
        self.contact_filt = torch.logical_or(self.contact, self.last_contacts)
        self.first_contact = (self.foot_air_time > 0.0) * self.contact_filt

    def _post_compute_reward(self):
        self.last_contacts = self.contact
        self.feet_heights *= ~self.contact
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        self.foot_air_time *= ~self.contact_filt
        env_ids = (
            (
                self.episode_length_buf
                % int(self.cfg.commands.resampling_time / self.dt)
                == 0
            )
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(env_ids)
        if self.cfg.commands.yaw_from_heading_target:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(
                0.5 * wrap_to_pi(self.heading_target - heading), -1.0, 1.0
            )

    def _resample_commands(self, env_ids):
        """Randomly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0],
            self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0],
            self.command_ranges["lin_vel_y"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        if self.cfg.commands.yaw_from_heading_target:
            self.heading_target[env_ids] = torch_rand_float(
                self.command_ranges["heading"][0],
                self.command_ranges["heading"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(
                self.command_ranges["ang_vel_yaw"][0],
                self.command_ranges["ang_vel_yaw"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (
            torch.norm(self.commands[env_ids, :2], dim=1)
            > self.cfg.commands.set_zero_threshold
        ).unsqueeze(1)
        self.stand_still_idx[env_ids] = (
            torch_rand_float(0, 1, (len(env_ids), 1), device=self.device).squeeze(1)
            < self.cfg.commands.still_ratio
        )
        self.commands[env_ids[self.stand_still_idx[env_ids]]] = 0.0

    def _compute_torques(self):
        """Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        control_type = self.cfg.control.control_type
        try:
            actions = self.domain_rands.domain_rands_type_action_lag.lagged_actions
        except:
            actions = self.actions

        actions_scaled = actions * self.action_scales
        self.controller_input[:, self.action_joint_indices] = actions_scaled
        try:
            motor_offsets = self.domain_rands.domain_rands_type_dof.motor_offsets
        except:
            motor_offsets = 0
        try:
            coulomb_friction = self.domain_rands.domain_rands_type_dof.coulomb_friction
        except:
            coulomb_friction = 0
        if control_type == "P":
            torques = (
                self.p_gains
                * (
                    self.controller_input
                    + self.default_dof_pos
                    - self.dof_pos
                    + motor_offsets
                )
                - self.d_gains * self.dof_vel
                - coulomb_friction
            )
        elif control_type == "V":
            torques = (
                self.p_gains * (self.controller_input - self.dof_vel)
                - self.d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
                - coulomb_friction
            )
        elif control_type == "T":
            torques = deepcopy(self.controller_input)
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        try:
            torque_multi = self.domain_rands.domain_rands_type_dof.torque_multi
        except:
            torque_multi = 1
        torques *= torque_multi
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """Resets DOF position and velocities of selected environments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environment ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos + torch_rand_float(
            -0.1, 0.1, (len(env_ids), self.num_dofs), device=self.device
        )
        self.dof_vel[env_ids] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reset_root_states(
        self,
        env_ids,
    ):
        """Resets ROOT states position and velocities of selected environments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environment ids
        """
        self.root_states[env_ids] = self.base_init_state
        # base position
        if self.custom_origins:
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(
                -4.0, 4.0, (len(env_ids), 2), device=self.device
            )  # xy position within 4m of the center
        else:
            self.root_states[env_ids, :3] += self.env_origins[env_ids]

        if self.cfg.asset.fix_base_link:
            self.root_states[env_ids, 7:13] = 0
            self.root_states[env_ids, 2] += 1.8

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _update_terrain_curriculum(self, env_ids):
        """Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(
            self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1
        )
        # robots that walked far enough progress to harder terrains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (
            distance
            < torch.norm(self.commands[env_ids, :2], dim=1)
            * self.max_episode_length_s
            * 0.5
        ) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0),
        )  # (the minimal level is zero)
        self.env_origins[env_ids] = self.terrain_origins[
            self.terrain_levels[env_ids], self.terrain_types[env_ids]
        ]

    def update_command_curriculum(self, env_ids):
        """Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if (
            torch.mean(self.episode_sums_raw["tracking_lin_vel"][env_ids])
            / self.max_episode_length
            > 0.8
        ):
            self.command_ranges["lin_vel_x"][0] = np.clip(
                self.command_ranges["lin_vel_x"][0] - 0.5,
                -self.cfg.commands.max_curriculum,
                0.0,
            )
            self.command_ranges["lin_vel_x"][1] = np.clip(
                self.command_ranges["lin_vel_x"][1] + 0.5,
                0.0,
                self.cfg.commands.max_curriculum,
            )

    # ----------------------------------------
    def _init_buffers(self):
        """Initialize torch tensors which will contain simulation states and processed quantities"""
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
            self.num_envs, -1, 3
        )  # shape: num_envs, num_bodies, xyz axis
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state).view(
            self.num_envs, self.num_bodies, 13
        )
        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.gravity_vec = to_torch(
            get_axis_params(-1.0, self.up_axis_idx), device=self.device
        ).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1.0, 0.0, 0.0], device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.torques = torch.zeros(
            self.num_envs,
            self.num_dofs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.controller_input = torch.zeros(
            self.num_envs,
            self.num_dofs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_last_actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.last_foot_z = (
            torch.ones(
                self.num_envs,
                len(self.foot_indices),
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            * self.cfg.asset.foot_half_z_size
        )
        self.feet_heights = torch.zeros(
            (self.num_envs, len(self.foot_indices)), device=self.device
        )
        self.commands = torch.zeros(
            self.num_envs,
            self.cfg.commands.num_commands,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # x vel, y vel, yaw vel
        self.commands_scales = torch.tensor(
            [
                self.obs_scales.get("base_lin_vel", 1),
                self.obs_scales.get("base_lin_vel", 1),
                self.obs_scales.get("base_ang_vel", 1),
            ],
            device=self.device,
            requires_grad=False,
        )
        self.foot_air_time = torch.zeros(
            self.num_envs,
            self.foot_indices.shape[0],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_contacts = torch.zeros(
            self.num_envs,
            len(self.foot_indices),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        self.base_lin_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 7:10]
        )
        self.base_ang_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
            self.measured_heights = torch.zeros(
                self.num_envs,
                self.num_height_points,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
        self.stand_still_idx = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
        )
        self.heading_target = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        # joint positions offsets and PD gains
        self.p_gains = torch.zeros(
            self.num_envs,
            self.num_dofs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.d_gains = torch.zeros(
            self.num_envs,
            self.num_dofs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.default_dof_pos = torch.zeros(
            self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.action_scales = torch.zeros(
            len(self.cfg.env.action_joints), device=self.device
        )
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[:, i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[:, i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[:, i] = 0.0
                self.d_gains[:, i] = 0.0
                if self.cfg.control.control_type in ["P", "V"]:
                    print(
                        f"PD gain of joint {name} were not defined, setting them to zero"
                    )
        for idx, action_joint_name in enumerate(self.cfg.env.action_joints):
            for (
                action_joint_type,
                action_scale,
            ) in self.cfg.control.action_scales.items():
                if action_joint_type in action_joint_name:
                    self.action_scales[idx] = action_scale
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.domain_rands.init_buffer_after_create_env()
        self.obs_dict = {}
        self.goal_dict = {}

    def get_phase(self):
        cycle_time = self.cfg.gait.cycle_time
        phase = self.episode_length_buf * self.dt / cycle_time
        phase[self.stand_still_idx] = 0.0
        return phase

    def _prepare_reward_function(self):
        """Prepares a list of reward functions, which will be called to compute the total reward.
        Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non-zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            if scale == 0:
                continue
            self.reward_names.append(name)
            name = "reward_" + name
            self.reward_functions.append(getattr(self.rewards, name))

        # reward episode sums
        self.episode_sums_raw = {
            name: torch.zeros(
                self.num_envs,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            for name in self.reward_scales.keys()
        }
        self.episode_sums_scaled = {
            name: torch.zeros(
                self.num_envs,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            for name in self.reward_scales.keys()
        }

    def _create_ground_plane(self):
        """Adds a ground plane to the simulation, sets friction and restitution based on the cfg."""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """Adds a heightfield terrain to the simulation, sets parameters based on the cfg."""
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.cfg.border_size
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = (
            torch.tensor(self.terrain.heightsamples)
            .view(self.terrain.tot_rows, self.terrain.tot_cols)
            .to(self.device)
        )

    def _create_trimesh(self):
        """Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        #"""
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(
            self.sim,
            self.terrain.vertices.flatten(order="C"),
            self.terrain.triangles.flatten(order="C"),
            tm_params,
        )
        self.height_samples = (
            torch.tensor(self.terrain.heightsamples)
            .view(self.terrain.tot_rows, self.terrain.tot_cols)
            .to(self.device)
        )

    def _create_envs(self):
        """Creates environments:
        1. loads the robot URDF/MJCF asset,
        2. For each environment
           2.1 creates the environment,
           2.2 calls DOF and Rigid shape properties callbacks,
           2.3 create actor with these properties and add them to the env
        3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(
            ENGINEAI_GYM_PACKAGE_DIR=ENGINEAI_GYM_PACKAGE_DIR
        )
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = (
            self.cfg.asset.replace_cylinder_with_capsule
        )
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )
        self.num_dofs = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        self.body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(self.body_names)
        self.action_joint_indices = [
            idx
            for idx, joint_name in enumerate(self.cfg.env.action_joints)
            if joint_name in self.cfg.env.action_joints
        ]
        if self.cfg.asset.foot_name is not None:
            foot_names = [s for s in self.body_names if self.cfg.asset.foot_name in s]
        else:
            foot_names = []
        if self.cfg.asset.knee_name is not None:
            knee_names = [s for s in self.body_names if self.cfg.asset.knee_name in s]
        else:
            knee_names = []
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in self.body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in self.body_names if name in s])

        base_init_state_list = (
            self.cfg.init_state.pos
            + self.cfg.init_state.rot
            + self.cfg.init_state.lin_vel
            + self.cfg.init_state.ang_vel
        )
        self.base_init_state = to_torch(
            base_init_state_list, device=self.device, requires_grad=False
        )
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.actor_handles = []
        self.envs = []
        self.env_frictions = torch.zeros(
            self.num_envs, 1, dtype=torch.float32, device=self.device
        )
        self.domain_rands.init_buffer_on_create_env()
        self.domain_rands.init_rand_vec_on_create_env()
        self.body_mass = torch.zeros(
            self.num_envs,
            1,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(
                self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs))
            )
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1.0, 1.0, (2, 1), device=self.device).squeeze(
                1
            )
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(
                rigid_shape_props_asset, i
            )
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(
                env_handle,
                robot_asset,
                start_pose,
                self.cfg.asset.name,
                i,
                self.cfg.asset.self_collisions,
                0,
            )
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(
                env_handle, actor_handle
            )
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(
                env_handle, actor_handle, body_props, recomputeInertia=True
            )
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.foot_indices = torch.zeros(
            len(foot_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(foot_names)):
            self.foot_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], foot_names[i]
            )
        self.knee_indices = torch.zeros(
            len(knee_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], knee_names[i]
            )

        self.penalised_contact_indices = torch.zeros(
            len(penalized_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], penalized_contact_names[i]
            )

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], termination_contact_names[i]
            )

    def _get_env_origins(self):
        """Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
        Otherwise, create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(
                self.num_envs, 3, device=self.device, requires_grad=False
            )
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum:
                max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(
                0, max_init_level + 1, (self.num_envs,), device=self.device
            )
            self.terrain_types = torch.div(
                torch.arange(self.num_envs, device=self.device),
                (self.num_envs / self.cfg.terrain.num_cols),
                rounding_mode="floor",
            ).to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = (
                torch.from_numpy(self.terrain.env_origins)
                .to(self.device)
                .to(torch.float)
            )
            self.env_origins[:] = self.terrain_origins[
                self.terrain_levels, self.terrain_types
            ]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(
                self.num_envs, 3, device=self.device, requires_grad=False
            )
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[: self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[: self.num_envs]
            self.env_origins[:, 2] = 0.0

    def _parse_cfg(self):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = {}
        for obs_name in self.cfg.env.obs_list:
            if isinstance(self.cfg.normalization.obs_scales.get(obs_name, 1), dict):
                obs_scales_tensor = torch.zeros(
                    self.num_dofs, device=self.device, dtype=torch.float
                )
                for idx, joint_name in enumerate(self.dof_names):
                    for (
                        joint_type,
                        obs_scale,
                    ) in self.cfg.normalization.obs_scales.get(obs_name).items():
                        if joint_type in joint_name:
                            obs_scales_tensor[idx] = obs_scale
                self.obs_scales[obs_name] = obs_scales_tensor
            else:
                self.obs_scales[obs_name] = self.cfg.normalization.obs_scales.get(
                    obs_name, 1
                )
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ["heightfield", "trimesh"]:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

    def _draw_debug_vis(self):
        """Draws visualizations for debugging (slows down simulation a lot).
        Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = (
                quat_apply_yaw(
                    self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]
                )
                .cpu()
                .numpy()
            )
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(
                    sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose
                )

    def _init_height_points(self):
        """Returns points at which the height measurements are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(
            self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False
        )
        x = torch.tensor(
            self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False
        )
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(
            self.num_envs,
            self.num_height_points,
            3,
            device=self.device,
            requires_grad=False,
        )
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == "plane":
            return torch.zeros(
                self.num_envs,
                self.num_height_points,
                device=self.device,
                requires_grad=False,
            )
        elif self.cfg.terrain.mesh_type == "none":
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(
                self.base_quat[env_ids].repeat(1, self.num_height_points),
                self.height_points[env_ids],
            ) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(
                self.base_quat.repeat(1, self.num_height_points), self.height_points
            ) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
