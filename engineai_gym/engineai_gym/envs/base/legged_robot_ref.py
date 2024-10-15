import torch
from isaacgym.torch_utils import *
from isaacgym import gymtorch
from .legged_robot import LeggedRobot
from engineai_gym import ENGINEAI_GYM_PACKAGE_DIR
from engineai_rl_lib.ref_state.ref_state_loader import RefStateLoader


class LeggedRobotRef(LeggedRobot):
    def __init__(
        self,
        obs_class,
        goal_class,
        domain_rand_class,
        reward_class,
        cfg,
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
        if self.cfg.ref_state.ref_state_loader:
            self.ref_state_loader = RefStateLoader(
                motion_files_path=self.cfg.ref_state.motion_files_path.format(
                    ENGINEAI_GYM_PACKAGE_DIR=ENGINEAI_GYM_PACKAGE_DIR
                ),
                device=self.device,
                time_between_states=self.dt,
                data_mapping=self.cfg.ref_state.data_mapping,
            )
            if self.cfg.ref_state.data_mapping is not None:
                ref_state_components = []
                if "absolute_dof_pos" in self.cfg.ref_state.data_mapping:
                    ref_state_components.append("absolute_dof_pos")
                if "dof_vel" in self.cfg.ref_state.data_mapping:
                    ref_state_components.append("dof_vel")
                if "base_pos" in self.cfg.ref_state.data_mapping:
                    ref_state_components.append("base_pos")
                if "base_rot" in self.cfg.ref_state.data_mapping:
                    ref_state_components.append("base_rot")
                if "base_lin_vel" in self.cfg.ref_state.data_mapping:
                    ref_state_components.append("base_lin_vel")
                if "base_ang_vel" in self.cfg.ref_state.data_mapping:
                    ref_state_components.append("base_ang_vel")
                print(f"{ref_state_components} will be reset by ref_state!")
        elif self.cfg.ref_state.ref_state_init:
            raise RuntimeError(
                "ref_state_init is not available since ref_state_loader doesn't exist!"
            )

    def reset_dofs_and_root_states(self, env_ids):
        if self.cfg.ref_state.ref_state_init:
            ref_mask = torch_rand_float(
                0, 1, (len(env_ids), 1), device=self.device
            ).squeeze(1) < getattr(self.cfg.ref_state, "ref_state_init_prob", 0)
            ref_state = self.ref_state_loader.get_frame_at_times(
                *self.ref_state_loader.sample_idxes_and_times(len(env_ids[ref_mask]))
            )
            self._reset_dofs(
                env_ids,
                ref_state.get("absolute_dof_pos", None),
                ref_state.get("dof_vel", None),
                ref_mask,
            )
            self._reset_root_states(
                env_ids,
                ref_state.get("base_pos", None),
                ref_state.get("base_rot", None),
                ref_state.get("base_lin_vel", None),
                ref_state.get("base_ang_vel", None),
                ref_mask,
            )
        else:
            self._reset_dofs(env_ids)
            self._reset_root_states(env_ids)

    def _reset_dofs(self, env_ids, absolute_dof_pos=None, dof_vel=None, ref_mask=None):
        """Resets DOF position and velocities of selected environments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environment ids
        """
        if ref_mask is None:
            ref_mask = torch.ones_like(env_ids, dtype=torch.bool, device=self.device)

        (
            dof_pos_ref_mask,
            absolute_dof_pos,
        ) = self.generate_ref_mask_and_empty_ref_states(
            absolute_dof_pos, self.dof_pos.shape, env_ids, ref_mask
        )
        self.dof_pos[
            env_ids[~dof_pos_ref_mask]
        ] = self.default_dof_pos + torch_rand_float(
            -0.1,
            0.1,
            (len(env_ids[~dof_pos_ref_mask]), self.num_dofs),
            device=self.device,
        )
        self.dof_pos[env_ids[dof_pos_ref_mask]] = absolute_dof_pos

        dof_vel_ref_mask, dof_vel = self.generate_ref_mask_and_empty_ref_states(
            dof_vel, self.dof_vel.shape, env_ids, ref_mask
        )
        self.dof_vel[env_ids[~dof_vel_ref_mask]] = 0.0
        self.dof_vel[env_ids[dof_vel_ref_mask]] = dof_vel

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
        root_pos=None,
        root_rot=None,
        root_lin_vel=None,
        root_ang_vel=None,
        ref_mask=None,
    ):
        """Resets ROOT states position and velocities of selected environments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environment ids
        """
        if ref_mask is None:
            ref_mask = torch.ones_like(env_ids, dtype=torch.bool, device=self.device)

        self.root_states[env_ids] = self.base_init_state

        # base position
        root_pos_ref_mask, root_pos = self.generate_ref_mask_and_empty_ref_states(
            root_pos, self.root_states[:, :3].shape, env_ids, ref_mask
        )
        self.root_states[env_ids[root_pos_ref_mask], :3] = root_pos
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        if self.custom_origins:
            self.root_states[env_ids, :2] += torch_rand_float(
                -4.0, 4.0, (len(env_ids), 2), device=self.device
            )  # xy position within 4m of the center

        root_rot_ref_mask, root_rot = self.generate_ref_mask_and_empty_ref_states(
            root_rot, self.root_states[:, 3:7].shape, env_ids, ref_mask
        )
        self.root_states[env_ids[root_rot_ref_mask], 3:7] = root_rot

        root_lin_vel_mask, root_lin_vel = self.generate_ref_mask_and_empty_ref_states(
            root_lin_vel, self.root_states[:, 7:10].shape, env_ids, ref_mask
        )
        self.root_states[env_ids[root_lin_vel_mask], 7:10] = quat_rotate(
            root_rot, root_lin_vel
        )
        root_ang_vel_mask, root_ang_vel = self.generate_ref_mask_and_empty_ref_states(
            root_ang_vel, self.root_states[:, 10:13].shape, env_ids, ref_mask
        )
        self.root_states[env_ids[root_ang_vel_mask], 10:13] = quat_rotate(
            root_rot, root_ang_vel
        )

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

    def generate_ref_mask_and_empty_ref_states(
        self, ref_states, original_size, env_ids, ref_mask
    ):
        if ref_states is None:
            state_ref_mask = torch.zeros_like(
                env_ids, dtype=torch.bool, device=self.device
            )
            ref_states = torch.zeros((0, *original_size[1:]), device=self.device)
        else:
            state_ref_mask = ref_mask
        return state_ref_mask, ref_states
