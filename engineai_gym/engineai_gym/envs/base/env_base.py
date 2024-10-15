import sys, inspect
from isaacgym import gymapi
from isaacgym import gymutil
import torch


# Base class for RL tasks
class EnvBase:
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
        self.gym = gymapi.acquire_gym()
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.headless = headless

        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        if sim_device_type == "cuda" and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = "cpu"

        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id

        self.num_envs = cfg.env.num_envs
        self.num_actions = len(cfg.env.action_joints)

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        self.obs = obs_class(self)
        self.goals = goal_class(self)
        self.domain_rands = domain_rand_class(self)
        self.domain_rands.init_domain_rands()
        self.rewards = reward_class(self)
        # allocate buffers
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.time_out_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )

        self.extras = {}

        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync"
            )

    def _pre_compute_observations(self):
        pass

    def compute_observations(self):
        """Computes observations"""
        non_lagged_obs_dict = {
            obs_name: getattr(self.obs, obs_name)() * self.obs_scales.get(obs_name, 1)
            for obs_name in self.cfg.env.obs_list
        }
        lagged_obs_raw_dict = {
            obs_name: getattr(self.obs, obs_name)(lag=True)
            for obs_name in self.cfg.env.obs_list
            if "lag" in inspect.signature(getattr(self.obs, obs_name)).parameters
        }
        lagged_obs_dict = {
            obs_name: lagged_obs_raw * self.obs_scales.get(obs_name, 1)
            for obs_name, lagged_obs_raw in lagged_obs_raw_dict.items()
            if lagged_obs_raw is not None
        }
        return {"non_lagged_obs": non_lagged_obs_dict, "lagged_obs": lagged_obs_dict}

    def compute_goals(self):
        """Computes goals"""
        return {
            goal_name: getattr(self.goals, goal_name)()
            for goal_name in self.cfg.env.goal_list
        }

    def get_observations(self):
        return self.obs_dict

    def reset_idx(self, env_ids):
        """Reset selected robots"""
        raise NotImplementedError

    def reset(self):
        """Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs_dict, goals, _, _, _ = self.step(
            torch.zeros(
                self.num_envs, self.num_actions, device=self.device, requires_grad=False
            )
        )
        return obs_dict, goals

    def step(self, actions):
        raise NotImplementedError

    def render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != "cpu":
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)
