from engineai_gym.envs.base.domain_rands.domain_rands_base import DomainRandsBase
from isaacgym.torch_utils import *
from copy import deepcopy


class DomainRandsTypeActionLag(DomainRandsBase):
    def init_domain_rands(self):
        self.action_lag = (
            getattr(self.env.cfg.domain_rands.action_lag, "action_lag_timesteps", 0) > 0
            or self.env.cfg.domain_rands.action_lag.randomize_action_lag_timesteps
        )

    def init_buffer_after_create_env(self):
        if self.action_lag:
            self.action_lag_timestep = torch.zeros(
                self.env.num_envs, device=self.env.device, dtype=int
            )
            if self.env.cfg.domain_rands.action_lag.randomize_action_lag_timesteps:
                self.action_lag_buffer = torch.zeros(
                    self.env.num_envs,
                    self.env.num_actions,
                    self.env.cfg.domain_rands.action_lag.action_lag_timesteps_range[1]
                    + 1,
                    device=self.env.device,
                )
                if (
                    self.env.cfg.domain_rands.action_lag.randomize_action_lag_timesteps_perstep
                ):
                    self.last_action_lag_timestep = torch.zeros(
                        self.env.num_envs, device=self.env.device, dtype=int
                    )
            else:
                self.action_lag_buffer = torch.zeros(
                    self.env.num_envs,
                    self.env.num_actions,
                    self.env.cfg.domain_rands.action_lag.action_lag_timesteps + 1,
                    device=self.env.device,
                )
                self.action_lag_timestep[
                    :
                ] = self.env.cfg.domain_rands.action_lag.action_lag_timesteps

    def init_rand_vec_on_reset_idx(self, env_ids):
        if self.action_lag:
            if self.env.cfg.domain_rands.action_lag.randomize_action_lag_timesteps:
                self.action_lag_timestep[env_ids] = torch.randint(
                    self.env.cfg.domain_rands.action_lag.action_lag_timesteps_range[0],
                    self.env.cfg.domain_rands.action_lag.action_lag_timesteps_range[1]
                    + 1,
                    (len(env_ids),),
                    device=self.env.device,
                )

    def init_rand_vec_on_step(self):
        if self.action_lag:
            if self.env.cfg.domain_rands.action_lag.randomize_action_lag_timesteps:
                if (
                    self.env.cfg.domain_rands.action_lag.randomize_action_lag_timesteps_perstep
                ):
                    self.action_lag_timestep = torch.randint(
                        self.env.cfg.domain_rands.action_lag.action_lag_timesteps_range[
                            0
                        ],
                        self.env.cfg.domain_rands.action_lag.action_lag_timesteps_range[
                            1
                        ]
                        + 1,
                        (self.env.num_envs,),
                        device=self.env.device,
                    )

    def process_on_reset_idx(self, env_ids):
        if self.action_lag:
            self.action_lag_buffer[env_ids, :, :] = 0.0
            if self.env.cfg.domain_rands.action_lag.randomize_action_lag_timesteps:
                if (
                    self.env.cfg.domain_rands.action_lag.randomize_action_lag_timesteps_perstep
                ):
                    self.last_action_lag_timestep[
                        env_ids
                    ] = self.env.cfg.domain_rands.action_lag.action_lag_timesteps_range[
                        1
                    ]

    def process_on_decimation(self):
        if self.action_lag:
            if self.env.cfg.domain_rands.action_lag.randomize_action_lag_timesteps:
                self.action_lag_buffer[:, :, 1:] = self.action_lag_buffer[
                    :,
                    :,
                    : self.env.cfg.domain_rands.action_lag.action_lag_timesteps_range[
                        1
                    ],
                ]
                self.action_lag_buffer[:, :, 0] = self.env.actions
                if (
                    self.env.cfg.domain_rands.action_lag.randomize_action_lag_timesteps_perstep
                ):
                    cond = self.action_lag_timestep > self.last_action_lag_timestep + 1
                    self.action_lag_timestep[cond] = (
                        self.last_action_lag_timestep[cond] + 1
                    )
                    self.last_action_lag_timestep = self.action_lag_timestep.clone()
            else:
                self.action_lag_buffer[:, :, 1:] = self.action_lag_buffer[
                    :, :, : self.env.cfg.domain_rands.action_lag.action_lag_timesteps
                ]
                self.action_lag_buffer[:, :, 0] = self.env.actions
            self.lagged_actions = self.action_lag_buffer[
                torch.arange(self.env.num_envs), :, self.action_lag_timestep.long()
            ]
