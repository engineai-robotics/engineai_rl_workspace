#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(
        self,
        num_actions,
        actor,
        critic,
        init_noise_std=1.0,
        fixed_std=False,
        min_std=None,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.actor = actor
        self.critic = critic

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        std = init_noise_std * torch.ones(num_actions)
        self.fixed_std = fixed_std
        if self.fixed_std:
            self.std = torch.tensor(std)
        else:
            self.std = nn.Parameter(std)
        self.min_std = min_std
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # seems that we get better performance without init  # self.init_memory_weights(self.memory_a, 0.001, 0.)  # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(
                mod for mod in sequential if isinstance(mod, nn.Linear)
            )
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, inputs, **kwargs):
        if not self.fixed_std and self.min_std is not None:
            self.std.data = self.std.data.clamp(min=self.min_std)
        self.update_distribution(inputs["actor"])
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, inputs):
        actions_mean = self.actor(inputs["actor"])
        return actions_mean

    def evaluate(self, inputs, **kwargs):
        value = self.critic(inputs["critic"])
        return value

    def eval_mode(self):
        self.eval()

    def train_mode(self):
        self.train()
