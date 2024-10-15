from typing import Tuple

import torch
import torch.nn as nn


class RunningMeanStd(nn.Module):
    def __init__(self, epsilon: float = 1e-4, num_input_dim: Tuple[int, ...] = ()):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        :param epsilon: helps with arithmetic issues
        :param num_input_dim: the shape of the data stream's output
        """
        super().__init__()
        self.register_buffer("mean", torch.zeros(num_input_dim, dtype=torch.float))
        self.register_buffer("var", torch.ones(num_input_dim, dtype=torch.float))
        self.count = epsilon

    def update(self, x) -> None:
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = (
            m_a
            + m_b
            + torch.square(delta)
            * self.count
            * batch_count
            / (self.count + batch_count)
        )
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class NormalizerAmp(RunningMeanStd):
    def __init__(
        self, num_input_dim, epsilon=1e-4, clip_obs=10.0, require_grad: bool = False
    ):
        super().__init__(num_input_dim=num_input_dim)
        self.epsilon = epsilon
        self.clip_obs = clip_obs
        self.require_grad = require_grad

    def forward(self, x):
        return torch.clip(
            (x - self.mean) / torch.sqrt(self.var + self.epsilon),
            -self.clip_obs,
            self.clip_obs,
        )

    def update_normalizer(self, rollouts, expert_loader):
        policy_data_generator = rollouts.feed_forward_generator_amp(
            None, mini_batch_size=expert_loader.batch_size
        )
        expert_data_generator = expert_loader.dataset.feed_forward_generator_amp(
            expert_loader.batch_size
        )

        for expert_batch, policy_batch in zip(
            expert_data_generator, policy_data_generator
        ):
            self.update(torch.vstack(tuple(policy_batch) + tuple(expert_batch)))
