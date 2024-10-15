import torch
import torch.nn as nn
from torch import autograd


class AmpDiscriminator(nn.Module):
    def __init__(self, network, amp_reward_coef, task_reward_lerp, device="cpu"):
        super().__init__()

        self.device = device
        self.amp_reward_coef = amp_reward_coef
        self.network = network.to(self.device)

        self.network.train()
        self.task_reward_lerp = task_reward_lerp

    def forward(self, x):
        return self.network(x)

    def compute_grad_pen(self, expert_state, expert_next_state, lambda_=10):
        expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
        expert_data.requires_grad = True

        disc = self.network.pure_forward(expert_data)
        ones = torch.ones(disc.size(), device=disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=expert_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Enforce that the grad norm approaches 0.
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_pen

    def predict_amp_reward(self, state, next_state, task_reward):
        with torch.no_grad():
            self.eval()

            d = self.network(torch.cat([state, next_state], dim=-1))
            reward = self.amp_reward_coef * torch.clamp(
                1 - (1 / 4) * torch.square(d - 1), min=0
            )
            if self.task_reward_lerp > 0:
                reward = self._lerp_reward(reward, task_reward.unsqueeze(-1))
            self.train()
        reward = reward.squeeze()
        if reward.dim() == 0:
            reward = reward.unsqueeze(0)
        return reward, d

    def _lerp_reward(self, disc_r, task_r):
        r = (1.0 - self.task_reward_lerp) * disc_r + self.task_reward_lerp * task_r
        return r
