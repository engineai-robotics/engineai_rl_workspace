from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from engineai_rl.modules import *
from engineai_rl.storage import RolloutStorage
from engineai_rl.algos.base.algo_base import AlgoBase


class Ppo(AlgoBase):
    def __init__(
        self,
        networks,
        policy_cfg,
        env,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        **kwargs,
    ):
        super().__init__(networks, policy_cfg, env, device, **kwargs)
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        if policy_cfg.get("min_normalized_std", None) is not None:
            min_normalized_std = policy_cfg.pop("min_normalized_std")
            policy_cfg["min_std"] = torch.tensor(
                min_normalized_std, device=self.device
            ) * (torch.abs(env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0])).to(
                self.device
            )

        # PPO components
        actor_critic_class = eval(policy_cfg.pop("class_name"))
        self.init_actor_critic(policy_cfg, actor_critic_class, networks)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_actor_critic(self, policy_cfg, actor_critic_class, networks):
        self.actor_critic: ActorCritic = actor_critic_class(
            actor=networks["actor"],
            critic=networks["critic"],
            num_actions=networks["actor"].num_output_dim,
            **policy_cfg,
        ).to(self.device)

    def init_storage(
        self, num_envs, num_transitions_per_env, input_shapes, input_types, action_shape
    ):
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            input_shapes,
            input_types,
            action_shape,
            self.device,
        )

    def eval_mode(self):
        self.actor_critic.eval_mode()

    def train_mode(self):
        self.actor_critic.train_mode()

    @property
    def inference_policy(self):
        return self.actor_critic.act_inference

    def act(self, inputs):
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(inputs).detach()
        self.transition.values = self.actor_critic.evaluate(inputs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.inputs = inputs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, **kwargs):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on timeouts
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values
                * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, inputs):
        last_values = self.actor_critic.evaluate(inputs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )
        for (
            inputs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            self.actor_critic.act(
                inputs_batch,
                masks=masks_batch,
                hidden_states=hid_states_batch[0],
            )
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(
                actions_batch
            )
            value_batch = self.actor_critic.evaluate(
                inputs_batch,
                masks=masks_batch,
                hidden_states=hid_states_batch[1],
            )
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (
                            torch.square(old_sigma_batch)
                            + torch.square(old_mu_batch - mu_batch)
                        )
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(
                actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            )
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()

        return {
            "mean_value_loss": mean_value_loss,
            "mean_surrogate_loss": mean_surrogate_loss,
        }
