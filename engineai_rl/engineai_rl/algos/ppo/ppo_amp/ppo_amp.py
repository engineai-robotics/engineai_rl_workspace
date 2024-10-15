#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from engineai_rl.storage import RolloutStorage
from engineai_rl.storage.replay_buffer import ReplayBuffer
from engineai_rl.algos.ppo.ppo_amp.amp_discriminator import AmpDiscriminator
from engineai_rl_lib.device import input_to_device
from engineai_rl.algos.ppo import Ppo


class PpoAmp(Ppo):
    def __init__(
        self,
        networks,
        policy_cfg,
        env,
        obs_cfg=None,
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
        amp_replay_buffer_size=100000,
        amp_discriminator_name="AmpDiscriminator",
        amp_reward_coef=2.0,
        amp_task_reward_lerp=0.3,
        preload_batches=False,
        num_preload_batches=1000000,
        **kwargs,
    ):

        super().__init__(
            networks,
            policy_cfg,
            env,
            num_learning_epochs,
            num_mini_batches,
            clip_param,
            gamma,
            lam,
            value_loss_coef,
            entropy_coef,
            learning_rate,
            max_grad_norm,
            use_clipped_value_loss,
            schedule,
            desired_kl,
            device,
            **kwargs,
        )
        self.obs_cfg = obs_cfg
        discriminator_class = eval(amp_discriminator_name)
        self.discriminator = discriminator_class(
            networks["discriminator"],
            amp_reward_coef,
            amp_task_reward_lerp,
            device=self.device,
        )
        self.amp_transition = RolloutStorage.Transition()
        self.amp_storage = ReplayBuffer(
            self.discriminator.network.num_input_dim // 2,
            amp_replay_buffer_size,
            device,
        )
        self.amp_data = env.ref_state_loader
        if preload_batches:
            self.amp_data.preload_batches(num_preload_batches)

        params = [
            {"params": self.actor_critic.parameters(), "name": "actor_critic"},
            {
                "params": self.discriminator.network.trunk.parameters(),
                "weight_decay": 10e-4,
                "name": "amp_trunk",
            },
            {
                "params": self.discriminator.network.head.parameters(),
                "weight_decay": 10e-2,
                "name": "amp_head",
            },
        ]
        self.optimizer = optim.Adam(params, lr=learning_rate)

    def process_env_step(self, rewards, dones, infos, inputs, next_inputs, **kwargs):
        rewards = self.discriminator.predict_amp_reward(
            inputs["amp"]["after_reset"], next_inputs["amp"]["before_reset"], rewards
        )[0]

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
        self.amp_storage.insert(
            inputs["amp"]["after_reset"], next_inputs["amp"]["before_reset"]
        )
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.amp_transition.clear()
        self.actor_critic.reset(dones)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_amp_loss = 0
        mean_grad_pen_loss = 0
        mean_policy_pred = 0
        mean_expert_pred = 0
        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )
        amp_policy_generator = self.amp_storage.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs
            * self.storage.num_transitions_per_env
            // self.num_mini_batches,
        )
        amp_expert_generator = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs
            * self.storage.num_transitions_per_env
            // self.num_mini_batches,
            self.obs_cfg["components"]["amp"]["obs_list"],
        )

        for sample, sample_amp_policy, sample_amp_expert in zip(
            generator, amp_policy_generator, amp_expert_generator
        ):

            (
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
            ) = sample
            self.actor_critic.act(
                inputs_batch, masks=masks_batch, hidden_states=hid_states_batch[0]
            )
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(
                actions_batch
            )
            value_batch = self.actor_critic.evaluate(
                inputs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
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

            # Discriminator loss.
            policy_state, policy_next_state = sample_amp_policy
            expert_state, expert_next_state = sample_amp_expert
            expert_state, expert_next_state = input_to_device(
                expert_state, self.device
            ), input_to_device(expert_next_state, self.device)

            policy_state_unnorm = torch.clone(policy_state)
            expert_state_unnorm = torch.clone(expert_state)

            policy_d = self.discriminator(
                torch.cat([policy_state, policy_next_state], dim=-1)
            )
            expert_d = self.discriminator(
                torch.cat([expert_state, expert_next_state], dim=-1)
            )
            expert_loss = torch.nn.MSELoss()(
                expert_d, torch.ones(expert_d.size(), device=self.device)
            )
            policy_loss = torch.nn.MSELoss()(
                policy_d, -1 * torch.ones(policy_d.size(), device=self.device)
            )
            amp_loss = 0.5 * (expert_loss + policy_loss)
            grad_pen_loss = self.discriminator.compute_grad_pen(
                expert_state, expert_next_state, lambda_=10
            )
            # Compute total loss.
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + amp_loss
                + grad_pen_loss
            )

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            if self.discriminator.network.normalizer is not None:
                self.discriminator.network.normalizer.update(
                    torch.cat((policy_state_unnorm, expert_state_unnorm), dim=-1)
                )

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_amp_loss += amp_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()
            mean_policy_pred += policy_d.mean().item()
            mean_expert_pred += expert_d.mean().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates
        self.storage.clear()

        return {
            "mean_value_loss": mean_value_loss,
            "mean_surrogate_loss": mean_surrogate_loss,
            "mean_amp_loss": mean_amp_loss,
            "mean_grad_pen_loss": mean_grad_pen_loss,
            "mean_policy_pred": mean_policy_pred,
            "mean_expert_pred": mean_expert_pred,
        }
