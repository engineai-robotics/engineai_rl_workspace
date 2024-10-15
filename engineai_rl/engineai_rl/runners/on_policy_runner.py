from __future__ import annotations

from .runner_base import RunnerBase
import os
import time
import torch
from collections import deque


class OnPolicyRunner(RunnerBase):
    """On-policy runner for training and evaluation."""

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        super().learn(init_at_random_ep_len=init_at_random_ep_len)
        inputs = self.reset()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    next_inputs, actions, dones, infos, rewards = self.step(
                        inputs, self.algo.act
                    )
                    self.algo.process_env_step(
                        rewards, dones, infos, inputs=inputs, next_inputs=next_inputs
                    )
                    inputs = next_inputs

                    if self.log_dir is not None:
                        # Book keeping
                        # note: we changed logging to use "log" instead of "episode" to avoid confusion with
                        # different types of logging data (rewards, curriculum, etc.)
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(
                            cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        lenbuffer.extend(
                            cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.algo.compute_returns(inputs)

            algo_infos = self.algo.update()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0 and not self.debug:
                self.save(
                    os.path.join(self.log_dir, "checkpoints", f"model_{it}.pt"),
                    infos={"input_sizes": self.env.input_sizes},
                )
            ep_infos.clear()

        self.save(
            os.path.join(
                self.log_dir,
                "checkpoints",
                f"model_{self.current_learning_iteration}.pt",
            ),
            infos={"input_sizes": self.env.input_sizes},
        )
