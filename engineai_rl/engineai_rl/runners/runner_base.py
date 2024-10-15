#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod
import os
import statistics
import time
import torch
from collections import deque
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

import engineai_rl
from engineai_rl.modules.networks import *
from engineai_rl.algos import *
from engineai_rl.wrapper.input_retrival_env_wrapper import InputRetrivalEnvWrapper
from engineai_rl.modules.normalizers import *
from engineai_rl_lib.dict_operations import convert_dicts


class RunnerBase(ABC):
    def __init__(
        self, env: InputRetrivalEnvWrapper, device="cpu", writer_cfg=None, debug=False
    ):
        super().__init__()
        self.device = device
        self.env = env
        self.writer_cfg = writer_cfg
        self.debug = debug

    def get_cfg(self, algo_cfg):
        self.algo_cfg = algo_cfg
        self.runner_cfg = algo_cfg["runner"]
        self.network_cfg = algo_cfg["networks"]
        self.param_cfg = algo_cfg["params"]
        self.policy_cfg = algo_cfg["policy"]

    def init_runner(self, algo_class, log_dir):
        networks = self.init_network()
        self.init_algo(algo_class, networks)
        self.init_logger(log_dir)
        self.init_writer()

    def init_logger(self, log_dir):
        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

    def init_algo(self, algo_class, networks):
        self.algo: Ppo = algo_class(
            networks,
            self.policy_cfg,
            obs_cfg=self.env.obs_cfg,
            env=self.env,
            device=self.device,
            **self.param_cfg,
        )
        self.num_steps_per_env = self.runner_cfg["num_steps_per_env"]
        self.save_interval = self.runner_cfg["save_interval"]
        # init storage and model
        self.algo.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            self.env.input_sizes,
            self.env.obs_types,
            [self.env.num_actions],
        )

    def init_network(self):
        networks = {}
        training_networks = self.network_cfg.pop("training")
        inference_networks = self.network_cfg.pop("inference")
        for network in self.network_cfg:
            network_class = eval(self.network_cfg[network].pop("class_name"))
            network_input_infos = self.network_cfg[network].pop("input_infos")
            input_dim_infos = {}
            for network_input_name, network_input_type in network_input_infos.items():
                if isinstance(network_input_type, list):
                    input_dim_infos[network_input_name] = 0
                    for network_input_subtype in network_input_type:
                        if isinstance(network_input_subtype, int):
                            input_dim_infos[network_input_name] += network_input_subtype
                        elif network_input_subtype in self.env.input_sizes:
                            input_dim_infos[network_input_name] += self.env.input_sizes[
                                network_input_subtype
                            ]
                        else:
                            raise ValueError(
                                f"Network input type {network_input_subtype} not supported"
                            )
                else:
                    if isinstance(network_input_type, int):
                        input_dim_infos[network_input_name] = network_input_type
                    elif network_input_type in self.env.input_sizes:
                        input_dim_infos[network_input_name] = self.env.input_sizes[
                            network_input_type
                        ]
                    else:
                        raise ValueError(
                            f"Network input type {network_input_type} not supported"
                        )
            network_output_infos = self.network_cfg[network].pop("output_infos")
            output_dim_infos = {}
            for (
                network_output_name,
                network_output_type,
            ) in network_output_infos.items():
                if isinstance(network_output_type, list):
                    input_dim_infos[network_output_name] = 0
                    for network_output_subtype in network_output_type:
                        if network_output_subtype == "action":
                            output_dim_infos[
                                network_output_name
                            ] += self.env.num_actions
                        elif network_output_subtype == "value":
                            output_dim_infos[network_output_name] += 1
                        elif isinstance(network_output_subtype, int):
                            output_dim_infos[
                                network_output_name
                            ] += network_output_subtype
                        else:
                            raise ValueError(
                                f"Network output type {network_output_subtype} not supported"
                            )
                else:
                    if network_output_type == "action":
                        output_dim_infos[network_output_name] = self.env.num_actions
                    elif network_output_type == "value":
                        output_dim_infos[network_output_name] = 1
                    elif isinstance(network_output_type, int):
                        output_dim_infos[network_output_name] = network_output_type
                    else:
                        raise ValueError(
                            f"Network output type {network_output_type} not supported"
                        )
            if self.network_cfg[network].get("normalizer_class_name", False):
                normalizer_class = eval(
                    self.network_cfg[network].pop("normalizer_class_name")
                )
                normalizer = normalizer_class(
                    **input_dim_infos,
                    **self.network_cfg[network].pop("normalizer_args"),
                )
            else:
                normalizer = None
            networks[network] = network_class(
                **input_dim_infos,
                **output_dim_infos,
                normalizer=normalizer,
                **self.network_cfg[network],
            ).to(self.device)
        return networks

    @abstractmethod
    def learn(self, init_at_random_ep_len: bool = False):
        checkpoint_path = os.path.join(self.log_dir, "checkpoints")
        if not os.path.isdir(checkpoint_path) and not self.debug:
            os.makedirs(checkpoint_path, exist_ok=True)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        self.train_mode()

    def init_writer(self):
        # initialize writer
        if not self.debug:
            if self.log_dir is not None and self.writer is None:
                # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
                self.logger_type = self.algo_cfg.get("logger", "tensorboard")
                self.logger_type = self.logger_type.lower()

                if self.logger_type == "neptune":
                    from engineai_rl.utils.neptune_utils import NeptuneSummaryWriter

                    self.writer = NeptuneSummaryWriter(
                        log_dir=self.log_dir,
                        flush_secs=10,
                        cfg={**self.algo_cfg, **self.writer_cfg},
                    )
                    self.writer.log_config(
                        self.env.cfg, self.algo_cfg, self.param_cfg, self.policy_cfg
                    )
                    self.writer.save_file(os.path.join(self.log_dir, "git_info.txt"))
                elif self.logger_type == "wandb":
                    from engineai_rl.utils.wandb_utils import WandbSummaryWriter

                    self.writer = WandbSummaryWriter(
                        log_dir=self.log_dir,
                        flush_secs=10,
                        cfg={**self.algo_cfg, **self.writer_cfg},
                    )
                    self.writer.log_config(
                        self.env.cfg, self.algo_cfg, self.param_cfg, self.policy_cfg
                    )
                    self.writer.save_file(os.path.join(self.log_dir, "git_info.txt"))
                elif self.logger_type == "tensorboard":
                    self.writer = TensorboardSummaryWriter(
                        log_dir=self.log_dir, flush_secs=10
                    )
                else:
                    raise AssertionError("logger type not found")

    def step(self, inputs, act, set_goals_callback=None, set_goals_callback_args=None):
        actions = act(inputs)
        return self.env.step(actions, set_goals_callback, set_goals_callback_args)

    def reset(self, set_goals_callback=None, set_goals_callback_args=None):
        return self.env.reset(set_goals_callback, set_goals_callback_args)

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        if locs["ep_infos"]:
            converted_ep_infos = convert_dicts(locs["ep_infos"])
            for key in converted_ep_infos[0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in converted_ep_infos:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    elif len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if not self.debug:
                    if key.startswith("rewards/scaled"):
                        self.writer.add_scalar(
                            key.replace("rewards/scaled", "rewards_scaled"),
                            value,
                            locs["it"],
                        )
                    elif key.startswith("rewards/raw"):
                        self.writer.add_scalar(
                            key.replace("rewards/raw", "rewards_raw"), value, locs["it"]
                        )
                    else:
                        self.writer.add_scalar("Episode/" + key, value, locs["it"])
        algo_string = ""
        if locs["algo_infos"]:
            for key, value in locs["algo_infos"].items():
                if not isinstance(value, torch.Tensor):
                    locs["algo_infos"][key] = torch.Tensor([value])
                elif len(value.shape) == 0:
                    locs["algo_infos"][key] = value.unsqueeze(0)
                # log to logger and terminal
                if not self.debug:
                    self.writer.add_scalar(key, value, locs["it"])
                algo_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.algo.actor_critic.std.mean()
        fps = int(
            self.num_steps_per_env
            * self.env.num_envs
            / (locs["collection_time"] + locs["learn_time"])
        )
        if not self.debug:
            self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
            self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
            self.writer.add_scalar(
                "Perf/collection time", locs["collection_time"], locs["it"]
            )
            self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
            if len(locs["rewbuffer"]) > 0:
                self.writer.add_scalar(
                    "Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"]
                )
                self.writer.add_scalar(
                    "Train/mean_episode_length",
                    statistics.mean(locs["lenbuffer"]),
                    locs["it"],
                )
                if (
                    self.logger_type != "wandb"
                ):  # wandb does not support non-integer x-axis logging
                    self.writer.add_scalar(
                        "Train/mean_reward/time",
                        statistics.mean(locs["rewbuffer"]),
                        self.tot_time,
                    )
                    self.writer.add_scalar(
                        "Train/mean_episode_length/time",
                        statistics.mean(locs["lenbuffer"]),
                        self.tot_time,
                    )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Run name:':>{pad}} {self.runner_cfg["run_name"]} \n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )  # f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""  #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Run name:':>{pad}} {self.runner_cfg["run_name"]} \n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )  # f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""  #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += algo_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] - locs['start_iter'] + 1) * (locs['tot_iter'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None):
        model_state_dict = {
            name: network.state_dict() for name, network in self.algo.networks.items()
        }
        saved_dict = {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": self.algo.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        torch.save(saved_dict, path)

        # Upload model to external logging service
        if self.logger_type in ["neptune", "wandb"]:
            if self.writer_cfg["upload_model"]:
                self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, weights_only=True, map_location=self.device)
        for name, network in self.algo.networks.items():
            network.load_state_dict(loaded_dict["model_state_dict"][name])
        if load_optimizer:
            self.algo.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        policy = lambda x: self.algo.inference_policy(
            {
                policy_input: x[policy_input]
                for policy_input in self.env.obs_cfg["inference"]
            }
        ).detach()
        return policy

    def train_mode(self):
        self.algo.train_mode()

    def eval_mode(self):
        self.algo.eval_mode()
