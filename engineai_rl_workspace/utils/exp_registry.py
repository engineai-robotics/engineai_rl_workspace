import os
from datetime import datetime

from engineai_rl_workspace import ENGINEAI_WORKSPACE_ROOT_DIR
from .helpers import (
    get_args,
    update_cfg_from_args,
    get_load_run_path,
    get_load_checkpoint_path,
    get_last_run_path,
    set_seed,
    parse_sim_params,
)
from engineai_rl_lib.class_operations import class_to_dict
from engineai_gym.envs.base.legged_robot import LeggedRobot
from engineai_gym.envs.base.obs.obs import Obs
from engineai_gym.envs.base.goals.goals import Goals
from engineai_gym.envs.base.domain_rands.domain_rands import DomainRands
from engineai_gym.envs.base.rewards.rewards import Rewards
from engineai_gym.envs.base.config_legged_robot import ConfigLeggedRobot
from engineai_rl.runners.on_policy_runner import OnPolicyRunner
from engineai_rl.algos import Ppo
from engineai_rl.algos import ConfigPpo
from engineai_rl.wrapper.input_retrival_env_wrapper import InputRetrivalEnvWrapper


class ExpRegistry:
    def __init__(self):
        self.task_classes = {}
        self.obs_classes = {}
        self.goal_classes = {}
        self.domain_rand_classes = {}
        self.reward_classes = {}
        self.env_cfgs = {}
        self.runner_classes = {}
        self.algo_classes = {}
        self.algo_cfgs = {}
        self.log_dir = os.path.join(ENGINEAI_WORKSPACE_ROOT_DIR, "logs")

    def register(
        self,
        name,
        task_class=LeggedRobot,
        obs_class=Obs,
        goal_class=Goals,
        domain_rand_class=DomainRands,
        reward_class=Rewards,
        env_cfg=ConfigLeggedRobot(),
        runner_class=OnPolicyRunner,
        algo_class=Ppo,
        algo_cfg=ConfigPpo(),
    ):
        self.task_classes[name] = task_class
        self.obs_classes[name] = obs_class
        self.goal_classes[name] = goal_class
        self.domain_rand_classes[name] = domain_rand_class
        self.reward_classes[name] = reward_class
        self.env_cfgs[name] = env_cfg
        self.runner_classes[name] = runner_class
        self.algo_classes[name] = algo_class
        self.algo_cfgs[name] = algo_cfg

    def get_task_class(self, name: str):
        return self.task_classes[name]

    def get_obs_class(self, name: str):
        return self.obs_classes[name]

    def get_goal_class(self, name: str):
        return self.goal_classes[name]

    def get_domain_rand_class(self, name: str):
        return self.domain_rand_classes[name]

    def get_reward_class(self, name: str):
        return self.reward_classes[name]

    def get_runner_class(self, name: str):
        return self.runner_classes[name]

    def get_algo_class(self, name: str):
        return self.algo_classes[name]

    def get_cfgs(self, name):
        algo_cfg = self.algo_cfgs[name]
        env_cfg = self.env_cfgs[name]
        # copy seed
        env_cfg.seed = algo_cfg.runner.seed
        return env_cfg, algo_cfg

    def make_env(
        self,
        task_class,
        obs_class,
        goal_class,
        domain_rand_class,
        reward_class,
        args,
        env_cfg,
    ):
        """Creates an environment either from a registered name or from the provided config file.

        Args:
            name (string): Name of a registered env.
            args (Args, optional): Isaac Gym command line arguments. If None get_args() will be called. Defaults to None.
            env_cfg (Dict, optional): Environment config file used to override the registered config. Defaults to None.

        Raises:
            ValueError: Error if no registered env corresponds to 'name'

        Returns:
            isaacgym.VecTaskPython: The created environment
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        set_seed(env_cfg.seed)
        # parse sim params (convert to dict first)
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)
        env = task_class(
            cfg=env_cfg,
            obs_class=obs_class,
            goal_class=goal_class,
            domain_rand_class=domain_rand_class,
            reward_class=reward_class,
            sim_params=sim_params,
            physics_engine=args.physics_engine,
            sim_device=args.sim_device,
            headless=args.headless,
        )
        return env

    def make_alg_runner(self, env, name, args, log_dir):
        """Creates the training algorithm  either from a registered name or from the provided config file.

        Args:
            env (isaacgym.VecTaskPython): The environment to train
            name (string): Name of a registered env. If None, the config file will be used instead. Defaults to None.
            args (Args): Isaac Gym command line arguments. If None get_args() will be called. Defaults to None.
            log_dir (str): Logging directory.

        Returns:
            Runner: runner for the run
        """
        algo_cfg_dict = class_to_dict(self.algo_cfgs[name])
        runner_class = self.runner_classes[name]
        writer_cfg = {}
        if args.exp_name is not None:
            writer_cfg["wandb_project"] = args.exp_name
            writer_cfg["wandb_group"] = args.sub_exp_name
        if args.upload_model is not None:
            writer_cfg["upload_model"] = args.upload_model
        input_retrival_env_wrapper = InputRetrivalEnvWrapper(
            env, obs_cfg=algo_cfg_dict["input"], device=args.rl_device
        )
        runner = runner_class(
            input_retrival_env_wrapper,
            device=args.rl_device,
            writer_cfg=writer_cfg,
            debug=args.debug,
        )
        runner.get_cfg(algo_cfg_dict)
        runner.init_runner(algo_class=self.algo_classes[args.exp_name], log_dir=log_dir)
        # save resume path before creating a new log_dir

        if args.resume:
            # load previously trained model
            load_checkpoint = get_load_checkpoint_path(
                load_run=log_dir, checkpoint=args.checkpoint
            )
            print(f"Loading model from: {load_checkpoint}")
            runner.load(load_checkpoint)
        return runner

    def get_class_and_cfg(self, args=None, name=None):
        if name is None:
            raise ValueError("Please specify an experiment name!")

        # if no args passed get command line arguments
        if args is None:
            args = get_args()

        env_cfg, _ = self.get_cfgs(name)

        # load config files
        _, algo_cfg = self.get_cfgs(name)
        log_root = args.log_root
        if log_root is None:
            log_root = os.path.join(
                ENGINEAI_WORKSPACE_ROOT_DIR, "logs", name, args.sub_exp_name
            )
        if args.resume or args.run_exist:
            log_dir, load_run = get_load_run_path(log_root, args.load_run)
            algo_cfg.runner.run_name = load_run
        else:
            # override cfg from args (if specified)
            env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
            # override cfg from args (if specified)
            _, algo_cfg = update_cfg_from_args(None, algo_cfg, args)
            if not hasattr(algo_cfg.runner, "run_name"):
                algo_cfg.runner.run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if log_root is None:
                log_dir = os.path.join(log_root, algo_cfg.runner.run_name)
            else:
                log_dir = os.path.join(log_root, algo_cfg.runner.run_name)

        # check if there is a registered env with that name
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
        if name in self.obs_classes:
            obs_class = self.get_obs_class(name)
        else:
            raise ValueError(f"Obs with name: {name} was not registered")
        if name in self.goal_classes:
            goal_class = self.get_goal_class(name)
        else:
            raise ValueError(f"Goals with name: {name} was not registered")
        if name in self.domain_rand_classes:
            domain_rand_class = self.get_domain_rand_class(name)
        else:
            raise ValueError(f"Domain Rands with name: {name} was not registered")
        if name in self.reward_classes:
            reward_class = self.get_reward_class(name)
        else:
            raise ValueError(f"Rewards with name: {name} was not registered")
        if name in self.runner_classes:
            runner_class = self.get_runner_class(name)
        else:
            raise ValueError(f"Runner with name: {name} was not registered")
        if name in self.algo_classes:
            algo_class = self.get_algo_class(name)
        else:
            raise ValueError(f"Algo with name: {name} was not registered")
        return (
            args,
            task_class,
            obs_class,
            goal_class,
            domain_rand_class,
            reward_class,
            runner_class,
            algo_class,
            log_dir,
            log_root,
            env_cfg,
            algo_cfg,
        )

    def _get_log_dir(self, exp_name: str, exp_id: str) -> str:
        """获取实验日志目录"""
        # 使用 os.path.join 替代字符串拼接
        log_dir = os.path.join(self.log_dir, exp_name, exp_id)
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def _get_checkpoint_dir(self, exp_name: str, exp_id: str) -> str:
        """获取检查点目录"""
        # 使用 os.path.join 替代字符串拼接
        checkpoint_dir = os.path.join(self.log_dir, exp_name, exp_id, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        return checkpoint_dir


# make global task registry
exp_registry = ExpRegistry()
