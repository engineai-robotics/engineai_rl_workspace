from engineai_rl_lib.files_and_dirs import (
    get_module_path_from_files_in_dir,
    get_folder_paths_from_dir,
)
import os
from engineai_rl_workspace import ENGINEAI_WORKSPACE_ROOT_DIR
from engineai_gym.envs.base.legged_robot import LeggedRobot
from engineai_gym.envs.base.obs.obs import Obs
from engineai_gym.envs.base.goals.goals import Goals
from engineai_gym.envs.base.domain_rands.domain_rands import DomainRands
from engineai_gym.envs.base.rewards.rewards import Rewards
from engineai_gym.envs.base.config_legged_robot import ConfigLeggedRobot
from engineai_rl.runners.on_policy_runner import OnPolicyRunner
from engineai_rl.algos import Ppo
from engineai_rl.algos import ConfigPpo

file_directory = os.path.dirname(os.path.abspath(__file__))
folders = get_folder_paths_from_dir(file_directory)
for folder in folders:
    import_modules = get_module_path_from_files_in_dir(
        ENGINEAI_WORKSPACE_ROOT_DIR, folder
    )
    for module in import_modules.values():
        exec(f"import {module}")
