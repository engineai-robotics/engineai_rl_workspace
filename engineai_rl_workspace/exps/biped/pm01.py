from engineai_rl_workspace.utils.exp_registry import exp_registry

from engineai_gym.envs.robots.biped.pm01.pm01 import Pm01

from engineai_gym.envs.robots.biped.pm01.rough.config_pm01_rough import (
    ConfigPm01Rough,
)
from engineai_gym.envs.robots.biped.pm01.flat.config_pm01_flat import (
    ConfigPm01Flat,
)
from engineai_gym.envs.robots.biped.goals_biped import GoalsBiped
from engineai_rl.algos.ppo import Ppo
from engineai_rl.exps.biped.pm01.config_pm01_ppo import ConfigPm01Ppo
from engineai_gym.envs.robots.biped.rewards_biped import RewardsBiped


exp_registry.register(
    name="pm01_rough_ppo",
    task_class=Pm01,
    goal_class=GoalsBiped,
    reward_class=RewardsBiped,
    env_cfg=ConfigPm01Rough(),
    algo_class=Ppo,
    algo_cfg=ConfigPm01Ppo(),
)
exp_registry.register(
    name="pm01_flat_ppo",
    task_class=Pm01,
    goal_class=GoalsBiped,
    reward_class=RewardsBiped,
    env_cfg=ConfigPm01Flat(),
    algo_class=Ppo,
    algo_cfg=ConfigPm01Ppo(),
)
