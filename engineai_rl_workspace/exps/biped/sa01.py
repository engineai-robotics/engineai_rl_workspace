from engineai_rl_workspace.utils.exp_registry import exp_registry

from engineai_gym.envs.robots.biped.sa01.config_sa01_rough import ConfigSa01Rough
from engineai_rl.algos.ppo import Ppo
from engineai_rl.exps.biped.sa01.config_sa01_ppo import ConfigSa01Ppo
from engineai_gym.envs.robots.biped.sa01.sa01 import Sa01
from engineai_gym.envs.robots.biped.rewards_biped import RewardsBiped
from engineai_gym.envs.robots.biped.goals_biped import GoalsBiped

exp_registry.register(
    name="sa01_rough_ppo",
    task_class=Sa01,
    goal_class=GoalsBiped,
    reward_class=RewardsBiped,
    env_cfg=ConfigSa01Rough(),
    algo_class=Ppo,
    algo_cfg=ConfigSa01Ppo(),
)
