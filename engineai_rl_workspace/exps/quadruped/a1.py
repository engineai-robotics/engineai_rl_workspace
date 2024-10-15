from engineai_rl_workspace.utils.exp_registry import exp_registry

from engineai_gym.envs.base.legged_robot_ref import LeggedRobotRef
from engineai_rl.runners.on_policy_runner import OnPolicyRunner
from engineai_gym.envs.robots.quadruped.a1.rough.config_a1_rough import ConfigA1Rough
from engineai_rl.algos.ppo import Ppo
from engineai_rl.algos.ppo.ppo_amp.ppo_amp import PpoAmp
from engineai_rl.exps.quadruped.a1.config_a1_ppo import ConfigA1Ppo
from engineai_gym.envs.robots.quadruped.a1.flat.config_a1_flat import ConfigA1Flat
from engineai_gym.envs.robots.quadruped.a1.flat.config_a1_flat_ref_state import (
    ConfigA1FlatRefState,
)
from engineai_rl.exps.quadruped.a1.config_a1_ppo_amp import ConfigA1PpoAmp
from engineai_gym.envs.base.legged_robot import LeggedRobot
from engineai_gym.envs.base.rewards.rewards import Rewards

exp_registry.register(
    name="a1_rough_ppo", env_cfg=ConfigA1Rough(), algo_class=Ppo, algo_cfg=ConfigA1Ppo()
)
exp_registry.register(
    name="a1_flat_ppo", env_cfg=ConfigA1Flat(), algo_class=Ppo, algo_cfg=ConfigA1Ppo()
)
exp_registry.register(
    name="a1_flat_ppo_amp",
    task_class=LeggedRobotRef,
    env_cfg=ConfigA1FlatRefState(),
    algo_class=PpoAmp,
    algo_cfg=ConfigA1PpoAmp(),
)
