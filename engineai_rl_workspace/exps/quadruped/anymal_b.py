from engineai_rl_workspace.utils.exp_registry import exp_registry

from engineai_rl.runners.on_policy_runner import OnPolicyRunner
from engineai_gym.envs.robots.quadruped.anymal_b.config_anymal_b_rough import (
    ConfigAnymalBRough,
)
from engineai_rl.algos.ppo import Ppo
from engineai_rl.exps.quadruped.anymal_b.config_anymal_b_ppo import AnymalBRoughPpoCfg
from engineai_gym.envs.robots.quadruped.anymal_c.anymal import Anymal
from engineai_gym.envs.base.rewards.rewards import Rewards

exp_registry.register(
    name="anymal_b_rough_ppo",
    task_class=Anymal,
    env_cfg=ConfigAnymalBRough(),
    algo_class=Ppo,
    algo_cfg=AnymalBRoughPpoCfg(),
)
