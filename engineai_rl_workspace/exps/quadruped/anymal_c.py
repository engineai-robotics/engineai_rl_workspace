from engineai_rl_workspace.utils.exp_registry import exp_registry
from engineai_rl.runners.on_policy_runner import OnPolicyRunner
from engineai_gym.envs.robots.quadruped.anymal_c.domain_rands_anymal_c import (
    DomainRandsAnymalC,
)
from engineai_gym.envs.robots.quadruped.anymal_c.rough.config_anymal_c_rough import (
    ConfigAnymalCRough,
)
from engineai_rl.exps.quadruped.anymal_c.rough.config_anymal_c_rough_ppo import (
    ConfigAnymalCRoughPpo,
)
from engineai_gym.envs.robots.quadruped.anymal_c.rough.rewards_anymal_c_rough import (
    RewardsRoughAnymalC,
)
from engineai_gym.envs.robots.quadruped.anymal_c.flat.config_anymal_c_flat import (
    ConfigAnymalCFlat,
)
from engineai_rl.exps.quadruped.anymal_c.flat.config_anymal_c_flat_ppo import (
    ConfigAnymalCFlatPpo,
)
from engineai_gym.envs.robots.quadruped.anymal_c.anymal import Anymal
from engineai_rl.algos.ppo import Ppo

exp_registry.register(
    name="anymal_c_flat_ppo",
    task_class=Anymal,
    domain_rand_class=DomainRandsAnymalC,
    env_cfg=ConfigAnymalCFlat(),
    algo_class=Ppo,
    algo_cfg=ConfigAnymalCFlatPpo(),
)
exp_registry.register(
    name="anymal_c_rough_ppo",
    task_class=Anymal,
    domain_rand_class=DomainRandsAnymalC,
    reward_class=RewardsRoughAnymalC,
    env_cfg=ConfigAnymalCRough(),
    runner_class=OnPolicyRunner,
    algo_class=Ppo,
    algo_cfg=ConfigAnymalCRoughPpo(),
)
