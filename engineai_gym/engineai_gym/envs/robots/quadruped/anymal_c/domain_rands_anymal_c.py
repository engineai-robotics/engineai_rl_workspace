from engineai_gym.envs.base.domain_rands.domain_rands import DomainRands
from .domain_rands_type_dof_anymal_c import DomainRandsTypeDofAnymalC


class DomainRandsAnymalC(DomainRands):
    def __init__(self, env):
        super().__init__(env)
        self.domain_rands_type_dof = DomainRandsTypeDofAnymalC(env)
