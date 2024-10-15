from engineai_gym.envs.base.domain_rands.domain_rands_base import DomainRandsBase
from isaacgym.torch_utils import *


class DomainRandsTypeRigidShape(DomainRandsBase):
    def init_rand_vec_on_create_env(self):
        num_buckets = 64
        if self.env.cfg.domain_rands.rigid_shape.randomize_friction:
            # prepare friction randomization
            friction_range = self.env.cfg.domain_rands.rigid_shape.friction_range
            bucket_ids = torch.randint(0, num_buckets, (self.env.num_envs, 1))
            friction_buckets = torch_rand_float(
                friction_range[0], friction_range[1], (num_buckets, 1), device="cpu"
            )
            self.friction_coeffs = friction_buckets[bucket_ids]
        if self.env.cfg.domain_rands.rigid_shape.randomize_restitution:
            # prepare restitution randomization
            restitution_range = self.env.cfg.domain_rands.rigid_shape.restitution_range
            bucket_ids = torch.randint(0, num_buckets, (self.env.num_envs, 1))
            friction_buckets = torch_rand_float(
                restitution_range[0],
                restitution_range[1],
                (num_buckets, 1),
                device="cpu",
            )
            self.restitution_coeffs = friction_buckets[bucket_ids]

    def process_on_create_env(self, props, env_id):
        """randomize the rigid shape properties of each environment.
        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id
        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.env.cfg.domain_rands.rigid_shape.randomize_friction:
            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
            self.env.env_frictions[env_id] = self.friction_coeffs[env_id]
        if self.env.cfg.domain_rands.rigid_shape.randomize_restitution:
            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]
        return props
