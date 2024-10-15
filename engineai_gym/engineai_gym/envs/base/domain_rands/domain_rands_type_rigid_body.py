from engineai_gym.envs.base.domain_rands.domain_rands_base import DomainRandsBase
from isaacgym.torch_utils import *
from isaacgym import gymapi


class DomainRandsTypeRigidBody(DomainRandsBase):
    def init_rand_vec_on_create_env(self):
        """Randomise some of the rigid body properties of the actor in the given environments, i.e.
        sample the mass, centre of mass position, friction and restitution."""
        if self.env.cfg.domain_rands.rigid_body.randomize_base_mass:
            (
                min_payload,
                max_payload,
            ) = self.env.cfg.domain_rands.rigid_body.added_mass_range

            self.added_payload_masses = torch_rand_float(
                min_payload, max_payload, (self.env.num_envs, 1), device=self.env.device
            )

        if self.env.cfg.domain_rands.rigid_body.randomize_com:
            self.com_displacements = torch.zeros(
                (self.env.num_envs, 3),
                device=self.env.device,
            )
            if isinstance(
                self.env.cfg.domain_rands.rigid_body.com_displacement_range, list
            ) and all(
                isinstance(item, list)
                for item in self.env.cfg.domain_rands.rigid_body.com_displacement_range
            ):
                for i in range(3):
                    (
                        min_com_displacement,
                        max_com_displacement,
                    ) = self.env.cfg.domain_rands.rigid_body.com_displacement_range[i]
                    self.com_displacements[:, i] = torch_rand_float(
                        min_com_displacement,
                        max_com_displacement,
                        (self.env.num_envs, 1),
                        device=self.env.device,
                    ).squeeze(-1)
            elif (
                isinstance(
                    self.env.cfg.domain_rands.rigid_body.com_displacement_range, list
                )
                and len(self.env.cfg.domain_rands.rigid_body.com_displacement_range)
                == 2
            ):
                (
                    min_link_mass,
                    max_link_mass,
                ) = self.env.cfg.domain_rands.rigid_body.link_mass_multi_range

                self.link_masses_multi = torch_rand_float(
                    min_link_mass,
                    max_link_mass,
                    (self.env.num_envs, self.env.num_bodies - 1),
                    device=self.env.device,
                )
            else:
                raise ValueError("Format of com_displacement_range is not correct")

        if self.env.cfg.domain_rands.rigid_body.randomize_com:
            (
                min_com_displacement,
                max_com_displacement,
            ) = self.env.cfg.domain_rands.rigid_body.com_displacement_range
            self.com_displacements = torch_rand_float(
                min_com_displacement,
                max_com_displacement,
                (self.env.num_envs, 3),
                device=self.env.device,
            )

    def process_on_create_env(self, props, env_id):
        # randomize base mass
        if self.env.cfg.domain_rands.rigid_body.randomize_base_mass:
            props[0].mass += self.added_payload_masses[env_id]

        if self.env.cfg.domain_rands.rigid_body.randomize_com:
            props[0].com += gymapi.Vec3(
                self.com_displacements[env_id, 0],
                self.com_displacements[env_id, 1],
                self.com_displacements[env_id, 2],
            )

        if self.env.cfg.domain_rands.rigid_body.randomize_link_mass:
            for i in range(1, len(props)):
                props[i].mass *= self.link_masses_multi[env_id, i - 1]

        return props
