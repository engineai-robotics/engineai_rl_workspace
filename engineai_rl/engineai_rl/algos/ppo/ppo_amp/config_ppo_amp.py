from engineai_rl.algos.ppo.config_ppo import ConfigPpo


class ConfigPpoAmp(ConfigPpo):
    class networks(ConfigPpo.networks):
        training = ["actor", "critic", "amp"]
        inference = ["actor"]

        class discriminator:
            class_name = "Mlp"
            input_infos = {"num_input_dim": ["amp"] * 2}
            output_infos = {"num_output_dim": "value"}
            hidden_dims = [1024, 512]
            activation = "lrelu"
            normalizer_class_name = "NormalizerAmp"
            normalizer_args = {"require_grad": False}

    class params(ConfigPpo.params):
        amp_discriminator_name = "AmpDiscriminator"
        amp_reward_coef = 2.0
        amp_task_reward_lerp = 0.3
        min_normalized_std = [0.01, 0.01, 0.01] * 4
        preload_batches = True
        num_preload_batches = 2000000

    class input(ConfigPpo.input):
        training = ["actor", "critic", "amp"]

        class components(ConfigPpo.input.components):
            class actor(ConfigPpo.input.components.actor):
                obs_with_goals = True
                obs_history_with_goals = False
                obs_history_length = 1

            class critic(ConfigPpo.input.components.critic):
                obs_with_goals = True
                obs_history_with_goals = False
                obs_history_length = 1

            class amp:
                obs_list = [
                    "absolute_dof_pos",
                    "foot_pos",
                    "base_lin_vel",
                    "base_ang_vel",
                    "dof_vel",
                    "base_z_pos",
                ]
                obs_with_goals = False
                obs_before_reset = True
                lag = False
