from engineai_rl.algos.ppo.ppo_amp.config_ppo_amp import ConfigPpoAmp


class ConfigA1PpoAmp(ConfigPpoAmp):
    class params(ConfigPpoAmp.params):
        entropy_coef = 0.01

    class runner(ConfigPpoAmp.runner):
        max_iterations = 50000

    class obs(ConfigPpoAmp.input):
        class components(ConfigPpoAmp.input.components):
            class actor(ConfigPpoAmp.input.components.actor):
                obs_list = ["projected_gravity", "dof_pos", "dof_vel", "actions"]

        class obs_noise(ConfigPpoAmp.input.obs_noise):
            class scales(ConfigPpoAmp.input.obs_noise.scales):
                actor = {"projected_gravity": 0.05, "dof_pos": 0.03, "dof_vel": 1.5}
                critic = {
                    "base_lin_vel": 0.1,
                    "base_ang_vel": 0.3,
                    "projected_gravity": 0.05,
                    "dof_pos": 0.03,
                    "dof_vel": 1.5,
                }
