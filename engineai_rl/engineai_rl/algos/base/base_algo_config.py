from engineai_rl_lib.base_config import BaseConfig


class ConfigAlgoBase(BaseConfig):
    class policy:
        class_name = "ActorCritic"
        init_noise_std = 1.0
        # if True, std is fixed, else it's a trainable parameter
        fixed_std = False

    class params:
        pass

    class runner:
        # number of policy updates
        max_iterations = 1500
        # check for potential saves every this many iterations
        save_interval = 50
        seed = 1
        num_steps_per_env = 24  # per iteration

    class networks:
        # networks used for training
        training = ["actor", "critic"]
        # networks used for inference
        inference = ["actor"]

        class actor:
            # network class
            class_name = "Mlp"
            # args related to network input size.
            # format: {network_arg_names: input_name | int} or {arg_name: list(input_type: input_name | int)}
            # when it's a single value, the size is as it is. when it's a list, the size is all sizes added up
            # when it's input_name, the size is the corresponding input, when it's an int, the size is the int
            input_infos = {"num_input_dim": "actor"}
            output_infos = {"num_output_dim": "action"}
            # other network args
            hidden_dims = [512, 256, 128]
            # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
            activation = "elu"

        class critic:
            # network class
            class_name = "Mlp"
            # args related to network output size.
            # format: {network_arg_names: output_type | int} or {arg_name: list(output_type | int)}
            # when it's a single value, the size is as it is. when it's a list, the size is all sizes added up
            # when it's output_type, it can be "action" (num_actions) and value (1)

            input_infos = {"num_input_dim": "critic"}
            output_infos = {"num_output_dim": "value"}
            hidden_dims = [512, 256, 128]
            # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
            activation = "elu"

    class input:
        obs_clip_threshold = 100.0
        # input used for training
        training = ["actor", "critic"]
        # input used for inference
        inference = ["actor"]

        class components:
            # goal used
            goal_list = ["commands"]

            class actor:
                # obs used
                obs_list = [
                    "base_ang_vel",
                    "projected_gravity",
                    "dof_pos",
                    "dof_vel",
                    "actions",
                ]
                # when obs_with_goals = False and obs_history_length <= 1, the input is obs
                # when obs_with_goals = True and obs_history_length <= 1, the input is torch.cat([obs, goals], dim=-1)
                # when obs_goals_history = True and obs_history_length > 1, the input is torch.cat(history([obs, goals]), dim=-1)
                # when obs_goals_history = False,  obs_history_with_goals = True, and obs_history_length > 1, the input is torch.cat([history(obs), goals], dim=-1)
                # when obs_goals_history = False,  obs_history_with_goals = False, and obs_history_length > 1, the input is history(obs)
                obs_with_goals = False
                obs_history_length = 6
                obs_goals_history = False
                obs_history_with_goals = True
                # retrieve obs before reset for terminal state
                obs_before_reset = False
                # add obs_lag in domain_rands
                lag = True

            class critic:
                obs_list = [
                    "base_lin_vel",
                    "base_ang_vel",
                    "projected_gravity",
                    "dof_pos",
                    "dof_vel",
                    "actions",
                    "height_measurements",
                ]
                # when obs_with_goals = False and obs_history_length <= 1, the input is obs
                # when obs_with_goals = True and obs_history_length <= 1, the input is torch.cat([obs, goals], dim=-1)
                # when obs_goals_history = True and obs_history_length > 1, the input is torch.cat(history([obs, goals]), dim=-1)
                # when obs_goals_history = False,  obs_history_with_goals = True, and obs_history_length > 1, the input is torch.cat([history(obs), goals], dim=-1)
                # when obs_goals_history = False,  obs_history_with_goals = False, and obs_history_length > 1, the input is history(obs)
                obs_with_goals = False
                obs_history_length = 3
                obs_goals_history = False
                obs_history_with_goals = True
                # retrieve obs before reset for terminal state
                obs_before_reset = False
                # add obs_lag in domain_rands
                lag = False

        class obs_noise:
            add_noise = True
            # scales other values
            noise_level = 1.0

            class scales:
                actor = {
                    "base_ang_vel": 0.2,
                    "projected_gravity": 0.05,
                    "dof_pos": 0.01,
                    "dof_vel": 1.5,
                }
