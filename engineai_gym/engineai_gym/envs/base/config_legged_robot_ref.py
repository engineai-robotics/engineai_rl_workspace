from .config_legged_robot import ConfigLeggedRobot


class ConfigLeggedRobotRef(ConfigLeggedRobot):
    class ref_state:
        ref_state_loader = False
        ref_state_init = False
        motion_files_path = ""
        ref_state_init_prob = 0.0
        preload_transitions = False
        num_preload_transitions = 1000000
        data_mapping = None
