import os

ENGINEAI_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
ENGINEAI_GYM_PACKAGE_DIR = os.path.join(ENGINEAI_GYM_ROOT_DIR, "engineai_gym")
ENGINEAI_GYM_ENVS_DIR = os.path.join(ENGINEAI_GYM_PACKAGE_DIR, "envs")
ENGINEAI_GYM_REWARDS_DIR = os.path.join(ENGINEAI_GYM_ENVS_DIR, "base", "rewards")
