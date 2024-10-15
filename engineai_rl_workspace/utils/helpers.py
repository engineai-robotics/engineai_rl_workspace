import os
import numpy as np
import random
from isaacgym import gymapi
from isaacgym import gymutil
import inspect
import torch
from abc import ABC
from engineai_rl_lib.type import tuple_type
from engineai_rl_workspace import ENGINEAI_WORKSPACE_PACKAGE_DIR
from engineai_gym import ENGINEAI_GYM_PACKAGE_DIR
from engineai_rl import ENGINEAI_RL_PACKAGE_DIR


def get_original_path_from_resume_path(resume_path, resume_dir):
    if (
        os.path.join(resume_dir, "exps") in resume_path
        or os.path.join(resume_dir, "algos") in resume_path
        or os.path.join(resume_dir, "runners") in resume_path
    ):
        return resume_path.replace(resume_dir, ENGINEAI_RL_PACKAGE_DIR, 1)
    elif os.path.join(resume_dir, "envs") in resume_path:
        return resume_path.replace(resume_dir, ENGINEAI_GYM_PACKAGE_DIR, 1)


def get_resume_path_from_original_path(original_path, resume_dir):
    if ENGINEAI_WORKSPACE_PACKAGE_DIR in original_path:
        return original_path.replace(ENGINEAI_WORKSPACE_PACKAGE_DIR, resume_dir, 1)
    elif ENGINEAI_GYM_PACKAGE_DIR in original_path:
        return original_path.replace(ENGINEAI_GYM_PACKAGE_DIR, resume_dir, 1)
    elif ENGINEAI_RL_PACKAGE_DIR in original_path:
        return original_path.replace(ENGINEAI_RL_PACKAGE_DIR, resume_dir, 1)


def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return


def get_class_and_parents(target_class, end_class=object):
    classes = []
    while True:
        if target_class == end_class or target_class == ABC:
            break
        classes.append(target_class)
        target_class = target_class.__base__
    return classes


def get_class_paths(classes):
    return [inspect.getfile(target_class) for target_class in classes]


def get_class_and_parent_paths(target_class, end_class=object):
    classes = get_class_and_parents(target_class, end_class=end_class)
    return get_class_paths(classes)


def get_attrs_from_instance(instance, prefix, suffix):
    attrs = []
    for attr in dir(instance):
        if attr.startswith(prefix) and attr.endswith(suffix):
            attrs.append(attr)
    return attrs


def get_classes_and_parents_paths_for_instance_created_in_class(
    target_class, prefix="", suffix="", end_class=object, **kwargs
):
    target_instance = target_class(**kwargs)
    attrs = get_attrs_from_instance(target_instance, prefix, suffix)
    items = []
    for attr in attrs:
        items += get_class_and_parent_paths(
            eval(f"type(target_instance.{attr})"), end_class=end_class
        )
    return list(set(items))


def get_classes_and_parents_paths_for_class_and_instance_created_in_class(
    target_class,
    prefix="",
    suffix="",
    end_class_target_class=object,
    end_class_instance=object,
    **kwargs,
):
    target_classes_and_parents = get_class_and_parents(
        target_class, end_class=end_class_target_class
    )
    retrieved_classes = []
    for target_class_and_parent in target_classes_and_parents:
        retrieved_classes += (
            get_classes_and_parents_paths_for_instance_created_in_class(
                target_class_and_parent,
                prefix=prefix,
                suffix=suffix,
                end_class=end_class_instance,
                **kwargs,
            )
        )
    return list(set(retrieved_classes + get_class_paths(target_classes_and_parents)))


def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print(f"Setting seed: {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params


def get_load_run_path(root, load_run=-1):
    if load_run == -1:
        log_dir, load_run = get_last_run_path(root)
    else:
        log_dir = os.path.join(root, load_run)

    return log_dir, load_run


def get_load_checkpoint_path(load_run, checkpoint=-1):
    if checkpoint == -1:
        models = [
            file
            for file in os.listdir(os.path.join(load_run, "checkpoints"))
            if "model" in file
        ]
        models.sort(key=lambda m: f"{m:0>15}")
        model = models[-1]
    else:
        model = f"model_{checkpoint}.pt"

    load_checkpoint = os.path.join(load_run, "checkpoints", model)
    return load_checkpoint


def get_last_run_path(root):
    try:
        runs = os.listdir(root)
        runs.sort()
        if "exported" in runs:
            runs.remove("exported")
        last_run_path = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)

    return last_run_path, runs[-1]


def update_cfg_from_args(env_cfg, algo_cfg, args):
    if env_cfg is not None:
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
    if algo_cfg is not None:
        if args.seed is not None:
            algo_cfg.seed = args.seed
        if args.max_iterations is not None:
            algo_cfg.runner.max_iterations = args.max_iterations
        if args.run_name is not None:
            algo_cfg.runner.run_name = args.run_name
        if args.logger is not None:
            algo_cfg.logger = args.logger

    return env_cfg, algo_cfg


def get_args():
    custom_parameters = [
        {
            "name": "--exp_name",
            "type": str,
            "help": "The experiment name used for training and testing.",
        },
        {
            "name": "--resume",
            "action": "store_true",
            "default": False,
            "help": "Resume training from a checkpoint.",
        },
        {
            "name": "--run_exist",
            "action": "store_true",
            "default": False,
            "help": "Start an existing run from resumes files, but not loading a checkpoint.",
        },
        {
            "name": "--sub_exp_name",
            "type": str,
            "default": "default",
            "help": "Name of the sub-experiment to run or load.",
        },
        {
            "name": "--run_name",
            "type": str,
            "help": "Name of the run. Overrides config file if provided.",
        },
        {
            "name": "--log_root",
            "type": str,
            "help": "Path of the log root. Overrides config file if provided.",
        },
        {
            "name": "--load_run",
            "type": str,
            "default": -1,
            "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided.",
        },
        {
            "name": "--checkpoint",
            "type": int,
            "default": -1,
            "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided.",
        },
        {
            "name": "--headless",
            "action": "store_true",
            "default": False,
            "help": "Force display off at all times",
        },
        {
            "name": "--video",
            "action": "store_true",
            "default": False,
            "help": "Record video during training.",
        },
        {
            "name": "--record_length",
            "type": int,
            "default": 200,
            "help": "The number of steps to record for videos.",
        },
        {
            "name": "--record_interval",
            "type": int,
            "default": 50,
            "help": "The number of iterations before each recording.",
        },
        {
            "name": "--fps",
            "type": int,
            "default": 50,
            "help": "The fps of recorded videos.",
        },
        {
            "name": "--frame_size",
            "type": tuple_type,
            "default": (1280, 720),
            "help": "The size of recorded frame.",
        },
        {
            "name": "--camera_offset",
            "type": tuple_type,
            "default": (0, -2, 0),
            "help": "The offset of the video filming camera.",
        },
        {
            "name": "--camera_rotation",
            "type": tuple_type,
            "default": (0, 0, 90),
            "help": "The rotation of the video filming camera.",
        },
        {
            "name": "--env_idx_record",
            "type": int,
            "default": 0,
            "help": "The env idx to record.",
        },
        {
            "name": "--actor_idx_record",
            "type": int,
            "default": 0,
            "help": "The actor idx to record.",
        },
        {
            "name": "--rigid_body_idx_record",
            "type": int,
            "default": 0,
            "help": "The rigid_body idx to record.",
        },
        {
            "name": "--rl_device",
            "type": str,
            "default": "cuda:0",
            "help": "Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)",
        },
        {
            "name": "--num_envs",
            "type": int,
            "help": "Number of environments to create. Overrides config file if provided.",
        },
        {
            "name": "--seed",
            "type": int,
            "help": "Random seed. Overrides config file if provided.",
        },
        {
            "name": "--max_iterations",
            "type": int,
            "help": "Maximum number of training iterations. Overrides config file if provided.",
        },
        {
            "name": "--logger",
            "type": str,
            "default": None,
            "choices": ["tensorboard", "wandb", "neptune"],
            "help": "Logger module to use.",
        },
        {
            "name": "--upload_model",
            "action": "store_true",
            "default": False,
            help: "upload models to Wandb or Neptune.",
        },
        {
            "name": "--use_joystick",
            "action": "store_true",
            "default": False,
            "help": "Use joystick in play mode",
        },
        {
            "name": "--joystick_scale",
            "type": tuple_type,
            "default": (1.5, 1, 3),
            "help": "Scale of joystick, only useful when use_joystick is True.",
        },
        {
            "name": "--debug",
            "action": "store_true",
            "default": False,
            "help": "In debug mode, no logs will be saved.",
        },
        {
            "name": "--test_length",
            "type": int,
            "default": 500,
            "help": "Number of iteration for each tester.",
        },
    ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy", custom_parameters=custom_parameters
    )

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    # value alignment
    if args.load_run == "-1":
        args.load_run = -1
    return args
