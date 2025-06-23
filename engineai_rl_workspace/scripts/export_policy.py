import os, asyncio, json

from engineai_rl_workspace.utils import (
    get_args,
    get_load_run_path,
    get_load_checkpoint_path,
    restore_resume_files,
    restore_original_files,
    convert_nn_to_onnx,
    convert_onnx_to_mnn,
)
from engineai_rl_workspace import (
    REDIS_HOST,
    LOCK_KEY,
    REDIS_PORT,
    LOCK_TIMEOUT,
    LOCK_MESSAGE,
    INITIALIZATION_COMPLETE_MESSAGE,
)
from engineai_rl_lib.class_operations import class_to_dict
from engineai_rl_lib.redis_lock import RedisLock
from engineai_rl.modules.networks import *

import torch


async def export_policy(args):
    global lock, original_files
    lock = RedisLock(REDIS_HOST, LOCK_KEY, REDIS_PORT, LOCK_TIMEOUT, LOCK_MESSAGE)
    if not await lock.acquire():
        print("Could not acquire lock, exiting...")
        return
    args.resume = True
    original_files = restore_resume_files(args)
    from engineai_rl_workspace.utils.exp_registry import exp_registry

    (
        args,
        task_class,
        obs_class,
        goal_class,
        domain_rand_class,
        reward_class,
        runner_class,
        algo_class,
        log_dir,
        log_root,
        env_cfg,
        algo_cfg,
    ) = exp_registry.get_class_and_cfg(name=args.exp_name, args=args)
    restore_original_files(original_files)
    if lock.redis.get(lock.lock_key) == lock.pid.encode():
        lock.release()
    print(INITIALIZATION_COMPLETE_MESSAGE)
    log_dir, load_run = get_load_run_path(log_root, load_run=args.load_run)
    load_checkpoint = get_load_checkpoint_path(
        load_run=log_dir, checkpoint=args.checkpoint
    )
    path = os.path.join(log_dir, "policies")
    loaded_dict = torch.load(load_checkpoint, map_location=torch.device(args.rl_device))
    inference_network_names = algo_cfg.networks.inference
    input_sizes = loaded_dict["infos"]["input_sizes"]
    for inference_network_name in inference_network_names:
        inference_network_cfg = class_to_dict(
            eval(f"algo_cfg.networks.{inference_network_name}")
        )
        inference_network_class_name = inference_network_cfg.pop("class_name")
        inference_network_class = eval(inference_network_class_name)
        network_input_infos = inference_network_cfg.pop("input_infos")
        input_dim_infos = {}
        for network_input_name, network_input_type in network_input_infos.items():
            if isinstance(network_input_type, list):
                input_dim_infos[network_input_name] = 0
                for network_input_subtype in network_input_type:
                    if isinstance(network_input_subtype, int):
                        input_dim_infos[network_input_name] += network_input_subtype
                    elif network_input_subtype in input_sizes:
                        input_dim_infos[network_input_name] += input_sizes[
                            network_input_subtype
                        ]
                    else:
                        raise ValueError(
                            f"Network input type {network_input_subtype} not supported"
                        )
            else:
                if isinstance(network_input_type, int):
                    input_dim_infos[network_input_name] = network_input_type
                elif network_input_type in input_sizes:
                    input_dim_infos[network_input_name] = input_sizes[
                        network_input_type
                    ]
                else:
                    raise ValueError(
                        f"Network input type {network_input_type} not supported"
                    )
        network_output_infos = inference_network_cfg.pop("output_infos")
        output_dim_infos = {}
        for (
            network_output_name,
            network_output_type,
        ) in network_output_infos.items():
            if isinstance(network_output_type, list):
                input_dim_infos[network_output_name] = 0
                for network_output_subtype in network_output_type:
                    if network_output_subtype == "action":
                        output_dim_infos[network_output_name] += len(
                            env_cfg.env.action_joints
                        )
                    elif network_output_subtype == "value":
                        output_dim_infos[network_output_name] += 1
                    elif isinstance(network_output_subtype, int):
                        output_dim_infos[network_output_name] += network_output_subtype
                    else:
                        raise ValueError(
                            f"Network output type {network_output_subtype} not supported"
                        )
            else:
                if network_output_type == "action":
                    output_dim_infos[network_output_name] = len(
                        env_cfg.env.action_joints
                    )
                elif network_output_type == "value":
                    output_dim_infos[network_output_name] = 1
                elif isinstance(network_output_type, int):
                    output_dim_infos[network_output_name] = network_output_type
                else:
                    raise ValueError(
                        f"Network output type {network_output_type} not supported"
                    )
        if inference_network_cfg.get("normalizer_class_name", False):
            normalizer_class = eval(inference_network_cfg.pop("normalizer_class_name"))
            normalizer = normalizer_class(
                **input_dim_infos,
                **inference_network_cfg.pop("normalizer_args"),
            )
        else:
            normalizer = None
        inference_network = inference_network_class(
            **input_dim_infos,
            **output_dim_infos,
            **inference_network_cfg,
            normalizer=normalizer,
        ).to("cpu")
        inference_network.load_state_dict(
            loaded_dict["model_state_dict"][inference_network_name]
        )
        convert_nn_to_onnx(
            inference_network,
            path,
            args.exp_name + "_" + args.load_run + "_" + inference_network_class_name,
            input_dim_infos,
            output_dim_infos,
        )

        convert_onnx_to_mnn(
            os.path.join(
                path,
                args.exp_name
                + "_"
                + args.load_run
                + "_"
                + inference_network_class_name
                + ".onnx",
            ),
            os.path.join(
                path,
                args.exp_name
                + "_"
                + args.load_run
                + "_"
                + inference_network_class_name
                + ".mnn",
            ),
        )

    # 保存配置
    config = {
        "exp_name": args.exp_name,
        "exp_id": args.load_run,
        "algo_name": algo_class,
        "checkpoint": args.checkpoint,
        "device": args.rl_device,
    }
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    print("配置保存成功")


if __name__ == "__main__":
    global lock, original_files
    try:
        args = get_args()
        asyncio.run(export_policy(args))
    except KeyboardInterrupt or SystemExit:
        try:
            if lock.redis.get(lock.lock_key) == lock.pid.encode():
                restore_original_files(original_files)
                lock.release()
        except:
            pass
