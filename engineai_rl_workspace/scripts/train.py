from engineai_rl_workspace import (
    REDIS_HOST,
    LOCK_KEY,
    REDIS_PORT,
    LOCK_TIMEOUT,
    LOCK_MESSAGE,
    PROGRAM_START_MESSAGE,
    INITIALIZATION_COMPLETE_MESSAGE,
)

print(PROGRAM_START_MESSAGE)
import os, asyncio, multiprocessing

from engineai_gym.wrapper import VecGymWrapper, RecordVideoWrapper
from engineai_rl_workspace.utils.helpers import (
    get_classes_and_parents_paths_for_class_and_instance_created_in_class,
)
from engineai_rl_workspace.utils import (
    get_args,
    restore_resume_files,
    restore_original_files,
    generate_resume_cfg_files_from_json,
    get_dict_from_cfg_before_modification,
)
from engineai_rl_workspace.utils.process_resume_files import (
    save_resume_classes_and_parents_files,
    save_resume_files_from_file_paths,
    get_resume_dir,
)
from engineai_rl_workspace.utils.convert_between_py_and_dict import (
    update_cfg_dict_from_args,
)
from engineai_rl_lib.git import store_code_state
from engineai_rl_lib.json import save_json_files
from engineai_rl_lib.redis_lock import RedisLock


async def train(args):
    global lock, original_files
    lock = RedisLock(REDIS_HOST, LOCK_KEY, REDIS_PORT, LOCK_TIMEOUT, LOCK_MESSAGE)
    if not await lock.acquire():
        print("Could not acquire lock, exiting...")
        return
    if args.resume or args.run_exist:
        if args.run_exist:
            process = multiprocessing.Process(
                target=generate_resume_cfg_files_from_json, args=(args,)
            )
            process.start()
            process.join()
            if process.exitcode == 0:
                original_files = restore_resume_files(args)
            else:
                if lock.redis.get(lock.lock_key) == lock.pid.encode():
                    lock.release()
                raise RuntimeError("Fail to generate resume files from json, exit.")
        else:
            original_files = restore_resume_files(args)
    else:
        if lock.redis.get(lock.lock_key) == lock.pid.encode():
            lock.release()
        import engineai_rl_workspace.exps
    from engineai_gym.envs.base.domain_rands.domain_rands_base import DomainRandsBase
    from engineai_gym.envs.base.rewards.rewards_base import RewardsBase
    from engineai_gym.envs.base.rewards.rewards import Rewards
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
    if not args.resume and not args.debug:
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        import engineai_rl_workspace

        store_code_state(log_dir, engineai_rl_workspace.__file__)
        env_cfg_raw_dict = get_dict_from_cfg_before_modification(env_cfg)
        algo_cfg_raw_dict = get_dict_from_cfg_before_modification(algo_cfg)
        cfg = {"env_cfg": env_cfg_raw_dict, "algo_cfg": algo_cfg_raw_dict}
        update_cfg_dict_from_args(cfg, args)
        save_json_files(cfg, log_dir=log_dir, filename="config.json")
        generate_resume_cfg_files_from_json(args, log_dir)
        save_resume_classes_and_parents_files(
            task_class, runner_class, algo_class, obs_class, goal_class, log_dir=log_dir
        )
        domain_rand_resume_files = (
            get_classes_and_parents_paths_for_class_and_instance_created_in_class(
                domain_rand_class,
                prefix="domain_rands_type",
                end_class_instance=DomainRandsBase,
                env=None,
            )
        )
        reward_class_resume_files = (
            get_classes_and_parents_paths_for_class_and_instance_created_in_class(
                reward_class,
                prefix="rewards_type",
                end_class_target_class=Rewards,
                end_class_instance=RewardsBase,
                env=None,
            )
        )
        save_resume_files_from_file_paths(
            domain_rand_resume_files + reward_class_resume_files,
            get_resume_dir(log_dir),
        )
    if args.resume or args.run_exist:
        restore_original_files(original_files)
        lock.release()
    print(INITIALIZATION_COMPLETE_MESSAGE)
    env = exp_registry.make_env(
        task_class,
        obs_class,
        goal_class,
        domain_rand_class,
        reward_class,
        args,
        env_cfg,
    )
    env = VecGymWrapper(env)
    if args.video and not args.debug:
        env = RecordVideoWrapper(
            env,
            frame_size=args.frame_size,
            fps=args.fps,
            record_interval=args.record_interval,
            record_length=args.record_length,
            num_steps_per_env=algo_cfg.runner.num_steps_per_env,
            env_idx=args.env_idx_record,
            actor_idx=args.actor_idx_record,
            rigid_body_idx=args.rigid_body_idx_record,
            camera_offset=args.camera_offset,
            camera_rotation=args.camera_rotation,
            video_path=os.path.join(log_dir, "train_videos"),
        )
    ppo_runner = exp_registry.make_alg_runner(env, args.exp_name, args, log_dir)
    ppo_runner.learn(
        num_learning_iterations=algo_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    global lock, original_files
    try:
        args = get_args()
        asyncio.run(train(args))
    except KeyboardInterrupt or SystemExit:
        try:
            if lock.redis.get(lock.lock_key) == lock.pid.encode():
                restore_original_files(original_files)
                lock.release()
        except:
            pass
