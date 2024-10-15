import os, asyncio
import pygame
from threading import Thread
import numpy as np
from tqdm import tqdm

from engineai_gym import ENGINEAI_GYM_PACKAGE_DIR
from engineai_gym.tester.tester import Tester
from engineai_gym.wrapper import VecGymWrapper, RecordVideoWrapper
from engineai_rl_workspace.utils import (
    get_args,
    restore_resume_files,
    restore_original_files,
)
from engineai_rl_workspace import (
    REDIS_HOST,
    LOCK_KEY,
    REDIS_PORT,
    LOCK_TIMEOUT,
    LOCK_MESSAGE,
    INITIALIZATION_COMPLETE_MESSAGE,
)
from engineai_rl_lib.redis_lock import RedisLock


async def play(args):
    global lock, original_files
    lock = RedisLock(REDIS_HOST, LOCK_KEY, REDIS_PORT, LOCK_TIMEOUT, LOCK_MESSAGE)
    global x_vel_cmd, y_vel_cmd, yaw_vel_cmd, last_x_vel_cmd, last_y_vel_cmd, last_yaw_vel_cmd
    (
        x_vel_cmd,
        y_vel_cmd,
        yaw_vel_cmd,
        last_x_vel_cmd,
        last_y_vel_cmd,
        last_yaw_vel_cmd,
    ) = (0, 0, 0, 0, 0, 0)
    args.resume = True
    if not await lock.acquire():
        print("Could not acquire lock, exiting...")
        return
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
    # override some parameters for testing
    if args.use_joystick or args.headless:
        env_cfg.env.num_envs = 1
    else:
        env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    algo_cfg.input.obs_noise.add_noise = False
    env_cfg.domain_rands.randomize_friction = False
    env_cfg.domain_rands.push_robots = False

    restore_original_files(original_files)
    if lock.redis.get(lock.lock_key) == lock.pid.encode():
        lock.release()
    print(INITIALIZATION_COMPLETE_MESSAGE)

    # prepare environment
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
    if args.video:
        env = RecordVideoWrapper(
            env,
            manual=True,
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
            video_path=os.path.join(log_dir, "test", "videos"),
        )

    # load policy
    runner = exp_registry.make_alg_runner(env, args.exp_name, args, log_dir)
    policy = runner.get_inference_policy()
    tester = Tester(
        env,
        args.test_length,
        env.dt,
        os.path.join(log_dir, "test"),
        env_cfg.env.tester_config_path.format(
            ENGINEAI_GYM_PACKAGE_DIR=ENGINEAI_GYM_PACKAGE_DIR
        ),
        args.video,
        extra_args={"robot_index": args.env_idx_record},
    )
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1.0, 1.0, 0.0])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    if args.use_joystick:
        iteration_range = range(100000)
    else:
        iteration_range = tqdm(range(tester.num_testers * args.test_length))
    if args.use_joystick:
        inputs = runner.reset(
            set_commands_from_joystick,
            set_goals_callback_args=(env, x_vel_cmd, y_vel_cmd, yaw_vel_cmd),
        )
    else:
        inputs = runner.reset(tester.set_env, set_goals_callback_args=(0,))
    for iter in iteration_range:
        if args.use_joystick:
            inputs, actions, _, _, _ = runner.step(
                inputs,
                policy,
                set_commands_from_joystick,
                set_goals_callback_args=(env, x_vel_cmd, y_vel_cmd, yaw_vel_cmd),
            )
        else:
            if iter + 1 < tester.num_testers * args.test_length:
                inputs, actions, _, _, _ = runner.step(
                    inputs, policy, tester.set_env, set_goals_callback_args=(iter + 1,)
                )
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)
        if not args.use_joystick:
            tester.step(iter, {"actions": actions})


def set_commands_from_joystick(env, x_vel_cmd, y_vel_cmd, yaw_vel_cmd):
    global last_x_vel_cmd, last_y_vel_cmd, last_yaw_vel_cmd
    env.commands[:, 0] = x_vel_cmd
    env.commands[:, 1] = y_vel_cmd
    env.commands[:, 2] = yaw_vel_cmd
    if (
        last_x_vel_cmd != x_vel_cmd
        or last_y_vel_cmd != y_vel_cmd
        or last_yaw_vel_cmd != yaw_vel_cmd
    ):
        print("Current command: ", env.commands[:, :3])
        last_x_vel_cmd = x_vel_cmd
        last_y_vel_cmd = y_vel_cmd
        last_yaw_vel_cmd = yaw_vel_cmd
    return env.goals


def use_joystick(args):
    joystick_opened = False
    pygame.init()
    try:
        # get joystick
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        joystick_opened = True
    except Exception as e:
        print(f"Unable to turn on joystick：{e}")

    # handle joystick thread
    def handle_joystick_input():
        global x_vel_cmd, y_vel_cmd, yaw_vel_cmd, last_x_vel_cmd, last_y_vel_cmd, last_yaw_vel_cmd
        while True:
            # get joystick input
            pygame.event.get()

            # update command
            x_vel_cmd = -joystick.get_axis(1) * args.joystick_scale[0]
            y_vel_cmd = -joystick.get_axis(0) * args.joystick_scale[1]
            yaw_vel_cmd = -joystick.get_axis(3) * args.joystick_scale[2]

            # wait for a short period of time
            pygame.time.delay(100)

    # start thread
    if joystick_opened:
        joystick_thread = Thread(target=handle_joystick_input)
        joystick_thread.start()


if __name__ == "__main__":
    global lock, original_files
    try:
        MOVE_CAMERA = False
        args = get_args()
        if args.use_joystick:
            use_joystick(args)
        asyncio.run(play(args))
    except KeyboardInterrupt or SystemExit:
        try:
            if lock.redis.get(lock.lock_key) == lock.pid.encode():
                restore_original_files(original_files)
                lock.release()
        except:
            pass
