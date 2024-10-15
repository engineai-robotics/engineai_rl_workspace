import os
from isaacgym import gymapi
import moviepy.editor as mpy
import numpy as np
from engineai_rl_lib.class_operations import (
    add_instance_properties_and_methods_to_class,
)


class RecordVideoWrapper:
    def __init__(self, env, manual=False, **kwargs):
        self._env = env  # Store the wrapped environment
        self._manual = manual

        add_instance_properties_and_methods_to_class(env, self)
        self._is_recording_video = False
        self._frames_buf = None
        self.fps = None
        self._record_cfg = kwargs
        self._step_idx = 0
        self._set_camera_video_props()
        if (
            self._record_cfg["record_interval"] * self._record_cfg["num_steps_per_env"]
            < self._record_cfg["record_length"]
        ):
            raise ValueError("Record interval must be larger than frame size")

    def step(self, actions):
        if (
            self._step_idx
            % (
                self._record_cfg["record_interval"]
                * self._record_cfg["num_steps_per_env"]
            )
            == 0
            and not self._manual
        ):
            self.start_recording_video()
        obs_dict, goal_dict, rew_buf, reset_buf, extras = self._env.step(actions)
        if self._is_recording_video:
            self._frames_buf.append(self._get_camera_image())
            if len(self._frames_buf) == self._record_cfg["record_length"]:
                file_name = (
                    str(
                        int(
                            (self._step_idx - len(self._frames_buf) + 1)
                            / self._record_cfg["num_steps_per_env"]
                        )
                    )
                    + ".mp4"
                )
                self.end_and_save_recording_video(file_name)
        self._step_idx += 1
        return obs_dict, goal_dict, rew_buf, reset_buf, extras

    def reset(self):
        return self._env.reset()

    def _set_camera_video_props(self):
        self._env_handle = self._env.envs[self.record_cfg["env_idx"]]
        self._camera_properties = gymapi.CameraProperties()
        self._camera_properties.width, self._camera_properties.height = self.record_cfg[
            "frame_size"
        ]
        self._camera_handle = self._env.gym.create_camera_sensor(
            self._env_handle, self._camera_properties
        )
        camera_offset = gymapi.Vec3(*self.record_cfg["camera_offset"])
        camera_rotation = gymapi.Quat.from_euler_zyx(
            *np.deg2rad(self.record_cfg["camera_rotation"])
        )
        self._actor_handle = self._env.gym.get_actor_handle(
            self._env_handle, self.record_cfg["actor_idx"]
        )
        self._body_handle = self._env.gym.get_actor_rigid_body_handle(
            self._env_handle, self._actor_handle, self.record_cfg["rigid_body_idx"]
        )
        self._env.gym.attach_camera_to_body(
            self._camera_handle,
            self._env_handle,
            self._body_handle,
            gymapi.Transform(camera_offset, camera_rotation),
            gymapi.FOLLOW_POSITION,
        )
        self.fps = self.record_cfg["fps"]
        self._camera_video_props_are_set = True

    def get_observations(self):
        return self._env.get_env_info()

    def _get_camera_image(self):
        self._env.gym.step_graphics(self.sim)
        self._env.gym.render_all_camera_sensors(self.sim)
        image = self._env.gym.get_camera_image(
            self.sim, self._env_handle, self._camera_handle, gymapi.IMAGE_COLOR
        )
        image = image.reshape(
            (self._camera_properties.height, self._camera_properties.width, 4)
        )
        return image

    @property
    def is_recording_video(self):
        return self._is_recording_video

    def start_recording_video(self):
        if self._is_recording_video:
            raise RuntimeError(
                "Videos are already recording! It must be ended before starting recording again."
            )
        else:
            self._is_recording_video = True
            if self._frames_buf is None:
                self._frames_buf = []

    def end_and_save_recording_video(self, filename):
        if not os.path.isdir(self._record_cfg["video_path"]):
            os.makedirs(self._record_cfg["video_path"], exist_ok=True)
        clip = mpy.ImageSequenceClip(self._frames_buf, fps=self.fps)
        self._frames_buf = []
        self._is_recording_video = False
        clip.write_videofile(
            os.path.join(self._record_cfg["video_path"], filename), codec="libx264"
        )

    @property
    def manual(self):
        return self._manual

    @property
    def record_length(self):
        return self._record_cfg["record_length"]

    @property
    def frames_buf(self):
        return self._frames_buf

    @property
    def record_cfg(self):
        return self._record_cfg
