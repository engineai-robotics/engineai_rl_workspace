import copy

import json
import glob
import os.path
from collections import OrderedDict
from engineai_rl_lib.class_operations import get_class_from_file

import torch
import numpy as np

from engineai_rl_lib.math import interpolate


class RefStateLoader:
    def __init__(self, device, time_between_states, motion_files_path, data_mapping):
        """
        time_between_frames: Amount of time in seconds between transition.
        """
        self.device = device
        self.time_between_states = time_between_states
        motion_files, self.trajectory_names = self.get_motion_files(motion_files_path)
        self.data_mapping = data_mapping
        # Values to store for each trajectory.
        self.trajectory_idxes = []
        self.trajectory_lens = []  # Traj length in seconds.
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []
        motion_data_dict = OrderedDict()
        for idx, (motion_file, trajectory_name) in enumerate(
            zip(motion_files, self.trajectory_names)
        ):
            with open(motion_file) as f:
                motion_json = json.load(f)
            motion_data = np.array(motion_json["Frames"])
            self.trajectory_weights.append(float(motion_json["MotionWeight"]))
            frame_duration = float(motion_json["FrameDuration"])
            self.trajectory_frame_durations.append(frame_duration)
            motion_data_dict[trajectory_name] = motion_data
            self.trajectory_idxes.append(idx)
            traj_len = (motion_data.shape[0] - 1) * frame_duration
            self.trajectory_lens.append(traj_len)
            self.trajectory_num_frames.append(float(motion_data.shape[0]))
            print(f"Loaded {traj_len}s. motion from {motion_file}.")

        data_class = get_class_from_file(
            os.path.join(motion_files_path, "data.py"), "Data"
        )
        self.data = data_class(motion_data_dict, self.trajectory_frame_durations)
        self.trajectory_dicts = self.map_to_device(
            self.map_component_names(self.data.trajectory_dicts)
        )
        self.component_sizes = self.get_component_sizes()
        # Trajectory weights are used to sample some trajectories more than others.
        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(
            self.trajectory_weights
        )
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)
        self.preloaded_batches = False

    def preload_batches(self, num_preload_batches):
        # Preload batches.
        print(f"Preloading {num_preload_batches} batches")
        traj_idxes = self.sample_weighted_traj_idx(num_preload_batches)
        times = self.sample_traj_time_batch(traj_idxes)
        self.preloaded_s = self.get_frame_at_times(traj_idxes, times)
        self.preloaded_s_next = self.get_frame_at_times(
            traj_idxes, times + self.time_between_states
        )
        self.num_preload_batches = num_preload_batches
        self.preloaded_batches = True
        print(f"Finished preloading")

    def map_component_names(self, trajectory_dicts):
        trajectory_dicts_copy = copy.deepcopy(trajectory_dicts)
        for trajectory_dict in trajectory_dicts_copy:
            for (
                mapped_component_name,
                original_component_name,
            ) in self.data_mapping.items():
                trajectory_dict[mapped_component_name] = trajectory_dict.pop(
                    original_component_name
                )
        return trajectory_dicts_copy

    def map_to_device(self, trajectory_dicts):
        for trajectory_dict in trajectory_dicts:
            for component_name, component in trajectory_dict.items():
                trajectory_dict[component_name] = torch.tensor(
                    component, dtype=torch.float, device=self.device
                )
        return trajectory_dicts

    def get_motion_files(self, motion_files_path):
        motion_files = [
            file
            for file in glob.glob(os.path.join(motion_files_path, "*"))
            if file.endswith(".json")
        ]
        trajectory_names = []
        for motion_file in motion_files:
            trajectory_names.append(motion_file.rsplit("/", 1)[-1].rsplit(".", 1)[0])
        return motion_files, trajectory_names

    def sample_weighted_traj_idx(self, size=1):
        """Batch sample traj idxes."""
        return np.random.choice(
            self.trajectory_idxes, size=size, p=self.trajectory_weights, replace=True
        )

    def sample_traj_time(self, traj_idx):
        """Sample random time for traj."""
        subst = self.time_between_states + self.trajectory_frame_durations[traj_idx]
        return max(0, (self.trajectory_lens[traj_idx] * np.random.uniform() - subst))

    def sample_traj_time_batch(self, traj_idxes):
        """Sample random time for multiple trajectories."""
        subst = self.time_between_states + self.trajectory_frame_durations[traj_idxes]
        time_samples = (
            self.trajectory_lens[traj_idxes] * np.random.uniform(size=len(traj_idxes))
            - subst
        )
        return np.maximum(np.zeros_like(time_samples), time_samples)

    def get_component_sizes(self):
        component_sizes = {}
        for components in self.trajectory_dicts[0]:
            component_sizes[components] = self.trajectory_dicts[0][components].shape[1:]
        return component_sizes

    def get_frame_at_times(self, traj_idxes, times):
        p = times / self.trajectory_lens[traj_idxes]
        n = self.trajectory_num_frames[traj_idxes]
        idx_low, idx_high = np.floor(p * n).astype(np.int64), np.ceil(p * n).astype(
            np.int64
        )
        start_end_frames = {}
        for component_name, size in self.component_sizes.items():
            start_end_frames[component_name] = torch.zeros(
                (len(traj_idxes),) + size + (2,), dtype=torch.float, device=self.device
            )
        for traj_idx in set(traj_idxes):
            for component_name, start_end_idx in start_end_frames.items():
                trajectory = self.trajectory_dicts[traj_idx][component_name]
                traj_mask = traj_idxes == traj_idx
                start_end_frames[component_name][traj_mask, ..., 0] = trajectory[
                    idx_low[traj_mask]
                ]
                start_end_frames[component_name][traj_mask, ..., 1] = trajectory[
                    idx_high[traj_mask]
                ]

        blend = torch.tensor(
            p * n - idx_low, dtype=torch.float, device=self.device
        ).unsqueeze(-1)

        trajectory_dicts_batch = {}
        for component_name, start_end_frame in start_end_frames.items():
            original_component_name = (
                self.data_mapping[component_name]
                if component_name in self.data_mapping
                else component_name
            )
            if f"blend_{original_component_name}" in dir(self.data) and callable(
                getattr(self.data, f"blend_{original_component_name}")
            ):
                blend_method = getattr(self.data, f"blend_{original_component_name}")
            else:
                blend_method = interpolate
            trajectory_dicts_batch[component_name] = blend_method(
                start_end_frame[..., 0], start_end_frame[..., 1], blend
            )

        return trajectory_dicts_batch

    def retrieve_frame_arrays_from_dict(self, frames_dict, component_names):
        return torch.cat(
            [frames_dict[component_name] for component_name in component_names], dim=-1
        )

    def sample_idxes_and_times(self, num_frames):
        traj_idxes = self.sample_weighted_traj_idx(num_frames)
        times = self.sample_traj_time_batch(traj_idxes)
        return traj_idxes, times

    def get_trajectory_on_idxes(self, trajectory_dicts_batch, traj_idxes):
        partial_trajectory_dicts_batch = {}
        for component_name, component in trajectory_dicts_batch.items():
            partial_trajectory_dicts_batch[component_name] = component[traj_idxes]
        return partial_trajectory_dicts_batch

    def feed_forward_generator(self, num_mini_batch, mini_batch_size, component_names):
        """Generates a batch of AMP transitions."""
        for _ in range(num_mini_batch):
            if self.preloaded_batches:
                idxes = np.random.choice(self.num_preload_batches, size=mini_batch_size)
                s, s_next = self.get_trajectory_on_idxes(
                    self.preloaded_s, idxes
                ), self.get_trajectory_on_idxes(self.preloaded_s_next, idxes)
            else:
                traj_idxes, times = self.sample_idxes_and_times(mini_batch_size)
                s, s_next = self.get_frame_at_times(
                    traj_idxes, times
                ), self.get_frame_at_times(traj_idxes, times + self.time_between_states)
            yield self.retrieve_frame_arrays_from_dict(
                s, component_names
            ), self.retrieve_frame_arrays_from_dict(s_next, component_names)

    @property
    def num_motions(self):
        return len(self.trajectory_names)
