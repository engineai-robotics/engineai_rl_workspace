from abc import ABC
import inspect


class DataBase(ABC):
    def __init__(self, trajectories, trajectory_frame_durations):
        super().__init__()
        self.trajectories = trajectories
        self.trajectory_dicts = self.get_trajectory_dicts(trajectory_frame_durations)

    def get_trajectory_dicts(self, trajectory_frame_durations):
        component_methods = {
            method.replace("component_", ""): getattr(self, method)
            for method in dir(self)
            if method.startswith("component_") and callable(getattr(self, method))
        }
        trajectory_dicts = []
        for idx, (trajectory_name, trajectory) in enumerate(self.trajectories.items()):
            trajectory_dict = {}
            for component_name, component_method in component_methods.items():
                if "frame_duration" in inspect.signature(component_method).parameters:
                    trajectory_dict[component_name] = component_method(
                        trajectory=trajectory,
                        frame_duration=trajectory_frame_durations[idx],
                    )
                else:
                    trajectory_dict[component_name] = component_method(
                        trajectory=trajectory
                    )
            trajectory_dicts.append(trajectory_dict)

        return trajectory_dicts
