from engineai_rl_lib.ref_state.data_base import DataBase
from engineai_rl_lib.ref_state import pose3d, ref_state_util
from engineai_rl_lib.math import interpolate, slerp


class Data(DataBase):
    def component_root_pos(self, trajectory):
        return trajectory[:, :3]

    def component_root_z_pos(self, trajectory):
        return trajectory[:, 2:3]

    def component_root_rot(self, trajectory):
        root_rot = trajectory[:, 3:7]
        root_rot = pose3d.quaternion_normalize(root_rot)
        root_rot = ref_state_util.standardize_quaternion(root_rot)
        return root_rot

    def component_dof_pos(self, trajectory):
        return trajectory[:, 7:19]

    def component_foot_pos_local(self, trajectory):
        return trajectory[:, 19:31]

    def component_root_lin_vel(self, trajectory):
        return trajectory[:, 31:34]

    def component_root_ang_vel(self, trajectory):
        return trajectory[:, 34:37]

    def component_dof_vel(self, trajectory):
        return trajectory[:, 37:49]

    def component_foot_vel_local(self, trajectory):
        return trajectory[:, 49:61]

    def blend_root_rot(self, frame_start, frame_end, blend):
        root_rot = slerp(frame_start, frame_end, blend)
        root_rot = ref_state_util.standardize_quaternion(root_rot)
        return root_rot
