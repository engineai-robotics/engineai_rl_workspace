import torch
import numpy as np
from torch import nn

from isaacgym.torch_utils import *


@torch.jit.script
def quat_to_angle_axis(q):
    # type: (Tensor) -> Tuple[Tensor, Tensor]
    # computes axis-angle representation from quaternion q
    # q must be normalized
    min_theta = 1e-5
    qx, qy, qz, qw = 0, 1, 2, 3

    sin_theta = torch.sqrt(1 - q[..., qw] * q[..., qw])
    angle = 2 * torch.acos(q[..., qw])
    angle = normalize_angle(angle)
    sin_theta_expand = sin_theta.unsqueeze(-1)
    axis = q[..., qx:qw] / sin_theta_expand

    mask = torch.abs(sin_theta) > min_theta
    default_axis = torch.zeros_like(axis)
    default_axis[..., -1] = 1

    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)
    return angle, axis


@torch.jit.script
def angle_axis_to_exp_map(angle, axis):
    # type: (Tensor, Tensor) -> Tensor
    # compute exponential map from axis-angle
    angle_expand = angle.unsqueeze(-1)
    exp_map = angle_expand * axis
    return exp_map


@torch.jit.script
def quat_to_exp_map(q):
    # type: (Tensor) -> Tensor
    # compute exponential map from quaternion
    # q must be normalized
    angle, axis = quat_to_angle_axis(q)
    exp_map = angle_axis_to_exp_map(angle, axis)
    return exp_map


@torch.jit.script
def cosine_angle(A, B):
    # type: (Tensor,Tensor) -> Tensor
    # 确保两个tensor形状相同
    assert A.shape == B.shape, "Tensors must have the same shape"

    # 计算A和B的点积
    dot_product = torch.sum(A * B, dim=1)

    # 计算A和B的模
    magnitude_A = torch.sqrt(torch.sum(A * A, dim=1))
    magnitude_B = torch.sqrt(torch.sum(B * B, dim=1))

    # 计算夹角的余弦
    cosine_angle = dot_product / (magnitude_A * magnitude_B + 1e-6)

    return cosine_angle


@torch.jit.script
def quat_to_tan_norm(q):
    # type: (Tensor) -> Tensor
    # represents a rotation using the tangent and normal vectors
    ref_tan = torch.zeros_like(q[..., 0:3])
    ref_tan[..., 0] = 1
    tan = quat_rotate(q, ref_tan)

    ref_norm = torch.zeros_like(q[..., 0:3])
    ref_norm[..., -1] = 1
    norm = quat_rotate(q, ref_norm)

    norm_tan = torch.cat([tan, norm], dim=len(tan.shape) - 1)
    return norm_tan


@torch.jit.script
def euler_xyz_to_exp_map(roll, pitch, yaw):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    q = quat_from_euler_xyz(roll, pitch, yaw)
    exp_map = quat_to_exp_map(q)
    return exp_map


@torch.jit.script
def exp_map_to_angle_axis(exp_map):
    min_theta = 1e-5

    angle = torch.norm(exp_map, dim=-1)
    angle_exp = torch.unsqueeze(angle, dim=-1)
    axis = exp_map / angle_exp
    angle = normalize_angle(angle)

    default_axis = torch.zeros_like(exp_map)
    default_axis[..., -1] = 1

    mask = torch.abs(angle) > min_theta
    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)

    return angle, axis


@torch.jit.script
def exp_map_to_quat(exp_map):
    angle, axis = exp_map_to_angle_axis(exp_map)
    q = quat_from_angle_axis(angle, axis)
    return q


@torch.jit.script
def slerp(q0, q1, t):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    cos_half_theta = torch.sum(q0 * q1, dim=-1)

    neg_mask = cos_half_theta < 0
    q1 = q1.clone()
    q1[neg_mask] = -q1[neg_mask]
    cos_half_theta = torch.abs(cos_half_theta)
    cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

    half_theta = torch.acos(cos_half_theta)
    sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

    ratioA = torch.sin((1 - t) * half_theta) / sin_half_theta
    ratioB = torch.sin(t * half_theta) / sin_half_theta

    new_q = ratioA * q0 + ratioB * q1

    new_q = torch.where(torch.abs(sin_half_theta) < 0.0001, 0.5 * q0 + 0.5 * q1, new_q)
    new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)

    return new_q


@torch.jit.script
def calc_heading(q):
    # type: (Tensor) -> Tensor
    # calculate heading direction from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    ref_dir = torch.zeros_like(q[..., 0:3])
    ref_dir[..., 0] = 1
    rot_dir = quat_rotate(q, ref_dir)

    heading = torch.atan2(rot_dir[..., 1], rot_dir[..., 0])
    return heading


@torch.jit.script
def calc_heading_quat(q):
    # type: (Tensor) -> Tensor
    # calculate heading rotation from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    heading = calc_heading(q)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1

    heading_q = quat_from_angle_axis(heading, axis)
    return heading_q


@torch.jit.script
def calc_heading_quat_inv(q):
    # type: (Tensor) -> Tensor
    # calculate heading rotation from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    heading = calc_heading(q)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1

    heading_q = quat_from_angle_axis(-heading, axis)
    return heading_q


class scheduler(nn.Module):
    """
    a custom scheduler
    """

    def __init__(self, begin_point, end_point, begin_value, end_value, skip=1):
        super().__init__()
        self.begin_point = begin_point
        self.end_point = end_point
        self.begin_value = begin_value
        self.end_value = end_value
        self.skip = skip
        self.cnt = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.skip_cnt = 1
        self.value = begin_value

    def step(self):
        self.skip_cnt += 1
        if self.skip_cnt == self.skip:
            self.skip_cnt = 0
            self.cnt += 1
            if self.cnt > self.end_point:
                self.value = self.end_value
            else:
                t = self.cnt / (self.end_point - self.begin_point)
                self.value = self.begin_value * (1 - t) + self.end_value * t
