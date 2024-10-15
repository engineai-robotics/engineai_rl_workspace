import torch
from torch import Tensor
import numpy as np
from isaacgym.torch_utils import quat_apply, normalize
from typing import Tuple


# @ torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.0
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)


# @ torch.jit.script
def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


# @ torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    r = 2 * torch.rand(*shape, device=device) - 1
    r = torch.where(r < 0.0, -torch.sqrt(-r), torch.sqrt(r))
    r = (r + 1.0) / 2.0
    return (upper - lower) * r + lower


# @ torch.jit.script
def quintic_interpolation(T, h, t):
    T3 = T * T * T
    T4 = T3 * T
    T5 = T4 * T
    T6 = T5 * T

    a0 = 0
    a1 = 0
    a2 = 0
    a3 = 64 * h / T3
    a4 = -192 * h / T4
    a5 = 192 * h / T5
    a6 = -64 * h / T6

    t3 = t * t * t
    t4 = t3 * t
    t5 = t4 * t
    t6 = t5 * t
    res = a3 * t3 + a4 * t4 + a5 * t5 + a6 * t6

    return res
