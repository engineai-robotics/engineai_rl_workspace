import torch
from torch import Tensor
import numpy as np
from isaacgym.torch_utils import quat_apply, normalize, get_euler_xyz
from typing import Tuple
from scipy.spatial.transform import Rotation as R


def ang_vel_interpolation(rot_c, rot_n, delta_t):
    rotvec = (R.from_quat(rot_n) * R.from_quat(rot_c).inv()).as_rotvec()
    angle = np.sqrt(np.sum(rotvec**2))
    axis = rotvec / (angle + 1e-8)
    #
    ang_vel = [
        (axis[0] * angle) / delta_t,
        (axis[1] * angle) / delta_t,
        (axis[2] * angle) / delta_t,
    ]
    return ang_vel


@torch.jit.script
def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_xyz(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz


@torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.0
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)


@torch.jit.script
def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


@torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    r = 2 * torch.rand(*shape, device=device) - 1
    r = torch.where(r < 0.0, -torch.sqrt(-r), torch.sqrt(r))
    r = (r + 1.0) / 2.0
    return (upper - lower) * r + lower


@torch.jit.script
def slerp(
    q0, q1, fraction, eps: float = 1e-14, spin: int = 0, shortestpath: bool = True
):
    """Batch quaternion spherical linear interpolation."""

    # Ensure inputs have the same shape and are properly batched
    if q0.dim() != q1.dim():
        raise RuntimeError("q0 and q1 must have the same number of dimensions")

    # Create output tensor with the same shape as inputs
    out = torch.zeros_like(q0)

    # Ensure fraction is properly shaped for broadcasting
    if fraction.dim() == 1:
        fraction = fraction.unsqueeze(-1)

    # Handle edge cases first - reshape masks to match tensor dimensions
    zero_mask = torch.isclose(fraction, torch.zeros_like(fraction))
    ones_mask = torch.isclose(fraction, torch.ones_like(fraction))

    # Ensure masks have the same shape for broadcasting
    while zero_mask.dim() < q0.dim():
        zero_mask = zero_mask.unsqueeze(-1)
    while ones_mask.dim() < q0.dim():
        ones_mask = ones_mask.unsqueeze(-1)

    # Apply masks
    out = torch.where(zero_mask, q0, out)
    out = torch.where(ones_mask, q1, out)

    # Calculate dot product
    d = torch.sum(q0 * q1, dim=-1, keepdim=True)
    d = torch.clamp(d, -1.0, 1.0)

    # Handle very close quaternions
    dist_mask = (torch.abs(torch.abs(d) - 1.0) < eps).squeeze(-1)
    dist_mask = dist_mask.unsqueeze(-1)
    out = torch.where(dist_mask, q0, out)

    if shortestpath:
        d_old = d.clone()
        d = torch.where(d_old < 0, -d, d)
        q1 = torch.where(d_old < 0, -q1, q1)

    angle = torch.arccos(d) + spin * torch.pi

    # Handle small angles with linear interpolation
    small_angle_mask = torch.abs(angle) < eps
    lerp = (1.0 - fraction) * q0 + fraction * q1
    out = torch.where(small_angle_mask, lerp, out)

    # Calculate remaining cases
    valid_mask = ~(zero_mask | ones_mask | dist_mask | small_angle_mask)

    # Safe division with small angle protection
    isin = torch.where(angle > eps, 1.0 / angle, torch.ones_like(angle))

    # Calculate the interpolation for valid cases
    valid_result = (
        torch.sin((1.0 - fraction) * angle) * q0 + torch.sin(fraction * angle) * q1
    ) * isin

    # Apply valid results
    out = torch.where(valid_mask, valid_result, out)

    return out


@torch.jit.script
def interpolate(val0, val1, blend):
    return (1.0 - blend) * val0 + blend * val1
