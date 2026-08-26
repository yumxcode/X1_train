# Copyright (c) 2024, AgiBot Inc. All rights reserved.
# Isaac Sim / IsaacLab port of the X1 training pipeline.
# Pure-torch replacements for the isaacgym.torch_utils helpers used by the
# original Isaac Gym pipeline (no isaacgym dependency).

import numpy as np
import torch


def torch_rand_float(low, high, size, device):
    """Uniform random tensor in [low, high). Mirrors isaacgym.torch_utils.torch_rand_float."""
    return (high - low) * torch.rand(size, device=device) + low


def normalize(x, eps: float = 1e-9):
    """Normalize a batch of vectors (any trailing dims)."""
    return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)


def quat_rotate(q, v):
    """Rotate vector v by quaternion q (w, x, y, z convention of Isaac).

    q: (..., 4), v: (..., 3)
    """
    q_w = q[..., 0]
    q_vec = q[..., 1:]
    a = v * (2.0 * q_w * q_w - 1.0).unsqueeze(-1)
    b = torch.linalg.cross(q_vec, v, dim=-1) * (2.0 * q_w).unsqueeze(-1)
    c = q_vec * (q_vec * v).sum(dim=-1, keepdim=True) * 2.0
    return a + b + c


def quat_rotate_inverse(q, v):
    """Rotate vector v by the inverse of quaternion q."""
    q_inv = q.clone()
    q_inv[..., 1:] = -q_inv[..., 1:]
    return quat_rotate(q_inv, v)


# alias matching isaacgym.torch_utils naming
quat_apply = quat_rotate
quat_apply_inverse = quat_rotate_inverse


def get_axis_params(axis_idx: int = 2, value: float = -1.0) -> list:
    """isaacgym.torch_utils.get_axis_params equivalent (for gravity vector)."""
    zs = [0.0] * 3
    zs[axis_idx] = value
    return zs


def get_euler_rpy(q):
    """Quaternion (x, y, z, w) -> roll/pitch/yaw, each mapped to [0, 2*pi).

    Kept identical to the original implementation (isaacgym quat layout xyzw).
    """
    qx, qy, qz, qw = 0, 1, 2, 3
    sinr_cosp = 2.0 * (q[..., qw] * q[..., qx] + q[..., qy] * q[..., qz])
    cosr_cosp = q[..., qw] * q[..., qw] - q[..., qx] * q[..., qx] - q[..., qy] * q[..., qy] + q[..., qz] * q[..., qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (q[..., qw] * q[..., qy] - q[..., qz] * q[..., qx])
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        torch.sign(sinp) * (np.pi / 2.0),
        torch.asin(torch.clamp(sinp, -1.0, 1.0)),
    )

    siny_cosp = 2.0 * (q[..., qw] * q[..., qz] + q[..., qx] * q[..., qy])
    cosy_cosp = q[..., qw] * q[..., qw] + q[..., qx] * q[..., qx] - q[..., qy] * q[..., qy] - q[..., qz] * q[..., qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2 * np.pi), pitch % (2 * np.pi), yaw % (2 * np.pi)


def get_euler_xyz_tensor(quat):
    """Quaternion (x, y, z, w) -> euler (roll, pitch, yaw) stacked on last dim, wrapped to [-pi, pi]."""
    r, p, w = get_euler_rpy(quat)
    euler_xyz = torch.stack((r, p, w), dim=-1)
    euler_xyz = torch.where(euler_xyz > np.pi, euler_xyz - 2 * np.pi, euler_xyz)
    return euler_xyz


def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-pi, pi)."""
    return (angles + torch.pi) % (2 * torch.pi) - torch.pi


def quat_apply_yaw(quat, vec):
    """Rotate vec by quaternion's yaw component only."""
    quat_yaw = quat.clone()
    quat_yaw[:, 1] = 0.0
    quat_yaw[:, 2] = 0.0
    quat_yaw = normalize(quat_yaw)
    return quat_rotate(quat_yaw, vec)
