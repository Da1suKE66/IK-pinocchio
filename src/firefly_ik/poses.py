"""SE(3) helpers: metres, radians, and quaternion order [x, y, z, w]."""

from collections.abc import Mapping, Sequence

import numpy as np
import pinocchio as pin


def vector(value: Sequence[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite numbers")
    return result.copy()


def checked_pose(pose: pin.SE3) -> pin.SE3:
    if not isinstance(pose, pin.SE3):
        raise ValueError("target must be a pinocchio.SE3")
    rotation = np.asarray(pose.rotation)
    if (
        not np.isfinite(pose.homogeneous).all()
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7, rtol=0)
    ):
        raise ValueError("pose must be a finite rigid transform with a proper rotation")
    return pose.copy()


def pose_from_xyz_rpy(xyz_m: Sequence[float], rpy_rad: Sequence[float]) -> pin.SE3:
    """Rotation is Rz(yaw) @ Ry(pitch) @ Rx(roll), as in URDF."""
    return pin.SE3(pin.rpy.rpyToMatrix(vector(rpy_rad, 3, "rpy_rad")),
                   vector(xyz_m, 3, "xyz_m"))


def pose_from_xyz_quat(xyz_m: Sequence[float], quat_xyzw: Sequence[float]) -> pin.SE3:
    quaternion = vector(quat_xyzw, 4, "quat_xyzw")
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("quat_xyzw must have nonzero finite norm")
    return pin.SE3(pin.Quaternion(quaternion / norm).matrix(), vector(xyz_m, 3, "xyz_m"))


def pose_from_dict(value: Mapping) -> pin.SE3:
    if not isinstance(value, Mapping) or "xyz_m" not in value:
        raise ValueError("a pose requires xyz_m and exactly one of rpy_rad / quat_xyzw")
    if ("rpy_rad" in value) == ("quat_xyzw" in value):
        raise ValueError("a pose requires exactly one of rpy_rad / quat_xyzw")
    if "rpy_rad" in value:
        return pose_from_xyz_rpy(value["xyz_m"], value["rpy_rad"])
    return pose_from_xyz_quat(value["xyz_m"], value["quat_xyzw"])


def pose_to_dict(pose: pin.SE3) -> dict:
    pose = checked_pose(pose)
    quaternion = pin.Quaternion(pose.rotation).coeffs().copy()
    if quaternion[3] < 0:
        quaternion *= -1
    return {
        "xyz_m": pose.translation.tolist(),
        "quat_xyzw": quaternion.tolist(),
        "matrix": pose.homogeneous.tolist(),
    }
