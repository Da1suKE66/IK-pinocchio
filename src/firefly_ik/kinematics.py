"""Bounded numerical IK using Pinocchio frame Jacobians and SciPy TRF."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares

from .branches import arm_geometry, planar_branches
from .poses import checked_pose, pose_from_dict, pose_to_dict, vector


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URDF = PROJECT_ROOT / "firefly_viewer_82mm/firefly_dual.urdf"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/firefly_dual.json"
ARMS = ("left", "right")


def _bounded_fit(residual, jacobian, start, lower, upper, evaluations):
    return least_squares(residual, start, jac=jacobian, bounds=(lower, upper), method="trf",
                         max_nfev=evaluations, ftol=1e-12, xtol=1e-12, gtol=1e-12)


@dataclass(frozen=True)
class IKOptions:
    position_tolerance_m: float = 1e-5
    orientation_tolerance_rad: float = 1e-4
    orientation_weight_m: float = 0.2
    max_evaluations: int = 300
    attempts: int = 4
    random_seed: int = 0

    def __post_init__(self):
        for name in ("position_tolerance_m", "orientation_tolerance_rad", "orientation_weight_m"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("max_evaluations", "attempts"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int) or self.random_seed < 0:
            raise ValueError("random_seed must be a nonnegative integer")


@dataclass
class IKResult:
    success: bool
    arm: str
    q: np.ndarray
    full_q: np.ndarray
    position_error_m: float
    orientation_error_rad: float
    evaluations: int
    attempts: int
    message: str
    position_only: bool

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "arm": self.arm,
            "q_rad": self.q.tolist(),
            "q_deg": np.rad2deg(self.q).tolist(),
            "full_q_rad": self.full_q.tolist(),
            "position_error_m": self.position_error_m,
            "orientation_error_rad": self.orientation_error_rad,
            "evaluations": self.evaluations,
            "attempts": self.attempts,
            "message": self.message,
            "position_only": self.position_only,
            "collision_checked": False,
        }


@dataclass
class DualIKResult:
    success: bool
    q: np.ndarray
    left: IKResult
    right: IKResult

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "full_q_rad": self.q.tolist(),
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
            "collision_checked": False,
        }


class FireflyIK:
    """Fixed-base 12-DOF workcell. Vectors use left J1..J6, right J1..J6.

    Public vectors are independent of Pinocchio's internal traversal order.
    Gripper joints are locked at URDF zero. Each call uses its own model data.
    A converged result certifies pose residual and joint limits only.
    """

    def __init__(
        self,
        urdf_path: str | Path = DEFAULT_URDF,
        *,
        tcp_offsets: Mapping[str, pin.SE3] | None = None,
        joint_limits: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
    ):
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        if not self.urdf_path.is_file():
            raise ValueError(f"URDF does not exist: {self.urdf_path}")
        self._xml = ET.parse(self.urdf_path).getroot()
        links = {link.get("name") for link in self._xml.findall("link")}
        children = {joint.find("child").get("link") for joint in self._xml.findall("joint")}
        roots = links - children
        if len(roots) != 1:
            raise ValueError("URDF must have exactly one root link")
        self.world_frame = roots.pop()
        self.joint_names = tuple(f"{arm}_joint_{i}" for arm in ARMS for i in range(1, 7))
        raw_model = pin.buildModelFromUrdf(str(self.urdf_path))
        missing = set(self.joint_names) - set(raw_model.names)
        if missing:
            raise ValueError(f"missing Firefly arm joints: {sorted(missing)}")
        self.locked_joints = tuple(name for name in raw_model.names[1:] if name not in self.joint_names)
        locked_ids = [raw_model.getJointId(name) for name in self.locked_joints]
        self.model = pin.buildReducedModel(raw_model, locked_ids, pin.neutral(raw_model))
        if self.model.nq != 12 or self.model.nv != 12:
            raise ValueError("the reduced Firefly model must have exactly 12 scalar arm joints")
        joints = [self.model.joints[self.model.getJointId(name)] for name in self.joint_names]
        if any(joint.nq != 1 or joint.nv != 1 for joint in joints):
            raise ValueError("only scalar bounded revolute arm joints are supported")
        self._q_indices = np.array([joint.idx_q for joint in joints])
        self._v_indices = np.array([joint.idx_v for joint in joints])
        self.lower_limits = self.model.lowerPositionLimit[self._q_indices].copy()
        self.upper_limits = self.model.upperPositionLimit[self._q_indices].copy()
        self.velocity_limits = self.model.velocityLimit[self._v_indices].copy()
        if not np.isfinite([self.lower_limits, self.upper_limits]).all():
            raise ValueError("all arm joint limits must be finite")
        for arm, limits in (joint_limits or {}).items():
            indices = self._arm_slice(arm)
            if not isinstance(limits, Mapping) or set(limits) != {"lower_rad", "upper_rad"}:
                raise ValueError("joint_limits entries require lower_rad and upper_rad")
            self.lower_limits[indices] = np.maximum(
                self.lower_limits[indices], vector(limits["lower_rad"], 6, f"{arm} lower_rad"))
            self.upper_limits[indices] = np.minimum(
                self.upper_limits[indices], vector(limits["upper_rad"], 6, f"{arm} upper_rad"))
        if np.any(self.lower_limits >= self.upper_limits):
            raise ValueError("joint limit intersection is empty or has a zero-width interval")
        self.tcp_offsets = {arm: pin.SE3.Identity() for arm in ARMS}
        for arm, offset in (tcp_offsets or {}).items():
            self._arm_slice(arm)
            self.tcp_offsets[arm] = checked_pose(offset)
        self._tcp_frames = {}
        self._base_frames = {}
        for arm in ARMS:
            base_id = self._frame_id(f"{arm}_base_link")
            if self.model.frames[base_id].parentJoint != 0:
                raise ValueError("arm bases must be fixed in the world frame")
            self._base_frames[arm] = base_id
            flange_id = self._frame_id(f"{arm}_link_6")
            flange = self.model.frames[flange_id]
            name = f"{arm}_ik_tcp"
            if self.model.existFrame(name):
                raise ValueError(f"reserved frame already exists: {name}")
            self._tcp_frames[arm] = self.model.addFrame(pin.Frame(
                name, flange.parentJoint, flange_id,
                flange.placement * self.tcp_offsets[arm], pin.FrameType.OP_FRAME))

    @classmethod
    def from_config(cls, path: str | Path = DEFAULT_CONFIG) -> "FireflyIK":
        path = Path(path).expanduser().resolve()
        config = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or set(config) - {"urdf", "tcp_offsets", "joint_limits"}:
            raise ValueError("config accepts urdf, tcp_offsets and joint_limits only")
        if "urdf" not in config:
            raise ValueError("config requires urdf")
        offsets = {arm: pose_from_dict(value) for arm, value in config.get("tcp_offsets", {}).items()}
        return cls(path.parent / config["urdf"], tcp_offsets=offsets,
                   joint_limits=config.get("joint_limits"))

    @staticmethod
    def _arm_slice(arm: str) -> slice:
        if arm not in ARMS:
            raise ValueError("arm must be 'left' or 'right'")
        return slice(0, 6) if arm == "left" else slice(6, 12)

    def _frame_id(self, name: str) -> int:
        if not self.model.existFrame(name):
            raise ValueError(f"unknown frame: {name}")
        return self.model.getFrameId(name)

    def _validate_q(self, q: Sequence[float]) -> np.ndarray:
        q = vector(q, 12, "full joint configuration")
        bad = np.flatnonzero((q < self.lower_limits) | (q > self.upper_limits))
        if bad.size:
            names = ", ".join(self.joint_names[i] for i in bad)
            raise ValueError(f"joint configuration exceeds effective limits: {names}")
        return q

    def neutral(self) -> np.ndarray:
        """URDF neutral projected into configured limits; not a hardware home pose."""
        q = pin.neutral(self.model)[self._q_indices]
        return np.clip(q, self.lower_limits, self.upper_limits)

    def configuration(self, *, left=None, right=None) -> np.ndarray:
        q = self.neutral()
        for arm, values in (("left", left), ("right", right)):
            if values is not None:
                q[self._arm_slice(arm)] = vector(values, 6, f"{arm} joint configuration")
        return self._validate_q(q)

    def _configuration_for_arm(self, arm, q) -> np.ndarray:
        indices = self._arm_slice(arm)
        if q is None:
            return self.neutral()
        values = np.asarray(q, dtype=float)
        if values.shape == (6,):
            full = self.neutral()
            full[indices] = vector(values, 6, f"{arm} joint configuration")
            return self._validate_q(full)
        return self._validate_q(values)

    def _pin_configuration(self, q):
        result = pin.neutral(self.model)
        result[self._q_indices] = q
        return result

    def _reference_pose(self, arm: str, reference: str) -> pin.SE3:
        if reference in ("world", self.world_frame):
            return pin.SE3.Identity()
        if reference in ("base", f"{arm}_base_link"):
            return self.model.frames[self._base_frames[arm]].placement.copy()
        raise ValueError(f"reference must be world / {self.world_frame} or base / {arm}_base_link")

    def frame_pose(self, frame: str, q: Sequence[float] | None = None) -> pin.SE3:
        """World pose of any URDF frame, including the nominal camera frames."""
        frame_id = self._frame_id(frame)
        values = self.neutral() if q is None else self._validate_q(q)
        data = self.model.createData()
        pin.framesForwardKinematics(self.model, data, self._pin_configuration(values))
        return data.oMf[frame_id].copy()

    def fk(self, arm: str, q=None, *, reference: str = "world", tcp: bool = True) -> pin.SE3:
        values = self._configuration_for_arm(arm, q)
        frame_id = self._tcp_frames[arm] if tcp else self._frame_id(f"{arm}_link_6")
        data = self.model.createData()
        pin.framesForwardKinematics(self.model, data, self._pin_configuration(values))
        return self._reference_pose(arm, reference).actInv(data.oMf[frame_id])

    def pose_branches(self, arm: str, target: pin.SE3, seed=None, *, reference: str = "world") -> dict:
        """Enumerate exact joint-limit-valid full poses, nearest the seed first.

        Singular families are explicitly marked incomplete. Each discrete
        candidate is checked against the configured TCP using Pinocchio FK.
        """
        indices = self._arm_slice(arm)
        initial = self._configuration_for_arm(arm, seed)
        world = self._reference_pose(arm, reference) * checked_pose(target)
        flange = self._reference_pose(arm, "base").actInv(world) * self.tcp_offsets[arm].inverse()
        offsets = arm_geometry(self._xml, arm)
        lower, upper = self.lower_limits[indices], self.upper_limits[indices]
        qs, rejected, singular = planar_branches(
            flange, offsets, lower, upper,
            seed_yaw=initial[indices][0], seed_pitch=np.sum(initial[indices][1:4]))
        origin = initial[indices]
        # An ill-conditioned analytic parameter (yaw or wrist phase) can swing
        # widely even when the previous posture is a few microns from the goal.
        nearest = min((np.linalg.norm(q - origin) for q in qs), default=np.inf)
        recovered = False
        if seed is not None and nearest > 1e-4:
            continued = self._continue_nearby(arm, world, initial)
            if continued is not None and np.linalg.norm(continued - origin) + 1e-6 < nearest:
                recovered = not qs
                qs.append(continued)
        candidates = []
        discarded = False
        def verify(q):
            actual = self.fk(arm, q)
            distance = float(np.linalg.norm(actual.translation - world.translation))
            angle = float(np.linalg.norm(pin.log3(actual.rotation.T @ world.rotation)))
            return {"q_rad": q.tolist(), "position_error_m": distance, "orientation_error_rad": angle}, (
                distance <= 1e-8 and angle <= 1e-7)

        for q in sorted(qs, key=lambda v: np.linalg.norm(v - origin)):
            checked, valid = verify(q)
            if not valid:
                # Regular and singular hypotheses overlap near the numerical pole.
                discarded = True
                rejected.append(dict(reason="fk_residual", **checked))
                continue
            candidates.append(checked)
        reach_bound = np.linalg.norm(offsets[1:], axis=1).sum()
        if not candidates and np.linalg.norm(flange.translation - offsets[0]) <= reach_bound + 1e-8:
            # Near a wrist pole, a small orientation error can move the computed
            # planar point past an elbow tangency. Check the original 6D pose.
            residual, jacobian = self._task(arm, world, initial, False, .2)
            for start in (origin, (lower + upper) / 2):
                q = _bounded_fit(residual, jacobian, start, lower, upper, 300).x
                checked, valid = verify(q)
                if valid:
                    candidates.append(dict(method="bounded_numerical_recovery", **checked))
                    recovered = True
                    break
        return {"arm": arm, "success": bool(candidates), "complete": not (singular or discarded or recovered), "target_world": pose_to_dict(world),
                "candidates": candidates, "rejected": rejected, "singular_families": singular,
                "collision_checked": False}

    def _continue_nearby(self, arm, world, initial):
        """Minimum-motion correction from the supplied posture for a nearby TCP."""
        indices = self._arm_slice(arm)
        q = initial[indices].copy()
        residual, jacobian = self._task(arm, world, initial, False, .2)
        error = residual(q)
        if np.linalg.norm(error[:3]) > .002 or np.linalg.norm(error[3:]) > .002:
            return None
        for _ in range(12):
            if np.linalg.norm(error[:3]) < 2e-10 and np.linalg.norm(error[3:]) < 2e-10:
                break
            j = jacobian(q)
            step = -j.T @ np.linalg.solve(j @ j.T + 1e-8 * np.eye(6), error)
            q = np.clip(q + step, self.lower_limits[indices], self.upper_limits[indices])
            error = residual(q)
        if np.linalg.norm(error[:3]) <= 5e-9 and np.linalg.norm(error[3:]) <= 1e-8:
            return q
        return None

    def _task(self, arm, target_world, initial, position_only, orientation_weight_m):
        """Return a cached residual/Jacobian pair for one bounded six-axis task."""
        arm_slice = self._arm_slice(arm)
        q_ids = self._q_indices[arm_slice]
        v_ids = self._v_indices[arm_slice]
        frame_id = self._tcp_frames[arm]
        data = self.model.createData()
        fixed_q = self._pin_configuration(initial)
        weights = np.array([1, 1, 1, orientation_weight_m, orientation_weight_m, orientation_weight_m])
        cached_x = None
        cached_error = None
        cached_jacobian = None

        def evaluate(x):
            nonlocal cached_x, cached_error, cached_jacobian
            if cached_x is not None and np.array_equal(x, cached_x):
                return cached_error, cached_jacobian
            q = fixed_q.copy()
            q[q_ids] = x
            pin.computeJointJacobians(self.model, data, q)
            pin.updateFramePlacements(self.model, data)
            current = data.oMf[frame_id]
            if position_only:
                cached_error = (current.translation - target_world.translation).copy()
                cached_jacobian = pin.getFrameJacobian(
                    self.model, data, frame_id, pin.LOCAL_WORLD_ALIGNED)[:3, v_ids].copy()
            else:
                delta = current.actInv(target_world)
                cached_error = weights * pin.log6(delta).vector
                jacobian = pin.getFrameJacobian(self.model, data, frame_id, pin.LOCAL)[:, v_ids]
                cached_jacobian = weights[:, None] * (-pin.Jlog6(delta.inverse()) @ jacobian)
            cached_x = x.copy()
            return cached_error, cached_jacobian

        return lambda x: evaluate(x)[0], lambda x: evaluate(x)[1]

    def solve(
        self, arm: str, target: pin.SE3, seed=None, *, reference: str = "world",
        position_only: bool = False, options: IKOptions | None = None,
    ) -> IKResult:
        """Solve a TCP target. On failure q/full_q are diagnostic candidates only.

        A six-value seed sets the active arm; a twelve-value seed also preserves
        the other arm. The first converged attempt wins, favoring the seed's
        local branch. Restarts can change branches and are not path planning.
        """
        options = options or IKOptions()
        arm_slice = self._arm_slice(arm)
        initial = self._configuration_for_arm(arm, seed)
        target_world = self._reference_pose(arm, reference) * checked_pose(target)
        lower = self.lower_limits[arm_slice]
        upper = self.upper_limits[arm_slice]
        residual, jacobian = self._task(arm, target_world, initial, position_only, options.orientation_weight_m)
        rng = np.random.default_rng(options.random_seed)
        best = None
        best_score = np.inf
        total_evaluations = 0
        for attempt in range(1, options.attempts + 1):
            if attempt == 1:
                start = initial[arm_slice].copy()
            elif attempt == 2:
                start = (lower + upper) / 2
                if not position_only:
                    # Keep single-attempt tracking local. For a restarted full-pose
                    # solve, use a verified Firefly branch when the layout supports it.
                    try:
                        branches = self.pose_branches(arm, target_world, initial)["candidates"]
                    except ValueError:
                        branches = []  # General URDF chains still use the numerical solver.
                    if branches:
                        start = np.asarray(branches[0]["q_rad"])
            else:
                start = rng.uniform(lower, upper)
            optimized = _bounded_fit(residual, jacobian, start, lower, upper, options.max_evaluations)
            total_evaluations += optimized.nfev
            q = initial.copy()
            q[arm_slice] = optimized.x
            current = self.fk(arm, q)
            position_error = float(np.linalg.norm(current.translation - target_world.translation))
            orientation_error = float(np.linalg.norm(pin.log3(current.rotation.T @ target_world.rotation)))
            success = position_error <= options.position_tolerance_m and (
                position_only or orientation_error <= options.orientation_tolerance_rad)
            score = position_error / options.position_tolerance_m
            if not position_only:
                score = max(score, orientation_error / options.orientation_tolerance_rad)
            result = IKResult(
                success, arm, q[arm_slice].copy(), q, position_error, orientation_error,
                total_evaluations, attempt,
                "converged" if success else "target_not_reached; candidate is diagnostic only",
                position_only,
            )
            if success:
                return result
            if score < best_score:
                best, best_score = result, score
        best.evaluations = total_evaluations
        best.attempts = options.attempts
        return best

    def solve_dual(
        self, left_target: pin.SE3, right_target: pin.SE3, seed=None, *,
        reference: str = "world", position_only: bool = False, options: IKOptions | None = None,
    ) -> DualIKResult:
        """Solve independent branches and merge in one common configuration.

        This does not add an inter-arm collision or relative-pose constraint.
        """
        initial = self.neutral() if seed is None else self._validate_q(seed)
        left = self.solve("left", left_target, initial, reference=reference,
                          position_only=position_only, options=options)
        right = self.solve("right", right_target, initial, reference=reference,
                           position_only=position_only, options=options)
        q = initial.copy()
        q[:6], q[6:] = left.q, right.q
        left.full_q, right.full_q = q.copy(), q.copy()
        return DualIKResult(left.success and right.success, q, left, right)

    def info(self) -> dict:
        links = self._xml.findall("link")
        missing_meshes = sorted({
            mesh.get("filename") for mesh in self._xml.findall(".//mesh")
            if not (self.urdf_path.parent / mesh.get("filename")).is_file()
        })
        return {
            "urdf": str(self.urdf_path), "pinocchio_version": pin.__version__,
            "world_frame": self.world_frame, "nq": self.model.nq, "nv": self.model.nv,
            "joint_order": list(self.joint_names), "locked_joints": list(self.locked_joints),
            "lower_limits_rad": self.lower_limits.tolist(),
            "upper_limits_rad": self.upper_limits.tolist(),
            "velocity_limits_rad_s": self.velocity_limits.tolist(),
            "effort_limits_nm": self.model.effortLimit[self._v_indices].tolist(),
            "base_poses_world": {arm: pose_to_dict(self._reference_pose(arm, "base")) for arm in ARMS},
            "tcp_offsets_from_link_6": {arm: pose_to_dict(self.tcp_offsets[arm]) for arm in ARMS},
            "urdf_links": len(links), "urdf_joints": len(self._xml.findall("joint")),
            "collision_elements": len(self._xml.findall(".//collision")),
            "links_without_collision": [link.get("name") for link in links if link.find("collision") is None],
            "missing_meshes": missing_meshes, "collision_checked": False,
        }
