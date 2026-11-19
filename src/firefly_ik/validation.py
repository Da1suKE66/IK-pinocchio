"""Reproducible offline FK -> IK -> FK numerical validation."""

from time import perf_counter

import numpy as np
import pinocchio as pin

from .kinematics import ARMS, FireflyIK, IKOptions
from .poses import pose_to_dict


def demo_seed(robot: FireflyIK) -> np.ndarray:
    # These are numerical example seeds, not validated hardware home positions.
    return robot.configuration(left=[0, -0.5, 1.3, -0.8, 0.3, 0],
                               right=[0, -0.5, 1.3, -0.8, -0.3, 0])


def demo_targets(robot: FireflyIK, seed: np.ndarray) -> dict:
    targets = {arm: robot.fk(arm, seed) for arm in ARMS}
    for target in targets.values():
        target.translation = target.translation + np.array([0.01, 0, 0])
    return targets


def demo(robot: FireflyIK) -> dict:
    seed = demo_seed(robot)
    targets = demo_targets(robot, seed)
    result = robot.solve_dual(targets["left"], targets["right"], seed)
    return {
        "description": "Offline: move both TCPs +10 mm along world X, keeping orientation",
        "reference": robot.world_frame,
        "seed_rad": seed.tolist(),
        "targets": {arm: pose_to_dict(pose) for arm, pose in targets.items()},
        "result": result.to_dict(),
    }


def validate(robot: FireflyIK, samples_per_arm: int = 20, random_seed: int = 42) -> dict:
    if samples_per_arm < 1:
        raise ValueError("samples_per_arm must be at least 1")
    if random_seed < 0:
        raise ValueError("random_seed must be nonnegative")
    rng = np.random.default_rng(random_seed)
    options = IKOptions()
    started = perf_counter()
    cases = []
    for arm in ARMS:
        for sample in range(samples_per_arm):
            q = robot.lower_limits + rng.uniform(0.1, 0.9, 12) * (robot.upper_limits - robot.lower_limits)
            target = robot.fk(arm, q)
            seed = np.clip(q + rng.normal(0, 0.15, 12), robot.lower_limits, robot.upper_limits)
            result = robot.solve(arm, target, seed, options=options)
            # Measure FK again at the returned configuration rather than relying
            # on an optimizer termination flag.
            actual = robot.fk(arm, result.full_q)
            position_error = float(np.linalg.norm(actual.translation - target.translation))
            orientation_error = float(np.linalg.norm(pin.log3(actual.rotation.T @ target.rotation)))
            within_limits = bool(np.all(result.full_q >= robot.lower_limits)
                                 and np.all(result.full_q <= robot.upper_limits))
            passed = bool(result.success and within_limits
                          and position_error <= options.position_tolerance_m
                          and orientation_error <= options.orientation_tolerance_rad)
            cases.append({
                "arm": arm, "sample": sample, "passed": passed,
                "position_error_m": position_error,
                "orientation_error_rad": orientation_error,
                "within_limits": within_limits,
                "attempts": result.attempts, "evaluations": result.evaluations,
            })
    return {
        "validation_type": "offline numerical FK-IK-FK; not physical robot calibration",
        "success": all(case["passed"] for case in cases),
        "pinocchio_version": pin.__version__, "urdf": str(robot.urdf_path),
        "random_seed": random_seed, "samples_per_arm": samples_per_arm,
        "total_cases": len(cases), "passed_cases": sum(case["passed"] for case in cases),
        "position_tolerance_m": options.position_tolerance_m,
        "orientation_tolerance_rad": options.orientation_tolerance_rad,
        "max_position_error_m": max(case["position_error_m"] for case in cases),
        "max_orientation_error_rad": max(case["orientation_error_rad"] for case in cases),
        "elapsed_seconds": perf_counter() - started,
        "collision_checked": False, "cases": cases,
    }
