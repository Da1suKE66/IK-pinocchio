"""JSON command-line interface; no hardware or network operations."""

import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from .kinematics import DEFAULT_CONFIG, FireflyIK, IKOptions
from .poses import pose_from_dict, pose_from_xyz_quat, pose_from_xyz_rpy, pose_to_dict
from .validation import demo, validate


def _task_arguments(parser):
    parser.add_argument("--reference", choices=("world", "base"), default="world")
    parser.add_argument("--seed", type=float, nargs="+", help="6 active joints or 12 full joints, in radians")
    parser.add_argument("--attempts", type=int, default=4, help="use 1 to disable restarts")
    parser.add_argument("--max-evaluations", type=int, default=300, help="per attempt")
    parser.add_argument("--all-branches", action="store_true",
                        help="enumerate discrete full-pose IK solutions within the configured joint limits")


def _options(args):
    return IKOptions(attempts=args.attempts, max_evaluations=args.max_evaluations)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Firefly dual-arm offline FK/IK (metres and radians)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)
    info = commands.add_parser("info", help="inspect model, frames, limits and mesh paths")
    fk = commands.add_parser("fk", help="TCP forward kinematics")
    fk.add_argument("--arm", required=True, choices=("left", "right"))
    fk.add_argument("--q", type=float, nargs="+", help="6 active or 12 full joint angles in radians")
    fk.add_argument("--reference", choices=("world", "base"), default="world")
    fk.add_argument("--flange", action="store_true", help="return link_6 instead of configured TCP")
    ik = commands.add_parser("ik", help="solve one TCP target")
    ik.add_argument("--arm", required=True, choices=("left", "right"))
    ik.add_argument("--xyz", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    orientation = ik.add_mutually_exclusive_group(required=True)
    orientation.add_argument("--rpy", type=float, nargs=3, metavar=("ROLL", "PITCH", "YAW"))
    orientation.add_argument("--quat", type=float, nargs=4, metavar=("X", "Y", "Z", "W"))
    orientation.add_argument("--position-only", action="store_true")
    _task_arguments(ik)
    dual = commands.add_parser("dual", help="solve left and right targets from a JSON file")
    dual.add_argument("--targets", type=Path, required=True, help="JSON with left/right pose objects")
    dual.add_argument("--position-only", action="store_true")
    _task_arguments(dual)
    example = commands.add_parser("demo", help="offline dual-arm +10 mm translation example")
    check = commands.add_parser("validate", help="reproducible FK -> IK -> FK validation")
    check.add_argument("--samples", type=int, default=20, help="samples for each arm")
    check.add_argument("--random-seed", type=int, default=42)
    for command in (info, fk, ik, dual, example, check):
        command.add_argument("--output", type=Path, help="also save JSON to this file")
    args = parser.parse_args(argv)
    try:
        if args.command in ("ik", "dual") and args.all_branches and args.position_only:
            raise ValueError("--all-branches requires a full target orientation")
        robot = FireflyIK.from_config(args.config)
        success = True
        if args.command == "info":
            result = robot.info()
        elif args.command == "fk":
            result = {
                "arm": args.arm, "reference": args.reference,
                "endpoint": f"{args.arm}_link_6" if args.flange else f"{args.arm}_ik_tcp",
                "pose": pose_to_dict(robot.fk(args.arm, args.q, reference=args.reference, tcp=not args.flange)),
            }
        elif args.command == "ik":
            if args.quat is not None:
                target = pose_from_xyz_quat(args.xyz, args.quat)
            else:
                target = pose_from_xyz_rpy(args.xyz, args.rpy if args.rpy is not None else [0, 0, 0])
            if args.all_branches:
                result = robot.pose_branches(args.arm, target, args.seed, reference=args.reference)
                success = result["success"]
            else:
                solved = robot.solve(args.arm, target, args.seed, reference=args.reference,
                                     position_only=args.position_only, options=_options(args))
                result, success = solved.to_dict(), solved.success
        elif args.command == "dual":
            targets = json.loads(args.targets.read_text(encoding="utf-8"))
            if args.all_branches:
                if args.seed is not None and len(args.seed) != 12:
                    raise ValueError("dual IK requires 12 seed joint angles")
                arms = {s: robot.pose_branches(s, pose_from_dict(targets[s]), args.seed, reference=args.reference)
                        for s in ("left", "right")}
                success = all(arm["success"] for arm in arms.values())
                result = {"success": success, "complete": all(arm["complete"] for arm in arms.values()),
                          "arms": arms, "collision_checked": False}
            else:
                solved = robot.solve_dual(pose_from_dict(targets["left"]), pose_from_dict(targets["right"]),
                                          args.seed, reference=args.reference,
                                          position_only=args.position_only, options=_options(args))
                result, success = solved.to_dict(), solved.success
        elif args.command == "demo":
            result = demo(robot)
            success = result["result"]["success"]
        else:
            result = validate(robot, args.samples, args.random_seed)
            success = result["success"]
        serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized, encoding="utf-8")
        sys.stdout.write(serialized)
        return 0 if success else 1
    except (ValueError, TypeError, KeyError, OSError, ET.ParseError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
