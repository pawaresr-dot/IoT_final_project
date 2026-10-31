"""Save the SO-ARM100's current joint readings without moving the arm.

Move the unpowered arm by hand to the desired pose, then run this command with
--name top_left/top_right/bottom_left/bottom_right. This script does not enable
torque or write any motor register.

Use --stage approach/grasp/grasp_closed/lift/place/retreat to save an observation in
pick_place_poses.json, separately from the old cell poses. Staged observations
are not motion-ready until gripper targets and the route have been verified.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig


PORT = "/dev/tty.usbmodem5B421348831"
ROBOT_ID = "pawacha_so100_follower"
MOTORS = (
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
)
NAMES = ("top_left", "top_right", "bottom_left", "bottom_right")
DEFAULT_OUTPUT = Path(__file__).with_name("arm_poses.json")
DEFAULT_STAGED_OUTPUT = Path(__file__).with_name("pick_place_poses.json")
STAGES = ("approach", "grasp", "grasp_closed", "lift", "place", "retreat")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", choices=NAMES, required=True)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--port", default=PORT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.output = args.output or (DEFAULT_STAGED_OUTPUT if args.stage else DEFAULT_OUTPUT)

    robot = SO100Follower(SO100FollowerConfig(
        port=args.port, id=ROBOT_ID, use_degrees=True,
    ))
    try:
        # robot.connect() enables torque. The bus-only connection is read-only.
        robot.bus.connect()
        if not robot.is_calibrated:
            raise RuntimeError("Calibration is missing or does not match")
        torque = {name: robot.bus.read("Torque_Enable", name, normalize=False)
                  for name in MOTORS}
        if any(torque.values()):
            raise RuntimeError(f"Torque must be OFF on every motor: {torque}")
        raw = {name: int(robot.bus.read("Present_Position", name, normalize=False))
               for name in MOTORS}
        normalized_by_id = robot.bus._normalize({
            robot.bus.motors[name].id: raw[name] for name in MOTORS
        })
        normalized = {
            name: normalized_by_id[robot.bus.motors[name].id] for name in MOTORS
        }
        limits = {
            name: {
                "min": int(robot.bus.read("Min_Position_Limit", name, normalize=False)),
                "max": int(robot.bus.read("Max_Position_Limit", name, normalize=False)),
            } for name in MOTORS
        }
        outside = [name for name in MOTORS
                   if not limits[name]["min"] <= raw[name] <= limits[name]["max"]]
        arm_outside = [name for name in outside if name != "gripper"]
        if arm_outside:
            raise RuntimeError(f"Arm position is outside calibrated limits: {arm_outside}")
        pose = {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "joint_degrees": {name: float(normalized[name]) for name in MOTORS[:-1]},
            "gripper_percent": float(normalized["gripper"]),
            "raw_position": raw,
            "limits": limits,
        }
        if args.stage:
            pose["validation"] = {
                "arm_in_range": True,
                "gripper_in_range": "gripper" not in outside,
                "outside_limits": outside,
                "ready_for_motion": False,
                "note": "Teaching observation; verify gripper targets, clearance, and route before motion.",
            }
            if args.stage == "grasp_closed":
                # The arm may drift while the operator closes the fingers.
                # Record that posture, but use only the gripper target for closure.
                pose["control_hint"] = "gripper_only"
    finally:
        if robot.bus.is_connected:
            robot.bus.disconnect(disable_torque=False)

    data = json.loads(args.output.read_text()) if args.output.exists() else {
        "robot_id": ROBOT_ID, "poses": {},
    }
    if data.get("robot_id") != ROBOT_ID:
        raise RuntimeError("Output file belongs to a different robot")
    if args.stage:
        if data.get("schema") not in (None, "staged_teaching_v1"):
            raise RuntimeError("Output file has an incompatible schema")
        if any("joint_degrees" in existing for existing in data["poses"].values()):
            raise RuntimeError("Do not mix staged teaching observations with old flat cell poses")
        stages = data["poses"].setdefault(args.name, {})
        if args.stage in stages and not args.overwrite:
            raise RuntimeError(f"{args.name}/{args.stage} already exists; use --overwrite to replace it")
        data["schema"] = "staged_teaching_v1"
        stages[args.stage] = pose
    else:
        if data.get("schema") == "staged_teaching_v1":
            raise RuntimeError("Use --stage with the staged teaching file")
        if args.name in data["poses"] and not args.overwrite:
            raise RuntimeError(f"{args.name} already exists; use --overwrite to replace it")
        data["poses"][args.name] = pose
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({"name": args.name, "stage": args.stage, **pose}, indent=2))
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
