"""Diagnose SO-ARM100 communication, power, and position response.

The default run only reads registers and preserves the existing torque state.
Add ``--move`` for a bounded, low-speed shoulder_pan response test. Only the
base motor is enabled; its old goal is replaced before enabling torque.
"""

import argparse
import time

from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

PORT = "/dev/tty.usbmodem5B421348831"
ROBOT_ID = "pawacha_so100_follower"
MOTORS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def read_raw(robot, register, motor):
    return robot.bus.read(register, motor, normalize=False, num_retry=2)


def stop_motor(robot, motor):
    """Attempt torque-off even if a previous write returned a servo alarm."""
    try:
        robot.bus.write("Torque_Enable", motor, 0, normalize=False, num_retry=2)
    except Exception as exc:
        print(f"  Torque-off response: {exc}")
    try:
        if read_raw(robot, "Torque_Enable", motor) == 0:
            print(f"  {motor}: torque OFF verified.")
            return True
    except Exception as exc:
        print(f"  Torque readback failed: {exc}")
    print("  STOP: could not verify torque-off. Disconnect the motor power supply.")
    return False


def print_snapshot(robot):
    registers = (
        "Torque_Enable",
        "Operating_Mode",
        "Goal_Position",
        "Present_Position",
        "Present_Velocity",
        "Present_Load",
        "Present_Current",
        "Present_Voltage",
        "Present_Temperature",
        "Status",
        "Moving",
    )
    snapshot = {}

    print("\nRegister snapshot (raw values):")
    for motor in MOTORS:
        values = {}
        for register in registers:
            try:
                values[register] = read_raw(robot, register, motor)
            except Exception as exc:
                values[register] = f"ERROR: {exc}"
        snapshot[motor] = values
        voltage = values["Present_Voltage"]
        voltage_text = f"{voltage / 10:.1f} V" if isinstance(voltage, int) else voltage
        print(
            f"  {motor:14s} "
            f"torque={values['Torque_Enable']} mode={values['Operating_Mode']} "
            f"present={values['Present_Position']} goal={values['Goal_Position']} "
            f"voltage={voltage_text} current={values['Present_Current']} "
            f"status={values['Status']}"
        )

    torques = [snapshot[motor]["Torque_Enable"] for motor in MOTORS]
    modes = [snapshot[motor]["Operating_Mode"] for motor in MOTORS]
    voltage_raw = [snapshot[motor]["Present_Voltage"] for motor in MOTORS]
    voltages = [value / 10 for value in voltage_raw if isinstance(value, int)]

    print("\nSnapshot interpretation:")
    print(f"  Torque enabled on all motors: {all(value == 1 for value in torques)}")
    print(f"  Position mode on all motors:  {all(value == 0 for value in modes)}")
    if voltages:
        print(f"  Motor-bus voltage range:       {min(voltages):.1f}-{max(voltages):.1f} V")
        if max(voltages) < 6.0:
            print("  Bus voltage is below 6 V; this does not identify the servo voltage variant.")
            print("  Check the kit/servo rating before changing the power supply.")

    return snapshot


def run_move_test(robot, motor, degrees):
    if motor != "shoulder_pan" or not 0 < abs(degrees) <= 2:
        raise ValueError("Only a nonzero shoulder_pan step of at most 2 degrees is supported.")
    if not robot.is_calibrated:
        raise RuntimeError("Calibration does not match; movement test cancelled.")
    if any(read_raw(robot, "Torque_Enable", name) != 0 for name in MOTORS):
        raise RuntimeError("All motors must already have torque OFF before this test.")
    if read_raw(robot, "Operating_Mode", motor) != 0:
        raise RuntimeError("Base is not in position mode; no settings changed.")
    if read_raw(robot, "Status", motor) != 0:
        raise RuntimeError("Servo reports an alarm; movement test cancelled.")
    if read_raw(robot, "Present_Temperature", motor) >= 50:
        raise RuntimeError("Servo temperature is elevated; movement test cancelled.")
    if read_raw(robot, "Goal_Time", motor) != 0:
        raise RuntimeError("Unexpected Goal_Time; movement test cancelled.")

    before_raw = read_raw(robot, "Present_Position", motor)
    motor_id = robot.bus.motors[motor].id
    before_deg = robot.bus._normalize({motor_id: before_raw})[motor_id]
    target_deg = before_deg + degrees
    expected_raw = robot.bus._unnormalize({motor_id: target_deg})[motor_id]
    lower = read_raw(robot, "Min_Position_Limit", motor)
    upper = read_raw(robot, "Max_Position_Limit", motor)
    if not (lower + 5 <= min(before_raw, expected_raw)
            and max(before_raw, expected_raw) <= upper - 5):
        raise RuntimeError("Current/target position is too close to a calibrated limit.")
    baseline_voltage = read_raw(robot, "Present_Voltage", motor)
    min_voltage = read_raw(robot, "Min_Voltage_Limit", motor)
    max_voltage = read_raw(robot, "Max_Voltage_Limit", motor)
    if not min_voltage < baseline_voltage < max_voltage:
        raise RuntimeError("Voltage is outside configured limits; movement test cancelled.")
    previous_speed = read_raw(robot, "Goal_Velocity", motor)

    print(f"\nMove test: {motor} {degrees:+.1f} degrees")
    print(f"  before={before_deg:.3f} degrees (raw={before_raw})")
    samples = []
    try:
        # Speed units vary by firmware/settings; start with a small raw value.
        robot.bus.write("Goal_Velocity", motor, 2, normalize=False)
        if read_raw(robot, "Goal_Velocity", motor) != 2:
            raise RuntimeError("Low speed setting did not read back correctly.")
        # Some firmware may enable torque on a Goal_Position write; use the
        # current in-range encoder position, and cover this write with cleanup.
        robot.bus.write("Goal_Position", motor, before_raw, normalize=False)
        if read_raw(robot, "Goal_Position", motor) != before_raw:
            raise RuntimeError("Current-position goal did not read back correctly.")
        if abs(read_raw(robot, "Present_Position", motor) - before_raw) > 3:
            raise RuntimeError("Base position drifted during preparation.")
        robot.bus.write("Torque_Enable", motor, 1, normalize=False)
        if read_raw(robot, "Torque_Enable", motor) != 1:
            raise RuntimeError("Torque enable was not confirmed.")
        print("  Current-position goal and torque enable verified; other motors remain OFF.")

        sent = robot.send_action({f"{motor}.pos": target_deg})
        goal_raw = read_raw(robot, "Goal_Position", motor)
        print(f"  send_action returned {sent}; goal readback raw={goal_raw}")
        if abs(goal_raw - expected_raw) > 1:
            raise RuntimeError("Goal readback does not match the requested bounded target.")

        start = time.monotonic()
        for sample_time in (0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0):
            remaining = sample_time - (time.monotonic() - start)
            if remaining > 0:
                time.sleep(remaining)
            sample = {
                "position": read_raw(robot, "Present_Position", motor),
                "voltage": read_raw(robot, "Present_Voltage", motor),
                "status": read_raw(robot, "Status", motor),
                "torque": read_raw(robot, "Torque_Enable", motor),
                "current": read_raw(robot, "Present_Current", motor),
                "load": read_raw(robot, "Present_Load", motor),
            }
            samples.append(sample)
            print(f"  t={time.monotonic()-start:.3f}s {sample}")
            if sample["status"] or sample["torque"] != 1:
                raise RuntimeError("Servo alarm or torque shutdown detected; test stopped.")
            if (sample["voltage"] <= max(min_voltage, baseline_voltage - 4)
                    or sample["voltage"] >= max_voltage):
                raise RuntimeError("Voltage drop/out-of-range detected; test stopped.")
            if not min(before_raw, expected_raw) - 5 <= sample["position"] <= max(before_raw, expected_raw) + 5:
                raise RuntimeError("Position left the small test interval; test stopped.")
    finally:
        if stop_motor(robot, motor):
            robot.bus.write("Goal_Velocity", motor, previous_speed, normalize=False)
        else:
            raise RuntimeError("Torque-off unconfirmed; cut motor power before continuing.")

    position_delta = samples[-1]["position"] - before_raw
    goal_delta = goal_raw - before_raw
    print("\nDiagnostic result:")
    if goal_delta == 0:
        print("  FAIL: Goal_Position did not change; investigate command writes.")
    elif abs(position_delta) < 2:
        print("  Goal_Position changed, but the servo position did not.")
        print("  Target register accepted the command; power/servo response remains unresolved.")
    else:
        print(f"  Servo moved (raw delta={position_delta}); final target error={samples[-1]['position']-goal_raw} ticks.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--move",
        action="store_true",
        help="run a small opt-in movement test after printing the read-only snapshot",
    )
    parser.add_argument("--motor", choices=("shoulder_pan",), default="shoulder_pan")
    parser.add_argument("--degrees", type=float, default=2.0)
    args = parser.parse_args()

    if not 0 < abs(args.degrees) <= 2.0:
        parser.error("--degrees must be nonzero and between -2.0 and +2.0")

    robot = SO100Follower(
        SO100FollowerConfig(
            port=PORT,
            id=ROBOT_ID,
            max_relative_target=2.0,
            use_degrees=True,
        )
    )

    try:
        print("Connecting...")
        # Never call robot.connect(): it enables every motor using stored goals.
        robot.bus.connect()
        print("Bus connected: torque and motor settings are unchanged.")
        print(f"Connected. Calibration valid: {robot.is_calibrated}")
        print_snapshot(robot)

        if args.move:
            run_move_test(robot, args.motor, args.degrees)
        else:
            print("\nRead-only test complete. Use --move for the 2-degree response test.")
    finally:
        if robot.bus.is_connected:
            robot.bus.disconnect(disable_torque=False)
            print("Serial port closed; torque state unchanged.")


if __name__ == "__main__":
    main()
