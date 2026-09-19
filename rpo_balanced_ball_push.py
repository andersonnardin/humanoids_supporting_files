"""Push a physical ball with the free-standing RoboParty RPO humanoid.

A right-arm Jacobian controller reaches and pushes the ball while a filtered
CoM-feedback controller adjusts only the legs and torso to maintain a
double-support stance.

Controls: E pause/resume, R restart, Q quit.
"""

from __future__ import annotations

import argparse
import math
import time
import tkinter as tk
from collections import deque
from pathlib import Path

import numpy as np

try:
    import pybullet as p
    import pybullet_data
except ImportError as error:  # pragma: no cover
    raise SystemExit("Install PyBullet first: python3 -m pip install pybullet") from error


ROOT = Path(__file__).resolve().parent
RPO_URDF = ROOT / "rpo_description" / "urdf" / "rpo.urdf"
TIME_STEP = 1.0 / 240.0
BALANCE_MONITOR_SAMPLE_STEPS = 4

LEFT_ANKLE = "left_ankle_roll_joint"
RIGHT_ANKLE = "right_ankle_roll_joint"
RIGHT_HAND = "right_elbow_yaw_joint"
RIGHT_ARM_JOINTS = (
    "right_arm_pitch_joint",
    "right_arm_roll_joint",
    "right_arm_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_yaw_joint",
)

# A compact, bent-knee double-support posture.  The standing controller adds
# small corrections to this reference; the right arm is controlled separately.
STANDING_POSTURE = {
    "left_thigh_yaw_joint": -0.02535,
    "left_thigh_roll_joint": 0.04388,
    "left_thigh_pitch_joint": -0.33319,
    "left_knee_joint": 0.85733,
    "left_ankle_pitch_joint": -0.52466,
    "left_ankle_roll_joint": -0.05065,
    "right_thigh_yaw_joint": 0.02536,
    "right_thigh_roll_joint": -0.04388,
    "right_thigh_pitch_joint": -0.33319,
    "right_knee_joint": 0.85732,
    "right_ankle_pitch_joint": -0.52466,
    "right_ankle_roll_joint": 0.05066,
    "torso_joint": 0.0,
    "left_arm_pitch_joint": 0.15,
    "left_arm_roll_joint": 0.0,
    "left_arm_yaw_joint": 0.0,
    "left_elbow_pitch_joint": 0.0,
    "left_elbow_yaw_joint": 0.0,
    "right_arm_pitch_joint": 0.15,
    "right_arm_roll_joint": 0.0,
    "right_arm_yaw_joint": 0.0,
    "right_elbow_pitch_joint": 0.0,
    "right_elbow_yaw_joint": 0.0,
}

# Task-scene setup. Positions are world coordinates in metres. Change these
# values to place the table, ball, and desired hand motion elsewhere.
ROBOT_BASE_POSITION = (0.0, 0.0, 0.695)
TABLE_POSITION = (0.20, 0.0, 0.57)
TABLE_HALF_EXTENTS = (0.18, 0.20, 0.025)
BALL_POSITION = (0.10, -0.18, 0.63)
BALL_RADIUS = 0.035
# The terminal link frame is at the elbow-yaw joint. This point lies at the
# visible wrist end of that link's mesh.
WRIST_POINT_OFFSET = (0.10, 0.0, 0.0)
WRIST_CLEARANCE = 0.01
PUSH_DIRECTION = (0.0, 1.0, 0.0)
APPROACH_DISTANCE = 0.03
PUSH_DISTANCE = 0.08
RETURN_START_TIME = 3.0
RETURN_DURATION = 1.0
ARM_JACOBIAN_GAIN = 1.0
ARM_JACOBIAN_DAMPING = 0.04
ARM_MAX_STEP = 0.035
ARM_JACOBIAN_EPSILON = 1.0e-4

# The visible wrist point moves: approach -> contact the ball -> push through it.


class BalanceMonitor:
    """Show live sagittal and lateral CoM error histories."""

    HISTORY_LENGTH = 300
    ERROR_LIMIT_METERS = 0.12

    def __init__(self) -> None:
        self.window = tk.Tk()
        self.window.title("RPO Balance Monitor")
        self.window.resizable(False, False)
        self.canvas = tk.Canvas(
            self.window,
            width=520,
            height=370,
            background="white",
            highlightthickness=1,
            highlightbackground="#666666",
        )
        self.canvas.pack(padx=12, pady=12)
        self.sagittal: deque[float] = deque(maxlen=self.HISTORY_LENGTH)
        self.lateral: deque[float] = deque(maxlen=self.HISTORY_LENGTH)
        self.sagittal_control = 0.0
        self.lateral_control = 0.0
        self.window.protocol("WM_DELETE_WINDOW", self.window.withdraw)
        self.redraw()

    def clear(self) -> None:
        self.sagittal.clear()
        self.lateral.clear()
        self.sagittal_control = self.lateral_control = 0.0
        self.redraw()

    def update(self) -> None:
        try:
            self.window.update_idletasks()
            self.window.update()
        except tk.TclError:
            pass

    def close(self) -> None:
        try:
            self.window.destroy()
        except tk.TclError:
            pass

    def add(
        self,
        sagittal_error: float,
        lateral_error: float,
        sagittal_control: float,
        lateral_control: float,
    ) -> None:
        self.sagittal.append(sagittal_error)
        self.lateral.append(lateral_error)
        self.sagittal_control = sagittal_control
        self.lateral_control = lateral_control
        if self.window.state() != "withdrawn":
            self.redraw()

    def redraw(self) -> None:
        self.canvas.delete("all")
        width, height = 520, 370
        left, right = 54, width - 14
        panel_height = 110
        history_seconds = self.HISTORY_LENGTH * BALANCE_MONITOR_SAMPLE_STEPS * TIME_STEP
        for panel, (title, values, control, color) in enumerate(
            (
                ("Sagittal CoM error", self.sagittal, self.sagittal_control, "#1565c0"),
                ("Lateral CoM error", self.lateral, self.lateral_control, "#c62828"),
            )
        ):
            top = 32 + panel * 165
            bottom = top + panel_height
            middle = 0.5 * (top + bottom)
            self.canvas.create_text(left, top - 13, anchor="w", text=title, fill="#222222")
            self.canvas.create_rectangle(left, top, right, bottom, outline="#aaaaaa")
            self.canvas.create_line(left, middle, right, middle, fill="#555555", dash=(4, 3))
            self.canvas.create_text(right, middle - 9, anchor="e", text="reference: 0 cm", fill="#555555")
            for label, y in ((12, top), (0, middle), (-12, bottom)):
                self.canvas.create_line(left - 4, y, left, y, fill="#555555")
                self.canvas.create_text(left - 7, y, anchor="e", text=f"{label:+d}", fill="#555555")
            self.canvas.create_text(left - 43, middle, angle=90, text="CoM error [cm]", fill="#555555")
            self.canvas.create_text(
                left,
                bottom + 13,
                anchor="w",
                text=f"current: {values[-1] * 100.0:+.1f} cm" if values else "current: +0.0 cm",
                fill=color,
            )
            self.canvas.create_text(
                right,
                bottom + 13,
                anchor="e",
                text=f"controller: {control:+.3f} rad",
                fill=color,
            )
            if panel == 1:
                for tick in range(6):
                    x = left + (right - left) * tick / 5.0
                    seconds = -history_seconds + history_seconds * tick / 5.0
                    self.canvas.create_line(x, bottom + 20, x, bottom + 24, fill="#555555")
                    self.canvas.create_text(x, bottom + 36, text=f"{seconds:.0f}", fill="#555555")
                self.canvas.create_text(
                    0.5 * (left + right),
                    bottom + 55,
                    text="time history [s, present = 0]",
                    fill="#555555",
                )
            if len(values) < 2:
                continue
            points = []
            for index, value in enumerate(values):
                x = left + (right - left) * index / (self.HISTORY_LENGTH - 1)
                clipped = max(-self.ERROR_LIMIT_METERS, min(self.ERROR_LIMIT_METERS, value))
                y = middle - clipped * (panel_height * 0.45 / self.ERROR_LIMIT_METERS)
                points.extend((x, y))
            self.canvas.create_line(*points, fill=color, width=2)

def joint_indices(robot: int) -> dict[str, int]:
    """Return URDF joint names and their PyBullet indices."""
    return {
        p.getJointInfo(robot, joint)[1].decode("utf-8"): joint
        for joint in range(p.getNumJoints(robot))
        if p.getJointInfo(robot, joint)[2] != p.JOINT_FIXED
    }


def clamp(robot: int, joint: int, target: float) -> float:
    """Keep a requested position inside the URDF joint range."""
    lower, upper = (float(value) for value in p.getJointInfo(robot, joint)[8:10])
    return target if lower >= upper else max(lower, min(upper, target))


def center_of_mass(robot: int) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return mass-weighted whole-body CoM position and linear velocity."""
    position_sum = [0.0, 0.0, 0.0]
    velocity_sum = [0.0, 0.0, 0.0]
    total_mass = 0.0
    for link in range(-1, p.getNumJoints(robot)):
        mass = float(p.getDynamicsInfo(robot, link)[0])
        if mass <= 0.0:
            continue
        if link == -1:
            position = p.getBasePositionAndOrientation(robot)[0]
            velocity = p.getBaseVelocity(robot)[0]
        else:
            state = p.getLinkState(robot, link, computeLinkVelocity=True)
            position, velocity = state[0], state[6]
        for axis in range(3):
            position_sum[axis] += mass * float(position[axis])
            velocity_sum[axis] += mass * float(velocity[axis])
        total_mass += mass
    return (
        tuple(value / total_mass for value in position_sum),
        tuple(value / total_mass for value in velocity_sum),
    )


def support_center(robot: int, indices: dict[str, int]) -> tuple[float, float, float]:
    """Use the midpoint between the two ankle-roll links as the CoM reference."""
    left = p.getLinkState(robot, indices[LEFT_ANKLE], computeForwardKinematics=True)[0]
    right = p.getLinkState(robot, indices[RIGHT_ANKLE], computeForwardKinematics=True)[0]
    return tuple(0.5 * (left[axis] + right[axis]) for axis in range(3))


def load_scene() -> tuple[int, dict[str, int], int]:
    """Create the physical floor, free-base RPO, fixed table, and free ball."""
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0.0, 0.0, -9.81)
    p.setTimeStep(TIME_STEP)
    p.setPhysicsEngineParameter(
        fixedTimeStep=TIME_STEP,
        numSolverIterations=220,
        enableConeFriction=1,
    )

    floor = p.loadURDF("plane.urdf")
    p.changeDynamics(floor, -1, lateralFriction=1.45, spinningFriction=0.10, rollingFriction=0.03)

    robot = p.loadURDF(
        str(RPO_URDF),
        basePosition=ROBOT_BASE_POSITION,
        useFixedBase=False,
        flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
    )
    indices = joint_indices(robot)
    for name, joint in indices.items():
        p.resetJointState(robot, joint, clamp(robot, joint, STANDING_POSTURE.get(name, 0.0)))

    # This base height and the leg angles above come from a double-support
    # leg-IK solution with both RPO soles on the physical plane.
    p.resetBaseVelocity(robot, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    for link in range(-1, p.getNumJoints(robot)):
        p.changeDynamics(
            robot,
            link,
            lateralFriction=1.45,
            spinningFriction=0.10,
            rollingFriction=0.03,
            linearDamping=0.08,
            angularDamping=0.18,
            frictionAnchor=1,
        )

    table_top = p.createCollisionShape(p.GEOM_BOX, halfExtents=TABLE_HALF_EXTENTS)
    table_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=TABLE_HALF_EXTENTS,
        rgbaColor=(0.62, 0.34, 0.12, 1.0),
    )
    p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=table_top,
        baseVisualShapeIndex=table_visual,
        basePosition=TABLE_POSITION,
    )

    # Four physical legs support the table and participate in contact checks.
    leg_half_extents = (0.018, 0.018, 0.5 * (TABLE_POSITION[2] - TABLE_HALF_EXTENTS[2]))
    leg_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=leg_half_extents)
    leg_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=leg_half_extents,
        rgbaColor=(0.38, 0.18, 0.06, 1.0),
    )
    leg_height = leg_half_extents[2]
    for x_sign in (-1.0, 1.0):
        for y_sign in (-1.0, 1.0):
            p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=leg_collision,
                baseVisualShapeIndex=leg_visual,
                basePosition=(
                    TABLE_POSITION[0] + x_sign * (TABLE_HALF_EXTENTS[0] - leg_half_extents[0]),
                    TABLE_POSITION[1] + y_sign * (TABLE_HALF_EXTENTS[1] - leg_half_extents[1]),
                    leg_height,
                ),
            )

    ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=BALL_RADIUS)
    ball_visual = p.createVisualShape(
        p.GEOM_SPHERE, radius=BALL_RADIUS, rgbaColor=(0.90, 0.06, 0.05, 1.0)
    )
    ball = p.createMultiBody(
        baseMass=0.075,
        baseCollisionShapeIndex=ball_shape,
        baseVisualShapeIndex=ball_visual,
        basePosition=BALL_POSITION,
    )
    p.changeDynamics(ball, -1, lateralFriction=0.55, rollingFriction=0.01, restitution=0.05)

    return robot, indices, ball


def balanced_targets(
    robot: int,
    indices: dict[str, int],
    previous_sagittal: float,
    previous_lateral: float,
) -> tuple[dict[str, float], float, float]:
    """Generate filtered CoM/torso feedback targets without commanding the right arm.

    u_x = 3.8 e_x + 1.05 xdot_com + 0.35 pitch + 0.08 omega_y
    u_y = -2.2 e_y - 0.85 ydot_com - 0.30 roll - 0.07 omega_x
    """
    com, velocity = center_of_mass(robot)
    support = support_center(robot, indices)
    _, orientation = p.getBasePositionAndOrientation(robot)
    angular_velocity = p.getBaseVelocity(robot)[1]
    roll, pitch, _ = p.getEulerFromQuaternion(orientation)

    error_x, error_y = com[0] - support[0], com[1] - support[1]
    if abs(error_x) < 0.004:
        error_x = 0.0
    if abs(error_y) < 0.004:
        error_y = 0.0

    requested_x = max(-0.12, min(0.12, 3.8 * error_x + 1.05 * velocity[0] + 0.35 * pitch + 0.08 * angular_velocity[1]))
    requested_y = max(-0.08, min(0.08, -2.2 * error_y - 0.85 * velocity[1] - 0.30 * roll - 0.07 * angular_velocity[0]))
    sagittal = 0.97 * previous_sagittal + 0.03 * requested_x
    lateral = 0.97 * previous_lateral + 0.03 * requested_y

    targets = STANDING_POSTURE.copy()
    for side in ("left", "right"):
        targets[f"{side}_ankle_pitch_joint"] += sagittal
        targets[f"{side}_thigh_pitch_joint"] -= 0.42 * sagittal
        targets[f"{side}_ankle_roll_joint"] += lateral
        targets[f"{side}_thigh_roll_joint"] -= 0.30 * lateral
    targets["torso_joint"] = max(-0.10, min(0.10, 0.35 * lateral))
    for name in RIGHT_ARM_JOINTS:
        targets.pop(name, None)
    return targets, sagittal, lateral


def arm_targets(
    robot: int,
    indices: dict[str, int],
    target: tuple[float, float, float],
) -> dict[str, float]:
    """Use right-arm-only damped Jacobian IK to place the visible wrist point."""
    hand = indices[RIGHT_HAND]
    def wrist_position() -> np.ndarray:
        state = p.getLinkState(robot, hand, computeForwardKinematics=True)
        position, _ = p.multiplyTransforms(
            state[4],
            state[5],
            WRIST_POINT_OFFSET,
            (0.0, 0.0, 0.0, 1.0),
        )
        return np.asarray(position, dtype=float)

    arm_states = {
        name: p.getJointState(robot, indices[name])
        for name in RIGHT_ARM_JOINTS
    }
    current_wrist = wrist_position()
    error = np.asarray(target, dtype=float) - current_wrist

    # Measure the wrist-point displacement caused by each arm joint directly.
    # This avoids mixing PyBullet's inertial-link Jacobian frame with the RPO
    # link frame used by the visual wrist point.
    columns = []
    for name in RIGHT_ARM_JOINTS:
        joint = indices[name]
        state = arm_states[name]
        p.resetJointState(
            robot,
            joint,
            float(state[0]) + ARM_JACOBIAN_EPSILON,
            float(state[1]),
        )
        columns.append((wrist_position() - current_wrist) / ARM_JACOBIAN_EPSILON)
        p.resetJointState(robot, joint, float(state[0]), float(state[1]))
    jacobian = np.column_stack(columns)

    damping = ARM_JACOBIAN_DAMPING
    delta = jacobian.T @ np.linalg.solve(
        jacobian @ jacobian.T + damping * damping * np.eye(3),
        ARM_JACOBIAN_GAIN * error,
    )

    targets: dict[str, float] = {}
    for name, change in zip(RIGHT_ARM_JOINTS, delta):
        joint = indices[name]
        measured = float(arm_states[name][0])
        bounded_change = max(-ARM_MAX_STEP, min(ARM_MAX_STEP, float(change)))
        targets[name] = clamp(robot, joint, measured + bounded_change)
    return targets


def command_posture(robot: int, indices: dict[str, int], targets: dict[str, float]) -> None:
    """Send lower-body balance and right-arm task targets to physical position motors."""
    for name, joint in indices.items():
        if name not in targets:
            continue
        arm_joint = name in RIGHT_ARM_JOINTS
        p.setJointMotorControl2(
            robot,
            joint,
            p.POSITION_CONTROL,
            targetPosition=clamp(robot, joint, targets[name]),
            force=55.0 if arm_joint else 120.0,
            positionGain=0.55 if arm_joint else 0.36,
            velocityGain=1.40 if arm_joint else 1.55,
        )


def return_to_rest_targets(
    start: dict[str, float],
    blend: float,
) -> dict[str, float]:
    """Blend the right arm smoothly from its pushed pose back to rest."""
    return {
        name: (1.0 - blend) * start[name] + blend * STANDING_POSTURE[name]
        for name in RIGHT_ARM_JOINTS
    }


def task_target(elapsed: float) -> tuple[float, float, float]:
    """Plan a ball-relative approach, contact, and push for the contact pad."""
    ball = BALL_POSITION
    direction_length = math.sqrt(sum(component * component for component in PUSH_DIRECTION))
    if direction_length == 0.0:
        raise ValueError("PUSH_DIRECTION must be non-zero.")
    direction = tuple(component / direction_length for component in PUSH_DIRECTION)
    contact_offset = BALL_RADIUS + WRIST_CLEARANCE
    approach = tuple(
        ball[axis] - direction[axis] * (contact_offset + APPROACH_DISTANCE)
        for axis in range(3)
    )
    contact = tuple(
        ball[axis] - direction[axis] * contact_offset
        for axis in range(3)
    )
    push = tuple(contact[axis] + direction[axis] * PUSH_DISTANCE for axis in range(3))
    if elapsed < 2.0:
        return approach
    if elapsed < 2.1:
        blend = (elapsed - 2.0) / 0.1
        return tuple(
            (1.0 - blend) * start + blend * finish
            for start, finish in zip(approach, push)
        )
    return push


def key_pressed(events: dict[int, int], key: str) -> bool:
    return bool(events.get(ord(key), 0) & p.KEY_WAS_TRIGGERED)


def run(gui: bool = True, duration: float | None = None) -> None:
    """Run the complete free-base RPO pushing trial."""
    if not RPO_URDF.exists():
        raise FileNotFoundError(f"RPO URDF not found: {RPO_URDF}")
    connection = p.GUI if gui else p.DIRECT
    if p.connect(connection, options="--width=900 --height=700") < 0:
        raise RuntimeError("PyBullet could not open a connection.")

    paused = False
    elapsed = 0.0
    sagittal = lateral = 0.0
    step_count = 0
    return_start_targets: dict[str, float] | None = None
    monitor = BalanceMonitor() if gui else None
    try:
        robot, indices, _ = load_scene()
        if gui:
            p.resetDebugVisualizerCamera(2.0, 48.0, -18.0, (0.25, -0.05, 0.45))
            print("Free-base RPO ball push. E: pause/resume | R: restart | Q: quit")

        while duration is None or elapsed < duration:
            if gui:
                monitor.update()
                events = p.getKeyboardEvents()
                if key_pressed(events, "q"):
                    break
                if key_pressed(events, "e"):
                    paused = not paused
                if key_pressed(events, "r"):
                    robot, indices, _ = load_scene()
                    elapsed = sagittal = lateral = 0.0
                    step_count = 0
                    return_start_targets = None
                    monitor.clear()

            if not paused:
                standing, sagittal, lateral = balanced_targets(robot, indices, sagittal, lateral)
                if elapsed < RETURN_START_TIME:
                    return_start_targets = None
                    standing.update(arm_targets(robot, indices, task_target(elapsed)))
                else:
                    if return_start_targets is None:
                        return_start_targets = {
                            name: float(p.getJointState(robot, indices[name])[0])
                            for name in RIGHT_ARM_JOINTS
                        }
                    return_blend = min(1.0, (elapsed - RETURN_START_TIME) / RETURN_DURATION)
                    standing.update(return_to_rest_targets(return_start_targets, return_blend))
                command_posture(robot, indices, standing)
                p.stepSimulation()
                elapsed += TIME_STEP
                step_count += 1
                if monitor is not None and step_count % BALANCE_MONITOR_SAMPLE_STEPS == 0:
                    com, _ = center_of_mass(robot)
                    support = support_center(robot, indices)
                    monitor.add(
                        com[0] - support[0],
                        com[1] - support[1],
                        sagittal,
                        lateral,
                    )

            if gui:
                time.sleep(TIME_STEP)
    finally:
        if monitor is not None:
            monitor.close()
        p.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct", action="store_true", help="Run without the PyBullet GUI.")
    parser.add_argument("--duration", type=float, default=None, help="Optional simulation duration in seconds.")
    args = parser.parse_args()
    run(gui=not args.direct, duration=args.duration)


if __name__ == "__main__":
    main()
