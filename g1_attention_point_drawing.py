"""Draw on a canvas and have a standing G1 trace the path with its right hand.

The canvas is normalized to a 30 x 30 cm vertical plane in front of the robot.
The G1 base is free. A CoM standing controller keeps the lower body in its
standing posture while the right arm follows the points drawn on the canvas.

Controls: E pause/resume, R restart, Q close the PyBullet window.
"""

from __future__ import annotations

import math
import time
import tkinter as tk
from collections import deque
from pathlib import Path

try:
    import pybullet as p
    import pybullet_data
except ImportError as error:  # pragma: no cover
    raise SystemExit("Install PyBullet first: python3 -m pip install pybullet") from error

ROOT = Path(__file__).resolve().parent
G1_URDF = ROOT / "g1_description" / "g1_29dof.urdf"
TIME_STEP = 1.0 / 240.0
FLOOR_CLEARANCE = 0.003
CANVAS_PIXELS = 420
CANVAS_SIZE_METERS = 0.30
DRAWING_PLANE_DISTANCE = 0.38
DRAWING_PLANE_RIGHT_OFFSET = -0.15
DRAWING_PLANE_HEIGHT = 0.29
TRACE_SAMPLE_STEPS = 1
BALANCE_MONITOR_SAMPLE_STEPS = 4
IK_BLEND = 0.20
POINT_DURATION = 0.015
DRAWING_COM_LIMIT_X = 0.08
DRAWING_COM_LIMIT_Y = 0.10
DRAWING_BALANCE_FILTER = 0.97
BALANCE_DEADBAND = 0.004
RIGHT_ARM_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

STANDING_POSTURE = {
    "left_hip_pitch_joint": -0.28,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.62,
    "left_ankle_pitch_joint": -0.30,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.28,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.62,
    "right_ankle_pitch_joint": -0.30,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.15,
    "left_shoulder_roll_joint": 0.12,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.0,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.15,
    "right_shoulder_roll_joint": -0.12,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}

def support_center(robot: int, indices: dict[str, int]) -> tuple[float, float, float]:
    left = p.getLinkState(robot, indices["left_ankle_roll_joint"], computeForwardKinematics=True)[0]
    right = p.getLinkState(robot, indices["right_ankle_roll_joint"], computeForwardKinematics=True)[0]
    return tuple(0.5 * (left[axis] + right[axis]) for axis in range(3))

def right_fingertip(robot: int) -> tuple[float, float, float]:
    """Return a virtual tip at the end of the rigid right-hand visual mesh."""
    hand_link = next(
        joint
        for joint in range(p.getNumJoints(robot))
        if p.getJointInfo(robot, joint)[12].decode("utf-8") == "right_rubber_hand"
    )
    state = p.getLinkState(robot, hand_link, computeForwardKinematics=True)
    tip, _ = p.multiplyTransforms(
        state[4], state[5], (0.115, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)
    )
    return tuple(float(value) for value in tip)

def center_of_mass(robot: int) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    base_position, _ = p.getBasePositionAndOrientation(robot)
    base_velocity, _ = p.getBaseVelocity(robot)
    base_mass = float(p.getDynamicsInfo(robot, -1)[0])
    position_sum = [base_mass * value for value in base_position]
    velocity_sum = [base_mass * value for value in base_velocity]
    total_mass = base_mass

    for link in range(p.getNumJoints(robot)):
        mass = float(p.getDynamicsInfo(robot, link)[0])
        if mass <= 0.0:
            continue
        state = p.getLinkState(robot, link, computeLinkVelocity=True)
        for axis in range(3):
            position_sum[axis] += mass * float(state[0][axis])
            velocity_sum[axis] += mass * float(state[6][axis])
        total_mass += mass

    return (
        tuple(value / total_mass for value in position_sum),
        tuple(value / total_mass for value in velocity_sum),
    )

def movable_joints(robot: int) -> dict[str, int]:
    return {
        p.getJointInfo(robot, joint)[1].decode("utf-8"): joint
        for joint in range(p.getNumJoints(robot))
        if p.getJointInfo(robot, joint)[2] != p.JOINT_FIXED
    }

def clamp_to_joint_range(robot: int, joint: int, target: float) -> float:
    """Return a target inside the position range declared by the URDF."""
    info = p.getJointInfo(robot, joint)
    lower, upper = float(info[8]), float(info[9])
    if lower >= upper:
        return target
    return max(lower, min(upper, target))

def configure_scene(
    fixed_base: bool = False,
) -> tuple[int, dict[str, int]]:
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0.0, 0.0, -9.81)
    p.setTimeStep(TIME_STEP)
    p.setPhysicsEngineParameter(fixedTimeStep=TIME_STEP, numSolverIterations=200, enableConeFriction=1)
    plane = p.loadURDF("plane.urdf")
    p.changeDynamics(plane, -1, lateralFriction=1.3, spinningFriction=0.08, rollingFriction=0.1)
    p.setAdditionalSearchPath(str(G1_URDF.parent))
    robot = p.loadURDF(
        str(G1_URDF),
        useFixedBase=fixed_base,
        flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
    )
    indices = movable_joints(robot)
    for name, joint in indices.items():
        p.resetJointState(robot, joint, clamp_to_joint_range(robot, joint, STANDING_POSTURE.get(name, 0.0)))
    p.resetBasePositionAndOrientation(robot, (0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0))
    lowest = min(p.getAABB(robot, link)[0][2] for link in range(-1, p.getNumJoints(robot)))
    p.resetBasePositionAndOrientation(robot, (0.0, 0.0, 1.0 - lowest + FLOOR_CLEARANCE), (0.0, 0.0, 0.0, 1.0))
    for link in range(-1, p.getNumJoints(robot)):
        p.changeDynamics(robot, link, lateralFriction=1.3, spinningFriction=0.08, rollingFriction=0.1,
                         linearDamping=0.0, angularDamping=0.0)
    return robot, indices

class DrawingCanvas:
    """Small Tk canvas that records a normalized drawing path."""

    def __init__(self, robot_name: str = "G1", size_centimeters: int = 30) -> None:
        self.root = tk.Tk()
        self.root.title(f"{robot_name} Drawing Canvas - {size_centimeters} x {size_centimeters} cm")
        self.root.resizable(False, False)
        self.canvas = tk.Canvas(
            self.root,
            width=CANVAS_PIXELS,
            height=CANVAS_PIXELS,
            background="white",
            highlightthickness=1,
            highlightbackground="#666666",
        )
        self.canvas.pack(padx=12, pady=(12, 6))
        tk.Label(
            self.root,
            text=f"Draw with the mouse. The robot traces the path on a {size_centimeters} x {size_centimeters} cm plane.",
        ).pack(padx=12, pady=(0, 6))
        tk.Label(self.root, text="Close this window when you are finished drawing.").pack(padx=12, pady=(0, 12))

        self.points: list[tuple[float, float]] = []
        self._last_canvas_point: tuple[float, float] | None = None
        self.closed = False
        self.canvas.bind("<Button-1>", self.begin_stroke)
        self.canvas.bind("<B1-Motion>", self.extend_stroke)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def normalized_point(self, event: tk.Event) -> tuple[float, float]:
        x = max(0.0, min(1.0, event.x / CANVAS_PIXELS))
        y = max(0.0, min(1.0, event.y / CANVAS_PIXELS))
        return x, y

    def begin_stroke(self, event: tk.Event) -> None:
        point = self.normalized_point(event)
        self.points.append(point)
        self._last_canvas_point = (event.x, event.y)

    def extend_stroke(self, event: tk.Event) -> None:
        point = self.normalized_point(event)
        if self._last_canvas_point is not None:
            self.canvas.create_line(
                self._last_canvas_point[0],
                self._last_canvas_point[1],
                event.x,
                event.y,
                fill="#c62828",
                width=3,
                capstyle=tk.ROUND,
            )
        self.points.append(point)
        self._last_canvas_point = (event.x, event.y)

    def update(self) -> None:
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.closed = True

    def clear(self) -> None:
        """Erase the visible drawing and discard its recorded path."""
        self.canvas.delete("all")
        self.points.clear()
        self._last_canvas_point = None

    def close(self) -> None:
        self.closed = True
        self.root.destroy()


class BalanceMonitor:
    """Live CoM-error histories relative to the double-support reference."""

    HISTORY_LENGTH = 300
    ERROR_LIMIT_METERS = 0.12
    HISTORY_SECONDS = HISTORY_LENGTH * BALANCE_MONITOR_SAMPLE_STEPS * TIME_STEP

    def __init__(self, parent: tk.Tk) -> None:
        self.window = tk.Toplevel(parent)
        self.window.title("G1 Balance Monitor")
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
                for second in range(6):
                    x = left + (right - left) * second / 5.0
                    self.canvas.create_line(x, bottom + 20, x, bottom + 24, fill="#555555")
                    self.canvas.create_text(
                        x,
                        bottom + 36,
                        text=f"{second - 5}",
                        fill="#555555",
                    )
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


def drawing_target(
    robot: int,
    normalized: tuple[float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Map normalized canvas coordinates to a 30 cm plane in front of the G1."""
    base_position, base_orientation = p.getBasePositionAndOrientation(robot)
    yaw = p.getEulerFromQuaternion(base_orientation)[2]
    forward = (math.cos(yaw), math.sin(yaw), 0.0)
    left = (-math.sin(yaw), math.cos(yaw), 0.0)
    lateral = DRAWING_PLANE_RIGHT_OFFSET + (normalized[0] - 0.5) * CANVAS_SIZE_METERS
    vertical = (0.5 - normalized[1]) * CANVAS_SIZE_METERS
    tip = (
        base_position[0] + DRAWING_PLANE_DISTANCE * forward[0] + lateral * left[0],
        base_position[1] + DRAWING_PLANE_DISTANCE * forward[1] + lateral * left[1],
        base_position[2] + DRAWING_PLANE_HEIGHT + vertical,
    )
    return tip, p.getQuaternionFromEuler((0.0, 0.0, yaw))


def right_hand_link(robot: int) -> int:
    return next(
        joint
        for joint in range(p.getNumJoints(robot))
        if p.getJointInfo(robot, joint)[12].decode("utf-8") == "right_rubber_hand"
    )


def hand_drawing_controller(
    robot: int,
    indices: dict[str, int],
    target_tip: tuple[float, float, float],
    target_orientation: tuple[float, float, float, float],
) -> dict[str, float]:
    """Convert one canvas point into right-arm targets, independently of balance."""
    hand = right_hand_link(robot)
    hand_position = p.getLinkState(robot, hand, computeForwardKinematics=True)[4]
    current_tip = right_fingertip(robot)
    target_hand = tuple(
        target_tip[axis] - (current_tip[axis] - hand_position[axis])
        for axis in range(3)
    )
    solution = p.calculateInverseKinematics(
        robot,
        hand,
        targetPosition=target_hand,
        maxNumIterations=80,
        residualThreshold=0.002,
    )
    movable = list(indices.values())
    result: dict[str, float] = {}
    for name in RIGHT_ARM_JOINTS:
        joint = indices[name]
        solution_index = movable.index(joint)
        measured = float(p.getJointState(robot, joint)[0])
        desired = clamp_to_joint_range(robot, joint, float(solution[solution_index]))
        result[name] = (1.0 - IK_BLEND) * measured + IK_BLEND * desired
    return result


def drawing_balanced_targets(
    robot: int,
    indices: dict[str, int],
    filtered_sagittal: float,
    filtered_lateral: float,
) -> tuple[dict[str, float], float, float]:
    """Create a deliberately damped standing reference for arm drawing."""
    com, velocity = center_of_mass(robot)
    support = support_center(robot, indices)
    _, base_orientation = p.getBasePositionAndOrientation(robot)
    angular_velocity = p.getBaseVelocity(robot)[1]
    roll, pitch, _ = p.getEulerFromQuaternion(base_orientation)

    error_x = com[0] - support[0]
    error_y = com[1] - support[1]
    if abs(error_x) < BALANCE_DEADBAND:
        error_x = 0.0
    if abs(error_y) < BALANCE_DEADBAND:
        error_y = 0.0

    requested_sagittal = max(
        -0.13,
        min(0.13, 3.8 * error_x + 1.05 * velocity[0] + 0.35 * pitch + 0.08 * angular_velocity[1]),
    )
    requested_lateral = max(
        -0.08,
        min(-2.2 * error_y - 0.85 * velocity[1] - 0.30 * roll - 0.07 * angular_velocity[0], 0.08),
    )
    sagittal = (
        DRAWING_BALANCE_FILTER * filtered_sagittal
        + (1.0 - DRAWING_BALANCE_FILTER) * requested_sagittal
    )
    lateral = (
        DRAWING_BALANCE_FILTER * filtered_lateral
        + (1.0 - DRAWING_BALANCE_FILTER) * requested_lateral
    )

    targets = STANDING_POSTURE.copy()
    for side in ("left", "right"):
        targets[f"{side}_ankle_pitch_joint"] += sagittal
        targets[f"{side}_hip_pitch_joint"] -= 0.42 * sagittal
        targets[f"{side}_ankle_roll_joint"] += lateral
        targets[f"{side}_hip_roll_joint"] -= 0.32 * lateral
    targets["waist_pitch_joint"] = max(-0.10, min(0.10, 0.35 * sagittal))
    targets["waist_roll_joint"] = max(-0.07, min(0.07, 0.28 * lateral))
    for name in RIGHT_ARM_JOINTS:
        targets.pop(name, None)
    return targets, sagittal, lateral


def command_drawing_targets(
    robot: int,
    indices: dict[str, int],
    targets: dict[str, float],
) -> None:
    """Track drawing targets with extra velocity damping."""
    for name, joint in indices.items():
        if name in RIGHT_ARM_JOINTS and name not in targets:
            continue
        effort_limit = max(0.0, float(p.getJointInfo(robot, joint)[10]))
        drawing_joint = name in RIGHT_ARM_JOINTS
        p.setJointMotorControl2(
            robot,
            joint,
            p.POSITION_CONTROL,
            targetPosition=clamp_to_joint_range(robot, joint, targets.get(name, 0.0)),
            force=effort_limit,
            positionGain=0.34 if drawing_joint else 0.18,
            velocityGain=1.10 if drawing_joint else 1.45,
        )


def add_drawing_damping(robot: int) -> None:
    """Reduce free-base oscillation without fixing or assisting the base."""
    for link in range(-1, p.getNumJoints(robot)):
        p.changeDynamics(
            robot,
            link,
            lateralFriction=1.6,
            spinningFriction=0.12,
            rollingFriction=0.02,
            linearDamping=0.04,
            angularDamping=0.10,
            frictionAnchor=1,
        )


def main() -> None:
    if not G1_URDF.exists():
        raise FileNotFoundError(f"G1 URDF not found: {G1_URDF}")
    if p.connect(p.GUI, options="--width=900 --height=700") < 0:
        raise RuntimeError("PyBullet could not open its GUI.")

    drawing = DrawingCanvas()
    monitor = BalanceMonitor(drawing.root)
    try:
        robot, indices = configure_scene()
        add_drawing_damping(robot)
        p.resetDebugVisualizerCamera(2.8, 45.0, -18.0, (0.0, 0.0, 0.70))
        sagittal = lateral = 0.0
        queued_points: deque[tuple[float, float]] = deque()
        consumed_points = 0
        active_point: tuple[float, float] | None = None
        point_elapsed = 0.0
        holding_last_point = False
        previous_tip: tuple[float, float, float] | None = None
        paused = False
        step_count = 0
        print("Draw in the canvas while the CoM standing controller keeps the G1 upright.")

        while p.isConnected() and not drawing.closed:
            drawing.update()
            if drawing.closed:
                break
            keys = p.getKeyboardEvents()
            if ord("q") in keys and keys[ord("q")] & p.KEY_WAS_TRIGGERED:
                break
            if ord("e") in keys and keys[ord("e")] & p.KEY_WAS_TRIGGERED:
                paused = not paused
            if ord("r") in keys and keys[ord("r")] & p.KEY_WAS_TRIGGERED:
                robot, indices = configure_scene()
                add_drawing_damping(robot)
                drawing.clear()
                monitor.clear()
                sagittal = lateral = 0.0
                queued_points.clear()
                consumed_points = 0
                active_point = previous_tip = None
                point_elapsed = 0.0
                holding_last_point = False

            if not paused:
                if len(drawing.points) > consumed_points:
                    queued_points.extend(drawing.points[consumed_points:])
                    consumed_points = len(drawing.points)
                    if holding_last_point:
                        holding_last_point = False
                        point_elapsed = POINT_DURATION
                if active_point is None and queued_points:
                    active_point = queued_points.popleft()
                    point_elapsed = 0.0

                standing_targets, sagittal, lateral = drawing_balanced_targets(
                    robot, indices, sagittal, lateral
                )
                if active_point is not None:
                    desired_tip, desired_orientation = drawing_target(robot, active_point)
                    drawing_targets = hand_drawing_controller(
                        robot, indices, desired_tip, desired_orientation
                    )
                    standing_targets.update(drawing_targets)
                    if not holding_last_point:
                        point_elapsed += TIME_STEP
                        if point_elapsed >= POINT_DURATION:
                            if queued_points:
                                active_point = queued_points.popleft()
                                point_elapsed = 0.0
                            else:
                                holding_last_point = True
                command_drawing_targets(robot, indices, standing_targets)
                p.stepSimulation()
                step_count += 1
                if step_count % BALANCE_MONITOR_SAMPLE_STEPS == 0:
                    com, _ = center_of_mass(robot)
                    support = support_center(robot, indices)
                    monitor.add(
                        com[0] - support[0],
                        com[1] - support[1],
                        sagittal,
                        lateral,
                    )
                if (
                    active_point is not None
                    and not holding_last_point
                    and step_count % TRACE_SAMPLE_STEPS == 0
                ):
                    current_tip = right_fingertip(robot)
                    if previous_tip is not None:
                        p.addUserDebugLine(
                            previous_tip,
                            current_tip,
                            (0.90, 0.05, 0.05),
                            lineWidth=3.0,
                            lifeTime=0.0,
                        )
                    previous_tip = current_tip
                elif active_point is None:
                    previous_tip = None
            time.sleep(TIME_STEP)
    finally:
        if p.isConnected():
            p.disconnect()
        if not drawing.closed:
            drawing.close()


if __name__ == "__main__":
    main()
